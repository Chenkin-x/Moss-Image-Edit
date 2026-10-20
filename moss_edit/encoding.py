"""Frozen GPU encoding in chunks; retain original CPU weights for cheap offload."""
import gc
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, QuantoConfig

from moss_edit.components import load_moss_module
from moss_edit.processing import encode_image


class FrozenPlacement:
    def __init__(self, module, *, frozen_only=False):
        # Capture CPU storages before the first CUDA move. Offloading restores
        # these immutable tensors rather than copying GPU weights back to RAM.
        self.module = module
        self.parameters = [(p,p.detach()) for p in module.parameters() if not frozen_only or not p.requires_grad]
        self.buffers = [(name,b.detach()) for name,b in module.named_buffers()]

    def move(self, device):
        for parameter, cpu in self.parameters:
            parameter.data = cpu.to(device=device)
        for name,cpu in self.buffers:
            parent,_,leaf = name.rpartition('.')
            self.module.get_submodule(parent)._buffers[leaf] = cpu.to(device=device)


class OnlineEncoder:
    def __init__(self, config, device):
        self.config,self.device = config,device
        root = Path(config['snapshot_dir'])
        self.tokenizer = AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
        self.resident = config.get('keep_models_on_gpu',False)
        options = dict(dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
        if config.get('text_quantization') == 'quanto_int8':
            assert self.resident
            options.update(quantization_config=QuantoConfig(weights='int8'),device_map={'':device.index})
        self.text = AutoModel.from_pretrained(root/'text_encoder',**options).requires_grad_(False).eval()
        self.text_placement = None if self.resident else FrozenPlacement(self.text)
        if self.resident:self.text.to(device)
        self.vae = load_moss_module(root).AutoencoderKLFlux2.from_pretrained(
            root/'vae',torch_dtype=torch.float32,local_files_only=True).requires_grad_(False).eval().to(device)
        self.null = None

    @torch.no_grad()
    def encode_text(self, prompts):
        formatted = [self.tokenizer.apply_chat_template(
            [{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            for text in prompts]
        tokens = self.tokenizer(formatted,return_tensors='pt',padding=True,truncation=True,
                                max_length=self.config['max_sequence_length']).to(self.device)
        result = self.text(**tokens,use_cache=False,return_dict=True).last_hidden_state
        return [x[mask.bool()].detach().cpu().contiguous()
                for x,mask in zip(result,tokens.attention_mask)]

    @torch.no_grad()
    def encode(self, rows):
        if self.text_placement:self.text_placement.move(self.device)
        if self.null is None: self.null = self.encode_text([''])[0]
        text = []
        bs = self.config['text_encode_batch_size']
        for begin in range(0,len(rows),bs):
            text.extend(self.encode_text([r['instruction'] for r in rows[begin:begin+bs]]))
        if self.text_placement:
            self.text_placement.move('cpu')
            torch.cuda.empty_cache()
        samples = []
        for row,prompt in zip(rows,text):
            # The pair has identical geometry. Encode both in one VAE batch;
            # different pairs retain their individual shapes (no bucketing).
            images = torch.stack((row['source'],row['target'])).to(self.device,dtype=torch.float32).div_(127.5).sub_(1)
            z = encode_image(self.vae,images).to(torch.bfloat16).cpu()
            samples.append({'target_latent':z[1], 'reference_latents':[z[0]],
                            'prompt_embeds':prompt, 'null_prompt_embeds':self.null,
                            'metadata':{k:row[k] for k in ('id','instruction','dataset')}})
        return samples


def release_cuda():
    gc.collect()
    torch.cuda.empty_cache()
