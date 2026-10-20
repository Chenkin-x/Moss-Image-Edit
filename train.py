"""Four-GPU LoRA: reference-only FA4 mask, DeepSpeed Muon/ZeRO-2, raw paired data."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam

from moss_edit.lora import inject_lora, save_adapter
from moss_edit.components import load_transformer, model_metadata, read_config
from moss_edit.attention import MossVarlenEditModel
from moss_edit.data import MixedEdits, MixedBatchSampler, keep_list, init_worker
from moss_edit.encoding import FrozenPlacement, OnlineEncoder, release_cuda


def flow_loss(model, samples, config, generator, device):
    xs = [s['target_latent'].to(device,dtype=torch.float32,non_blocking=True) for s in samples]
    noise = [torch.randn(x.shape,device=device,generator=generator) for x in xs]
    t = torch.randn(len(xs),device=device,generator=generator)
    t = (t*config['time_std']+config['time_mean']).sigmoid().clamp(1e-5,1-1e-5)
    drops = (torch.rand(len(xs),device=device,generator=generator)<config['cond_dropout']).tolist()
    contexts = [s['null_prompt_embeds' if drop else 'prompt_embeds'].to(device,non_blocking=True)
                for s,drop in zip(samples,drops)]
    refs = [s['reference_latents'][0].to(device,non_blocking=True) for s in samples]
    noisy = [((1-ti)*x+ti*n).to(torch.bfloat16) for x,n,ti in zip(xs,noise,t)]
    predictions = model(noisy,t,contexts,refs)
    # Equal weight per editing pair, irrespective of resolution/text length.
    losses = torch.stack([(p.float()-(n-x)).square().mean() for p,n,x in zip(predictions,noise,xs)])
    return losses.mean(), t.detach().mean()


@torch.no_grad()
def sample_and_log(engine, encoder, sample, raw, config, step, rank, run):
    from PIL import Image, ImageDraw
    from moss_edit.components import Condition, load_moss_module
    from moss_edit.processing import decode_latents
    from moss_edit.sampling import sample_latents
    editor = engine.module
    editor.eval()
    device = engine.device
    positive = sample['prompt_embeds'][None].to(device)
    negative = sample['null_prompt_embeds'][None].to(device)
    pos = Condition(positive,torch.ones(positive.shape[:2],device=device,dtype=torch.bool))
    neg = Condition(negative,torch.ones(negative.shape[:2],device=device,dtype=torch.bool))
    height,width = raw['target'].shape[-2:]
    scheduler = load_moss_module(config['snapshot_dir']).MOSSImageScheduler.from_pretrained(
        Path(config['snapshot_dir'])/'scheduler',local_files_only=True,shift=config['inference_shift'])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        latent = sample_latents(editor,scheduler,[sample['reference_latents'][0][None].to(device)],pos,neg,
            height=height,width=width,steps=config['sample_steps'],cfg=config['sample_cfg'],
            seed=config['sample_seed']+rank,device=device,use_kv_cache=False)
    pixels = ((decode_latents(encoder.vae,latent)[0].float().cpu()+1)*127.5).round().clamp(0,255).byte()
    panels = [raw['source'],pixels,raw['target']]
    board = Image.new('RGB',(3*width,height+24),'white')
    draw = ImageDraw.Draw(board)
    for i,(name,panel) in enumerate(zip(['source','generated','target'],panels)):
        board.paste(Image.fromarray(panel.permute(1,2,0).numpy()),(i*width,24))
        draw.text((i*width+5,4),name,fill='black')
    folder = Path(config['output_dir'])/'samples'/f'step-{step:08d}'
    folder.mkdir(parents=True,exist_ok=True)
    board.save(folder/f'{rank:02d}.png')
    (folder/f'{rank:02d}.json').write_text(json.dumps({'id':raw['id'],'instruction':raw['instruction'],
        'seed':config['sample_seed']+rank,'step':step},ensure_ascii=False,indent=2))
    dist.barrier()
    if rank == 0 and run is not None:
        import wandb
        images = [wandb.Image(str(folder/f'{i:02d}.png'),caption=json.loads((folder/f'{i:02d}.json').read_text())['instruction'])
                  for i in range(dist.get_world_size())]
        run.log({'samples/editing':images},step=step)
        print(json.dumps({'event':'samples_uploaded','step':step,'images':len(images)}),flush=True)
    elif rank == 0:
        print(json.dumps({'event':'samples_saved','step':step,'path':str(folder)}),flush=True)
    dist.barrier()
    editor.train()
    release_cuda()


def checkpoint(engine, base, config, metadata, step, rank, generator):
    output = Path(config['output_dir'])
    tag = f'checkpoint-{step:08d}'
    engine.save_checkpoint(str(output),tag=tag,client_state={'step':step,'config':config},
                           exclude_frozen_parameters=True)
    torch.save({'noise':generator.get_state(),'cpu':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state()},output/tag/f'rng-rank{rank}.pt')
    if rank == 0:
        save_adapter(base,output/tag/'adapter',metadata)
        print(json.dumps({'event':'checkpoint_saved','step':step,'path':str(output/tag)}),flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = read_config(args.config)
    rank,local = int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    deepspeed.init_distributed(timeout=timedelta(minutes=45))
    world = dist.get_world_size()
    micro,accum = config['micro_batch_size'],config['gradient_accumulation_steps']
    assert micro*accum*world == config['global_batch_size']
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    if rank == 0:
        (output/'resolved_config.json').write_text(json.dumps(config,indent=2))
        print(json.dumps({'event':'loading_model','base':config['snapshot_dir'],'attention':config['attention_backend']}),flush=True)
    base = load_transformer(config['snapshot_dir'],device='cpu',dtype=torch.bfloat16)
    inject_lora(base,rank=config['lora_rank'],alpha=config['lora_alpha'],param_dtype=torch.bfloat16)
    metadata = model_metadata(base,config['snapshot_dir'])
    metadata.update(attention_backend=config['attention_backend'],
                    dit_quantization=config.get('dit_quantization','none'),
                    text_quantization=config.get('text_quantization','none'))
    frozen = None if config.get('keep_models_on_gpu',False) else FrozenPlacement(base,frozen_only=True)
    model = MossVarlenEditModel(base,gradient_checkpointing=config['gradient_checkpointing']).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable_numel = sum(p.numel() for p in params)
    for p in params: p.use_muon=True
    optimizer = MuonWithAuxAdam([dict(params=params,use_muon=True,lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'],momentum=.95,ns_method='gram',
                                     adjust_lr_fn=config['muon_adjust_lr_fn'])])
    ds_config = {'train_batch_size':config['global_batch_size'],'train_micro_batch_size_per_gpu':micro,
        'gradient_accumulation_steps':accum,'bf16':{'enabled':True},'gradient_clipping':0.0,
        'zero_optimization':{'stage':2,'reduce_scatter':True,'overlap_comm':True,'contiguous_gradients':True,
                             'reduce_bucket_size':25000000,'allgather_bucket_size':25000000},
        'zero_allow_untested_optimizer':True,'steps_per_print':1000000000}
    engine,optimizer,_,_ = deepspeed.initialize(model=model,optimizer=optimizer,model_parameters=params,config=ds_config)
    if config.get('dit_quantization') == 'quanto_int8':
        assert frozen is None
        from moss_edit.quantization import quantize_frozen_linears,keep_adapter_state
        quantize_frozen_linears(base,engine)
        model.register_state_dict_post_hook(keep_adapter_state)
        release_cuda()
    step=0
    generator=torch.Generator(device=device).manual_seed(config['seed']+104729*rank)
    if args.resume:
        folder=Path(args.resume);_,state=engine.load_checkpoint(str(folder.parent),tag=folder.name,load_module_strict=False)
        step=state['step'];rng=torch.load(folder/f'rng-rank{rank}.pt',map_location='cpu',weights_only=True)
        generator.set_state(rng['noise']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'])
    # A resumed optimizer can carry the old parameter-group options.
    for group in engine.optimizer.param_groups:
        group['adjust_lr_fn'] = config['muon_adjust_lr_fn']
    assert engine.optimizer.clip_grad == 0
    if rank == 0:
        print(json.dumps({'event':'optimizer_ready','optimizer':engine.optimizer.optimizer.__class__.__module__,
            'muon_adjust_lr_fn':config['muon_adjust_lr_fn'],'gradient_clipping':0.0}),flush=True)
    data=MixedEdits(config)
    sampler=MixedBatchSampler(data,rank=rank,world=world,batch_size=micro,steps=config['max_steps'],
                              accumulation=accum,seed=config['seed'],start_step=step)
    loader=DataLoader(data,batch_sampler=sampler,num_workers=config['num_workers'],collate_fn=keep_list,
                      pin_memory=True,persistent_workers=True,prefetch_factor=config['prefetch_factor'],
                      multiprocessing_context='spawn',worker_init_fn=init_worker)
    stream=iter(loader)
    # DiT and Qwen are alternated on GPU to fit unquantized 5B + 9B models on
    # 32GB. Only immutable base weights move; LoRA/ZeRO optimizer stay on GPU.
    if frozen:frozen.move('cpu')
    release_cuda()
    encoder=OnlineEncoder(config,device)
    validation=data[tuple(config['validation_keys'][rank])]
    val_sample=encoder.encode([validation])[0]
    run=None
    if rank == 0 and config['wandb']['enabled']:
        import wandb
        opts=config['wandb']
        run=wandb.init(project=opts['project'],name=opts['name'],config=config,dir=str(output),
                       settings=wandb.Settings(init_timeout=90))
        (output/'wandb_run.json').write_text(json.dumps({'id':run.id,'url':run.url}))
    if rank == 0:
        print(json.dumps({'event':'training_ready','trainable_parameters':sum(p.numel() for p in params),
            'global_batch':micro*accum*world,'noob_samples':data.manifest['noob_count'],
            'edit_samples':data.ends[-1],'wandb':run.url if run else None}),flush=True)
    if frozen:frozen.move(device)
    if config['sample_at_start'] and step == 0:
        sample_and_log(engine,encoder,val_sample,validation,config,step,rank,run)
    stop_step = config['max_steps']
    while step < stop_step:
        nsteps=min(config['encode_steps_per_chunk'],stop_step-step)
        began=time.monotonic()
        raw_batches=[next(stream) for _ in range(nsteps*accum)]
        data_wait=time.monotonic()-began
        began=time.monotonic()
        if frozen:
            frozen.move('cpu');release_cuda()
        encoded=encoder.encode([row for batch in raw_batches for row in batch])
        if frozen:frozen.move(device)
        torch.cuda.synchronize()
        encode_seconds=(time.monotonic()-began)/nsteps
        engine.train()
        for chunk_step in range(nsteps):
            began=time.monotonic()
            lr=config['learning_rate']*min((step+1)/config['warmup_steps'],1.)
            for group in engine.optimizer.param_groups:group['lr']=lr
            loss_sum=torch.zeros((),device=device);t_sum=torch.zeros((),device=device)
            for micro_step in range(accum):
                start=(chunk_step*accum+micro_step)*micro
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,t=flow_loss(engine,encoded[start:start+micro],config,generator,device)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at step {step+1}')
                engine.backward(loss);engine.step()
                loss_sum+=loss.detach();t_sum+=t
            step+=1
            dist.all_reduce(loss_sum);dist.all_reduce(t_sum)
            torch.cuda.synchronize()
            compute=time.monotonic()-began
            # This DeepSpeed Muon path reports the post-orthogonalization update norm,
            # not the raw gradient norm. With clipping off, lr * update is applied.
            update_norm=float(engine.get_global_grad_norm())
            update_rms=update_norm/trainable_numel**.5
            metrics={'step':step,'train/loss':float(loss_sum/(world*accum)),
                     'train/t_mean':float(t_sum/(world*accum)),'train/lr':lr,
                     'train/muon_update_norm':update_norm,'train/muon_update_rms':update_rms,
                     'train/muon_step_rms':lr*update_rms,
                     'perf/train_seconds':compute,'perf/encode_seconds':encode_seconds,
                     'perf/data_wait_seconds':data_wait/nsteps,
                     'perf/step_seconds':compute+encode_seconds+data_wait/nsteps,
                     'perf/peak_gib':torch.cuda.max_memory_allocated()/2**30,
                     'train/images_seen':step*config['global_batch_size']}
            if rank == 0:
                print(json.dumps(metrics),flush=True)
                with (output/'metrics.jsonl').open('a') as file:file.write(json.dumps(metrics)+'\n')
                if run:run.log(metrics,step=step)
            if step%config['save_every']==0 or step==config['max_steps']:
                checkpoint(engine,base,config,metadata,step,rank,generator)
            if step%config['sample_every']==0:
                sample_and_log(engine,encoder,val_sample,validation,config,step,rank,run)
        del encoded,raw_batches
    if run:run.finish()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
