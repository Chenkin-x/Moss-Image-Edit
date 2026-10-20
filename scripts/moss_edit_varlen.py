"""Packed editing attention without buckets, padding, or cross-sample attention."""
from functools import partial

import torch
import cutlass
import cutlass.cute as cute
from flash_attn.cute import flash_attn_varlen_func
from flash_attn.cute import utils as fa_utils
from flash_attn.cute.interface import _flash_attn_bwd
from torch.utils.checkpoint import checkpoint

from moss_edit_model import MossEditModel


def varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k):
    # Explicit FA4 backend, including backward. No automatic attention fallback.
    out, _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                   max_seqlen_q=max_q, max_seqlen_k=max_k,
                                   causal=False)
    return out


@cute.jit
def editing_mask_mod(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
    nref = fa_utils.scalar_to_ssa(aux_tensors[0][batch[0]], cutlass.Int32)
    ntext = fa_utils.scalar_to_ssa(aux_tensors[1][batch[0]], cutlass.Int32)
    # Sequence-local indices. cu_seqlens provides the inter-sample boundary.
    # [text, ref, target]: only ref queries are restricted, to the ref interval.
    k_is_ref = (kv_idx >= ntext) & (kv_idx < ntext + nref)
    return (q_idx < ntext) | (q_idx >= ntext + nref) | k_is_ref


class EditAttention(torch.autograd.Function):
    """FA4 mask_mod forward + mathematically equivalent FA4 SM120 backward.

    SM120 currently rejects mask_mod in its backward kernel. This mask has two
    dense rectangles: ref->ref, body->all. Their FA4 backward calls use the
    forward's actual output/LSE; ref K/V receive gradients from BOTH rectangles.
    There is no detached reference path, extra forward, or SDPA fallback.
    """
    @staticmethod
    def forward(ctx,q,k,v,layout):
        out,lse=flash_attn_varlen_func(q,k,v,cu_seqlens_q=layout['cu_all'],cu_seqlens_k=layout['cu_all'],
            max_seqlen_q=layout['max_all'],max_seqlen_k=layout['max_all'],causal=False,
            mask_mod=editing_mask_mod,aux_tensors=[layout['ref_lengths'],layout['text_lengths']],return_lse=True)
        ctx.save_for_backward(q,k,v,out,lse)
        ctx.layout=layout
        return out

    @staticmethod
    def backward(ctx,dout):
        q,k,v,out,lse=ctx.saved_tensors
        layout=ctx.layout;ri,bi=layout['ref_indices'],layout['body_indices']
        scale=q.shape[-1]**-.5
        dqr,dkr,dvr=_flash_attn_bwd(q[ri],k[ri],v[ri],out[ri],dout[ri].contiguous(),lse[:,ri].contiguous(),scale,False,0.,
            cu_seqlens_q=layout['cu_ref'],cu_seqlens_k=layout['cu_ref'],
            max_seqlen_q=layout['max_ref'],max_seqlen_k=layout['max_ref'])
        dqb,dk,dv=_flash_attn_bwd(q[bi],k,v,out[bi],dout[bi].contiguous(),lse[:,bi].contiguous(),scale,False,0.,
            cu_seqlens_q=layout['cu_body'],cu_seqlens_k=layout['cu_all'],
            max_seqlen_q=layout['max_body'],max_seqlen_k=layout['max_all'])
        dq=torch.empty_like(q);dq[ri]=dqr;dq[bi]=dqb
        dk[ri]=(dk[ri].float()+dkr.float()).to(dk.dtype)
        dv[ri]=(dv[ri].float()+dvr.float()).to(dv.dtype)
        return dq,dk,dv,None


def edit_layout(ref_lengths, text_lengths, all_lengths, device):
    ref_indices,body_indices=[],[]
    offset=0
    for nr,nt,n in zip(ref_lengths,text_lengths,all_lengths):
        ref_indices.extend(range(offset+nt,offset+nt+nr))
        body_indices.extend(range(offset,offset+nt))
        body_indices.extend(range(offset+nt+nr,offset+n))
        offset+=n
    body_lengths=[n-r for n,r in zip(all_lengths,ref_lengths)]
    result={k:torch.tensor(v,device=device,dtype=torch.long) for k,v in
            [('ref_indices',ref_indices),('body_indices',body_indices)]}
    result['ref_lengths']=torch.tensor(ref_lengths,device=device,dtype=torch.int32)
    result['text_lengths']=torch.tensor(text_lengths,device=device,dtype=torch.int32)
    for name,lengths in [('all',all_lengths),('ref',ref_lengths),('body',body_lengths)]:
        result['cu_'+name]=cumulative(lengths,device);result['max_'+name]=max(lengths)
    return result


def cumulative(lengths, device):
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)


class MossVarlenEditModel(MossEditModel):
    def _times(self, times):
        m = self.base_model
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = (times.double() * 1000)[:, None] * freq
        sinusoid = torch.cat((phase.cos(), phase.sin()), -1).float()
        t = m.time_embedding(sinusoid.to(m.time_embedding[0].weight.dtype)).float()
        mod = m.shared_block_modulation(t.to(m.shared_block_modulation.weight.dtype)).view(-1,6,self.config.dim)
        head = m.video_head_modulation(t.to(m.video_head_modulation.weight.dtype)).view(-1,2,self.config.dim)
        return mod, head

    def _text(self, contexts, ropes):
        m = self.base_model.text_embedder
        lengths = [len(x) for x in contexts]
        cu = cumulative(lengths, contexts[0].device)
        rope = torch.cat(ropes)[None]
        x = m.proj(torch.cat(contexts).to(m.proj.weight.dtype))
        for block in m.blocks:
            q,k,v = block.qkv(block.norm1(x)).chunk(3,-1)
            shape = (-1,block.heads,x.shape[-1]//block.heads)
            q = block.q_norm(q.reshape(shape))
            k = block.k_norm(k.reshape(shape))
            q = self._rotate(q[None],rope)[0].to(v.dtype)
            k = self._rotate(k[None],rope)[0].to(v.dtype)
            y = varlen_attn(q,k,v.reshape(shape),cu,cu,max(lengths),max(lengths))
            x = x + block.proj(y.flatten(1))
            y = block.norm2(x)
            h,g = block.ffn_up(y),block.ffn_gate(y)
            x = x + block.ffn_down((4*torch.tanh(g/4)*torch.sigmoid(g))*(25*torch.tanh(h/25)))
        return x.split(lengths)

    def _packed_block(self, layer, x, rope, mod, layout):
        # t=0 for reference tokens; each target/text pair has its own sampled t.
        ids = layout['time_ids']
        y = layer.norm1(x)*(1+mod[:,1][ids]) + mod[:,0][ids]
        y = y.to(layer.proj['q'].weight.dtype)
        shape = (-1,layer.heads,self.config.dim//layer.heads)
        q = layer.q_norm(layer.proj['q'](y).reshape(shape))
        k = layer.k_norm(layer.proj['k'](y).reshape(shape))
        v = layer.proj['v'](y).reshape(shape)
        q = self._rotate(q[None],rope[None])[0].to(v.dtype)
        k = self._rotate(k[None],rope[None])[0].to(v.dtype)
        attention = EditAttention.apply(q,k,v,layout).flatten(1)
        x = x + layer.proj['o'](attention)*mod[:,2][ids]
        y = layer.norm2(x)*(1+mod[:,4][ids]) + mod[:,3][ids]
        h,g = layer.proj['ffn_up_gate'](y.to(layer.proj['q'].weight.dtype)).chunk(2,-1)
        h = (4*torch.tanh(g/4)*torch.sigmoid(g))*(25*torch.tanh(h/25))
        return x + layer.proj['ffn_down'](h)*mod[:,5][ids]

    def forward_packed(self, images, times, contexts, references):
        m, device = self.base_model, images[0].device
        body_ropes = [m.rope(x.shape[-2],x.shape[-1],len(c),device)[0].to(x.dtype)
                      for x,c in zip(images,contexts)]
        text_ropes = [r[x.shape[-2]*x.shape[-1]:] for r,x in zip(body_ropes,images)]
        text = self._text(contexts,text_ropes)
        tokens, ropes = [],[]
        target_indices, target_ids, time_ids = [],[],[]
        ref_lengths,text_lengths,all_lengths = [],[],[]
        offset = 0
        for i,(image,ref,context,rope) in enumerate(zip(images,references,text,body_ropes)):
            prefix,ref_rope = self._reference_inputs((ref[None],))
            video = m.video_embedder(image.flatten(1).T.to(m.video_embedder.weight.dtype))
            nr,nv,nt = prefix.shape[1],video.shape[0],context.shape[0]
            tokens.append(torch.cat((context,prefix[0],video)))
            ropes.append(torch.cat((rope[nv:],ref_rope[0],rope[:nv])))
            target_indices.extend(range(offset+nt+nr,offset+nt+nr+nv))
            target_ids.extend([i]*nv)
            time_ids.extend([i+1]*nt+[0]*nr+[i+1]*nv)
            ref_lengths.append(nr); text_lengths.append(nt); all_lengths.append(nt+nr+nv)
            offset += nt+nr+nv
        layout = edit_layout(ref_lengths,text_lengths,all_lengths,device)
        layout['time_ids'] = torch.tensor(time_ids,device=device,dtype=torch.long)
        mod,head = self._times(torch.cat((times.new_zeros(1),times)))
        x,rope = torch.cat(tokens),torch.cat(ropes)
        for layer in m.layers:
            fn = partial(self._packed_block,layer,layout=layout)
            x = checkpoint(fn,x,rope,mod,use_reentrant=False) if self.gradient_checkpointing and self.training else fn(x,rope,mod)
        targets = torch.tensor(target_indices,device=device,dtype=torch.long)
        ids = torch.tensor(target_ids,device=device,dtype=torch.long)
        x = m.final_norm_video(x[targets].to(m.final_norm_video.weight.dtype))
        x = x*(1+head[1:,1][ids]) + head[1:,0][ids]
        x = m.final_linear_video(x.to(m.final_linear_video.weight.dtype))
        sizes = [im.shape[-2]*im.shape[-1] for im in images]
        return [out.T.reshape_as(im) for out,im in zip(x.split(sizes),images)]

    def forward(self, hidden_states, timestep, encoder_hidden_states, reference_latents=(), **kwargs):
        if isinstance(hidden_states,list):
            return self.forward_packed(hidden_states,timestep,encoder_hidden_states,reference_latents)
        # Validation also uses the same FA4 mask_mod; no alternate attention path.
        context = encoder_hidden_states[0]
        if kwargs.get('encoder_attention_mask') is not None:
            context = context[kwargs['encoder_attention_mask'][0].bool()]
        return self.forward_packed([hidden_states[0]],timestep.reshape(1),[context],
                                   [reference_latents[0][0]])[0][None]
