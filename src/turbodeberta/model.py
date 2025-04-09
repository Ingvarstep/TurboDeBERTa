import math
import torch
from transformers.models.deberta_v2 import (DisentangledSelfAttention,
                                            DebertaV2Attention,
                                            )

from .padding import _upad_input, pad_input
from .ops.flash_attention import flash_attention_with_disentangled
from .ops.flash_attention_varlen import flash_attention_with_disentangled_varlen

class FlashDisentangledSelfAttention(DisentangledSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=None,
                    relative_pos=None,
                    rel_embeddings=None):
        """
        Performs the flash attention forward pass with disentangled relative attention.

        Args:
            hidden_states (Tensor): Input tensor of shape (B, L, hidden_size).
            attention_mask (Tensor): The attention mask.
            output_attentions (bool): Whether to return attention weights.
            query_states (Tensor, optional): If provided, used as Q.
            relative_pos (Tensor, optional): Relative position encoding. If None, will be built.
            causal (bool): Whether to apply causal masking.
            sm_scale (float, optional): Scaling factor for softmax.

        Returns:
            Tuple[Tensor, None]: A tuple where the first element is the output tensor of shape (B, L, hidden_size).
        """
        if attention_mask is not None:
            total_length = torch.sum(attention_mask)
            max_length = torch.prod(torch.tensor(attention_mask.shape).to(attention_mask.device))
            if max_length==total_length:
                varlen = False
            else:
                varlen = True
        else:
            varlen = False

        if query_states is None:
            query_states = hidden_states

        B, L, _ = hidden_states.shape

        def transform(x, attention_heads):
            new_x_shape = x.size()[:-1] + (attention_heads, -1)
            x = x.view(new_x_shape).permute(0, 2, 1, 3).contiguous()
            return x

        def get_heads(x, attention_heads):
            new_x_shape = x.size()[:-2] + (attention_heads, -1)
            x = x.view(new_x_shape).contiguous()
            return x

        query_layer = self.query_proj(query_states)
        key_layer = self.key_proj(hidden_states)
        value_layer = self.value_proj(hidden_states)

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        sm_scale = 1/math.sqrt(self.head_dim*scale_factor)

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            if self.share_att_key:
                pos_key_layer = transform(
                    self.key_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                ) # (1, NH, MD, head_dim)
                pos_query_layer = transform(
                    self.query_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                )
            else:
                if "c2p" in self.pos_att_type:
                    pos_key_layer = transform(
                        self.pos_key_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                    )

                if "p2c" in self.pos_att_type:
                    pos_query_layer = transform(
                        self.pos_query_proj(rel_embeddings.unsqueeze(0)), self.num_attention_heads
                    )
            pos_key = None
            pos_query = None
        else:
            pos_key, pos_query = None, None

        query_layer = transform(query_layer, self.num_attention_heads)
        key_layer = transform(key_layer, self.num_attention_heads)
        value_layer = transform(value_layer, self.num_attention_heads)

        if "c2p" in self.pos_att_type:
            pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
        if "p2c" in self.pos_att_type:
            pos_query = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))

        causal = False
        if not varlen:
            query_layer = transform(query_layer, self.num_attention_heads) # (B, NH, L, head_dim)
            key_layer = transform(key_layer, self.num_attention_heads)
            value_layer = transform(value_layer, self.num_attention_heads)

            if "c2p" in self.pos_att_type:
                pos_key = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            if "p2c" in self.pos_att_type:
                pos_query = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))

            out = flash_attention_with_disentangled(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                causal,
                sm_scale,
                self.position_buckets,
                self.max_relative_positions,
            )
        else:
            query_layer = get_heads(query_layer, self.num_attention_heads) # (B, L, NH, head_dim)
            key_layer = get_heads(key_layer, self.num_attention_heads)
            value_layer = get_heads(value_layer, self.num_attention_heads)

            if "c2p" in self.pos_att_type:
                # query_layer = (1, NH, L, head_dim)
                pos_key = torch.einsum("bqhd,zhmd->bqhm", query_layer, pos_key_layer)
            if "p2c" in self.pos_att_type:
                pos_query = torch.einsum("bqhd,zhmd->bqhm", key_layer, pos_query_layer)

            (query_layer, 
             key_layer,
             value_layer,
             pos_key,
             pos_query,
             indices_q,
             cu_seq_lens,
             max_seq_lens) = _upad_input(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                attention_mask,
                L,
                self.num_attention_heads)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            out_unpad = flash_attention_with_disentangled_varlen(
                query_layer,
                key_layer,
                value_layer,
                pos_key,
                pos_query,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_in_batch_q, max_seqlen_in_batch_k,
                causal,
                sm_scale,
                self.position_buckets,
                self.max_relative_positions,
            )
            out =  pad_input(out_unpad, indices_q, B, L).transpose(1, 2)

        out = out.view(B, self.num_attention_heads, L, self.head_dim).transpose(1, 2).reshape(B, L, self.all_head_size)
        return out
