import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


MemoryKV = Tuple[Tensor, Tensor]


class StepState:
    """One layer's self-attention keys and values of the tags decoded so far,
    (bsz, heads, capacity, head_dim), of which the first `length` are filled."""

    def __init__(self, k: Tensor, v: Tensor):
        self.k = k
        self.v = v
        self.length = 0

    def grow(self) -> None:
        self.k = torch.cat([self.k, torch.empty_like(self.k)], dim=2)
        self.v = torch.cat([self.v, torch.empty_like(self.v)], dim=2)


class TMTransformerDecoder(nn.TransformerDecoder):
    def precompute_memory_kv(self, memory: Tensor) -> Optional[List[MemoryKV]]:
        """Each layer's cross-attention keys and values of `memory`.

        `memory` is the same for every decoding step of a table, so its
        key/value projections are too. Compute them once and pass the result
        as `memory_kv` to `forward`, instead of projecting `memory` again in
        every layer at every step. None if a layer cannot use them.
        """
        memory_kv = []
        for mod in self.layers:
            if not mod.can_use_memory_kv():
                return None
            memory_kv.append(mod.project_memory(memory))
        return memory_kv

    def can_step(self) -> bool:
        """Whether `step` can stand in for `forward` (inference only)."""
        return all(mod.can_use_memory_kv() and mod.can_step() for mod in self.layers)

    def new_step_state(self, max_len: int, bsz: int = 1) -> List["StepState"]:
        """Empty per-layer self-attention key/value buffers for `step`."""
        return [mod.new_step_state(max_len, bsz) for mod in self.layers]

    def step(
        self,
        tgt_last_tok: Tensor,
        memory_kv: List[MemoryKV],
        state: List["StepState"],
    ) -> Tensor:
        """Decode one more tag from the embedding of the newest tag alone.

        `forward` gets every layer's whole input sequence at every step and
        projects all of it to self-attention keys and values again, although
        only the newest row is new. `step` keeps each layer's keys and values
        in `state` and appends one row, so a step costs the same however long
        the sequence is.

        It is the same arithmetic on the same numbers, but not bit for bit:
        a row projected alone and the same row projected inside a longer
        matrix can differ in the last bits (the BLAS kernel depends on the
        matrix shape). Results agree to float32 rounding.

        Args:
            tgt_last_tok: embedded newest tag, (1, bsz, hidden_dim)
            memory_kv: `precompute_memory_kv(memory)`
            state: `new_step_state(...)`, updated in place
        Returns:
            the last layer's output for the newest tag, (1, bsz, hidden_dim)
        """
        output = tgt_last_tok
        for mod, kv, layer_state in zip(self.layers, memory_kv, state):
            output = mod.step(output, kv, layer_state)
        return output

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_kv: Optional[List[MemoryKV]] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[Tensor]): None during training, only used during inference.
            memory_kv: `precompute_memory_kv(memory)`, only used during inference.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        output = tgt

        # cache
        tag_cache = []
        for i, mod in enumerate(self.layers):
            if memory_kv is None:
                output = mod(output, memory)
            else:
                output = mod(output, memory, memory_kv=memory_kv[i])
            tag_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            out_cache = torch.cat([cache, torch.stack(tag_cache, dim=0)], dim=1)
        else:
            out_cache = torch.stack(tag_cache, dim=0)

        return output, out_cache  # type: ignore


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def can_use_memory_kv(self) -> bool:
        """Whether cross attention can run from `project_memory` output.

        `_cross_attention` covers the plain case of `nn.MultiheadAttention`
        only: packed projection weights, no extra bias or zero attention
        entries, sequence-first tensors, and no dropout (eval mode).
        """
        mha = self.multihead_attn
        return (
            not self.training
            and mha._qkv_same_embed_dim
            and mha.in_proj_weight is not None
            and mha.bias_k is None
            and mha.bias_v is None
            and not mha.add_zero_attn
            and not mha.batch_first
        )

    def project_memory(self, memory: Tensor) -> MemoryKV:
        """Cross-attention keys and values of `memory`, (bsz, heads, src_len, head_dim).

        The same operations, in the same order, as
        `torch.nn.functional.multi_head_attention_forward` applies to its key
        and value arguments, so attending with them gives the same result.
        """
        mha = self.multihead_attn
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        src_len, bsz, _ = memory.shape

        _, w_kv = mha.in_proj_weight.split([embed_dim, embed_dim * 2])
        b_kv = None
        if mha.in_proj_bias is not None:
            _, b_kv = mha.in_proj_bias.split([embed_dim, embed_dim * 2])
        kv_proj = F.linear(memory, w_kv, b_kv)
        kv_proj = (
            kv_proj.unflatten(-1, (2, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        k, v = kv_proj[0], kv_proj[1]
        k = k.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)
        return k, v

    def _cross_attention(self, query: Tensor, memory_kv: MemoryKV) -> Tensor:
        """`self.multihead_attn(query, memory, memory, need_weights=False)[0]`
        from the precomputed keys and values of `memory`."""
        mha = self.multihead_attn
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        tgt_len, bsz, _ = query.shape

        w_q, _ = mha.in_proj_weight.split([embed_dim, embed_dim * 2])
        b_q = None
        if mha.in_proj_bias is not None:
            b_q, _ = mha.in_proj_bias.split([embed_dim, embed_dim * 2])
        q = F.linear(query, w_q, b_q)
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        q = q.view(bsz, num_heads, tgt_len, head_dim)

        k, v = memory_kv
        attn_output = F.scaled_dot_product_attention(q, k, v, None, 0.0, False)
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )
        attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)
        return attn_output.view(tgt_len, bsz, attn_output.size(1))

    def can_step(self) -> bool:
        """Whether self attention can run from cached keys and values."""
        mha = self.self_attn
        return (
            not self.training
            and mha._qkv_same_embed_dim
            and mha.in_proj_weight is not None
            and mha.bias_k is None
            and mha.bias_v is None
            and not mha.add_zero_attn
            and not mha.batch_first
        )

    def new_step_state(self, max_len: int, bsz: int = 1) -> "StepState":
        mha = self.self_attn
        head_dim = mha.embed_dim // mha.num_heads
        weight = mha.in_proj_weight
        shape = (bsz, mha.num_heads, max_len, head_dim)
        return StepState(
            k=torch.empty(shape, dtype=weight.dtype, device=weight.device),
            v=torch.empty(shape, dtype=weight.dtype, device=weight.device),
        )

    def step(
        self, tgt_last_tok: Tensor, memory_kv: MemoryKV, state: "StepState"
    ) -> Tensor:
        """`forward` for the newest tag, with the keys and values of the
        earlier tags taken from `state` (see `TMTransformerDecoder.step`)."""
        mha = self.self_attn
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        _, bsz, _ = tgt_last_tok.shape

        w_q, w_kv = mha.in_proj_weight.split([embed_dim, embed_dim * 2])
        b_q = b_kv = None
        if mha.in_proj_bias is not None:
            b_q, b_kv = mha.in_proj_bias.split([embed_dim, embed_dim * 2])
        q = F.linear(tgt_last_tok, w_q, b_q)
        kv_proj = F.linear(tgt_last_tok, w_kv, b_kv)
        kv_proj = (
            kv_proj.unflatten(-1, (2, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q = q.view(1, bsz * num_heads, head_dim).transpose(0, 1)
        q = q.view(bsz, num_heads, 1, head_dim)
        pos = state.length
        if pos >= state.k.shape[2]:
            state.grow()
        state.k[:, :, pos] = kv_proj[0].view(bsz, num_heads, head_dim)
        state.v[:, :, pos] = kv_proj[1].view(bsz, num_heads, head_dim)
        state.length = pos + 1
        k = state.k[:, :, : pos + 1]
        v = state.v[:, :, : pos + 1]

        attn_output = F.scaled_dot_product_attention(q, k, v, None, 0.0, False)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz, embed_dim)
        attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)
        tmp_tgt = attn_output.view(1, bsz, attn_output.size(1))
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        tmp_tgt = self._cross_attention(tgt_last_tok, memory_kv)
        tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
        tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_kv: Optional[MemoryKV] = None,
    ) -> Tensor:
        """
        Args:
            same as TMTransformerDecoder
            memory_kv: `project_memory(memory)`; cross attention then skips
                projecting `memory`. Only without memory masks.
        Returns:
            Tensor:
                During training (seq_len,bsz,hidden_dim)
                If eval mode: embedding of last tag: (1,bsz,hidden_dim)
        """

        # From PyTorch but modified to only use the last tag
        tgt_last_tok = tgt[-1:, :, :]

        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # None, because we only care about the last tag
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,  # Optimization: Don't compute attention weights
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        if (
            memory_kv is not None
            and memory_mask is None
            and memory_key_padding_mask is None
        ):
            tmp_tgt = self._cross_attention(tgt_last_tok, memory_kv)
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)
        elif memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,  # Optimization: Don't compute attention weights
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


class Tag_Transformer(nn.Module):
    """
    "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        device,
        vocab_size,
        td_encode,
        embed_dim,
        encoder_layers,
        decoder_layers,
        enc_image_size,
        dropout=0.1,
        n_heads=4,
        dim_ff=1024,
    ):

        super(Tag_Transformer, self).__init__()

        self._device = device
        self._n_heads = n_heads
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim)
        self._td_encode = td_encode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_ff
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )

        self._decoder = TMTransformerDecoder(
            TMTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
            ),
            num_layers=decoder_layers,
        )

        self._decoder_dim = embed_dim
        self._enc_image_size = enc_image_size
        self._input_filter = u.resnet_block(stride=1)
        self._fc = nn.Linear(embed_dim, vocab_size)

    def inference(self, enc_inputs, tags, tag_lens, num_cells):
        # CNN backbone image encoding
        enc_inputs = self._input_filter(enc_inputs.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )

        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)

        enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).to(self._device)

        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]
        # Transformer Encoder Encoded Image mask need to check if its useful
        encoder_mask = torch.zeros(
            (batch_size * self._n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * self._n_heads, positions, positions), device=self._device
        )

        # Transformer Encoder
        encoder_out = self._encoder(enc_inputs, mask=encoder_mask)

        decode_lengths = (tag_lens - 1).tolist()

        tgt = self._positional_encoding(self._embedding(tags).permute(1, 0, 2))

        decoded = self._decoder(tgt, memory=encoder_out)
        decoded = decoded.permute(1, 0, 2)
        predictions = self._fc(decoded)
        return predictions, decode_lengths
