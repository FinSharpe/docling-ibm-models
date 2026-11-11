#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""The TableFormer decoders reuse what does not change between decoding steps
(cross-attention keys/values of the encoded image) or between cells (the
image and initial-state terms of the cell attention). These tests hold the
reuse to the original, recompute-everything arithmetic bit for bit.
"""

import torch

from docling_ibm_models.tableformer.models.table04_rs.bbox_decoder_rs import (
    BBoxDecoder,
)
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    TMTransformerDecoder,
    TMTransformerDecoderLayer,
)

EMBED_DIM = 64
HEADS = 4
LAYERS = 3
POSITIONS = 49


def _decoder() -> TMTransformerDecoder:
    torch.manual_seed(7)
    decoder = TMTransformerDecoder(
        TMTransformerDecoderLayer(d_model=EMBED_DIM, nhead=HEADS, dim_feedforward=128),
        num_layers=LAYERS,
    )
    return decoder.eval()


def _decode(decoder, memory, steps, memory_kv):
    """Step-by-step decoding as `TableModel04_rs.predict` drives it."""
    torch.manual_seed(11)
    tags = torch.randn(steps, 1, EMBED_DIM)
    cache = None
    outputs = []
    for step in range(1, steps + 1):
        decoded, cache = decoder(tags[:step], memory, cache, memory_kv=memory_kv)
        outputs.append(decoded[-1])
    return torch.stack(outputs), cache


def test_precomputed_memory_kv_gives_bit_identical_decoding():
    decoder = _decoder()
    torch.manual_seed(3)
    memory = torch.randn(POSITIONS, 1, EMBED_DIM)

    with torch.no_grad():
        memory_kv = decoder.precompute_memory_kv(memory)
        assert memory_kv is not None and len(memory_kv) == LAYERS
        expected, expected_cache = _decode(decoder, memory, steps=12, memory_kv=None)
        actual, actual_cache = _decode(decoder, memory, steps=12, memory_kv=memory_kv)

    assert torch.equal(actual, expected)
    assert torch.equal(actual_cache, expected_cache)


def test_cross_attention_matches_multihead_attention():
    layer = _decoder().layers[1]
    torch.manual_seed(5)
    memory = torch.randn(POSITIONS, 2, EMBED_DIM)  # batch of 2
    query = torch.randn(1, 2, EMBED_DIM)

    with torch.no_grad():
        expected = layer.multihead_attn(query, memory, memory, need_weights=False)[0]
        actual = layer._cross_attention(query, layer.project_memory(memory))

    assert torch.equal(actual, expected)


def test_memory_kv_is_not_used_in_training_mode():
    decoder = _decoder()
    decoder.train()

    assert decoder.precompute_memory_kv(torch.randn(POSITIONS, 1, EMBED_DIM)) is None


def test_a_memory_mask_falls_back_to_full_cross_attention():
    layer = _decoder().layers[0]
    torch.manual_seed(9)
    memory = torch.randn(POSITIONS, 1, EMBED_DIM)
    tgt = torch.randn(4, 1, EMBED_DIM)
    mask = torch.zeros(1, POSITIONS, dtype=torch.bool)
    mask[0, 10:] = True  # hide most of the image

    with torch.no_grad():
        expected = layer(tgt, memory, memory_key_padding_mask=mask)
        actual = layer(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            memory_kv=layer.project_memory(memory),
        )
        unmasked = layer(tgt, memory)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, unmasked)


def _bbox_reference(decoder: BBoxDecoder, encoder_out, tag_H):
    """`BBoxDecoder.inference` as originally written: every term per cell."""
    encoder_out = decoder._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(
        0, 2, 3, 1
    )
    encoder_out = encoder_out.view(1, -1, encoder_out.size(3))
    boxes, classes = [], []
    for cell_tag_H in tag_H:
        h = decoder._init_hidden_state(encoder_out, 1)
        awe, _ = decoder._attention(encoder_out, cell_tag_H, h)
        gate = decoder._sigmoid(decoder._f_beta(h))
        awe = gate * awe
        h = awe * h
        boxes.append(decoder._bbox_embed(h).sigmoid())
        classes.append(decoder._class_embed(h))
    return (
        torch.stack([x[0] for x in classes]),
        torch.stack([x[0] for x in boxes]),
    )


def test_bbox_decoder_matches_per_cell_reference():
    torch.manual_seed(13)
    decoder = BBoxDecoder(
        "cpu",
        attention_dim=96,
        embed_dim=16,
        tag_decoder_dim=EMBED_DIM,
        decoder_dim=512,
        num_classes=2,
        encoder_dim=512,
    ).eval()
    encoder_out = torch.randn(1, 7, 7, 256)
    tag_H = [torch.randn(1, EMBED_DIM) for _ in range(9)]

    with torch.no_grad():
        expected_classes, expected_boxes = _bbox_reference(decoder, encoder_out, tag_H)
        classes, boxes = decoder.inference(encoder_out, tag_H)
        no_classes, no_boxes = decoder.inference(encoder_out, [])

    assert torch.equal(classes, expected_classes)
    assert torch.equal(boxes, expected_boxes)
    assert no_classes.numel() == 0 and no_boxes.numel() == 0


def test_step_decoding_agrees_with_full_decoding():
    """`step` appends one key/value row per tag instead of projecting the
    whole sequence again. Same numbers up to float32 rounding (a row
    projected alone and inside a longer matrix differ in the last bits)."""
    decoder = _decoder()
    torch.manual_seed(3)
    memory = torch.randn(POSITIONS, 1, EMBED_DIM)
    steps = 40
    torch.manual_seed(11)
    tags = torch.randn(steps, 1, EMBED_DIM)

    with torch.no_grad():
        memory_kv = decoder.precompute_memory_kv(memory)
        assert decoder.can_step()
        # capacity 8: the buffers must grow three times on the way to 40
        state = decoder.new_step_state(max_len=8)
        cache = None
        for step in range(steps):
            expected, cache = decoder(
                tags[: step + 1], memory, cache, memory_kv=memory_kv
            )
            actual = decoder.step(tags[step : step + 1], memory_kv, state)

            # `forward` returns the whole sequence so far, `step` the newest tag
            assert actual.shape == (1, 1, EMBED_DIM)
            assert torch.allclose(actual[-1], expected[-1], rtol=1e-4, atol=1e-5)

    assert all(layer_state.length == steps for layer_state in state)


def test_step_decoding_is_not_offered_in_training_mode():
    decoder = _decoder()
    decoder.train()

    assert decoder.can_step() is False
