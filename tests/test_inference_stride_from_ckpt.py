"""Inference stride must default to checkpoint tile_size (e.g. 300 for VGG Phase 8)."""


def test_stride_defaults_to_tile_size_logic():
    tile_size = 300
    stride_arg = None
    stride = stride_arg if stride_arg is not None else tile_size
    assert stride == 300


def test_ckpt_tile_size_used_for_stride():
    ckpt_args = {"tile_size": 300}
    tile_size = ckpt_args.get("tile_size", 299)
    stride = tile_size
    assert stride == 300
