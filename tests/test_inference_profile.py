import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.predict import InferenceTimings, format_inference_profile, predict_image_tiled
from data.tiling import iter_tile_windows


def test_inference_timings_accumulates_counts():
    timings = InferenceTimings()
    timings.add_image(n_tiles=10, elapsed_sec=1.0)
    timings.n_batch_forwards = 2
    timings.load_image_sec = 0.1
    timings.preprocess_sec = 0.2
    timings.forward_sec = 0.5

    summary = format_inference_profile(timings)

    assert timings.n_images == 1
    assert timings.n_tiles == 10
    assert timings.n_batch_forwards == 2
    assert summary["images_per_sec"] == 1.0
    assert summary["tiles_per_sec"] == 10.0
    assert summary["forward_pct"] > 0.0


def test_predict_image_tiled_records_timings(mini_data_dir):
    from model.build import build_counter

    device = torch.device("cpu")
    model = build_counter("resnet18", pretrained=False).to(device)
    model.eval()

    test_path = sorted((mini_data_dir / "Test").glob("*.jpg"))[0]
    timings = InferenceTimings()

    pred = predict_image_tiled(
        model,
        test_path,
        tile_size=128,
        device=device,
        shifts=1,
        stride=128,
        batch_size=2,
        timings=timings,
    )

    assert pred.shape == (5,)
    assert timings.n_images == 1
    from PIL import Image

    with Image.open(test_path) as img:
        expected_tiles = len(list(iter_tile_windows(*img.size, tile_size=128, stride=128, shifts=1)))
    assert timings.n_tiles == expected_tiles
    assert timings.n_batch_forwards > 0
    assert timings.load_image_sec >= 0.0
    assert timings.preprocess_sec >= 0.0
    assert timings.forward_sec >= 0.0


def test_format_inference_profile_handles_zero_images():
    summary = format_inference_profile(InferenceTimings())
    assert summary["images_per_sec"] == 0.0
    assert summary["tiles_per_sec"] == 0.0
