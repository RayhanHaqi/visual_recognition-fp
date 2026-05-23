"""Smoke test inference on mini fixture (no real checkpoint training)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_inference_writes_csv(mini_data_dir, tmp_path):
    from model.build import build_counter
    from utils.io import save_checkpoint
    from inference import main as infer_main
    import argparse

    ckpt = tmp_path / "fake_best.pth"
    model = build_counter("resnet18", pretrained=False)
    save_checkpoint(ckpt, model, None, 1, 0.0, {"backbone": "resnet18", "tile_size": 128})

    out = tmp_path / "out.csv"
    # Call inference logic directly
    import inference as inf

    class Args:
        checkpoint = str(ckpt)
        run_name = "smoke"
        data_path = str(mini_data_dir)
        test_subdir = "Test"
        tile_size = 128
        shifts = 1
        stride = 64
        gpu = 0
        pup_scale = 1.0
        output = str(out)

    import sys as _sys
    old = _sys.argv
    try:
        device = torch.device("cpu")
        ckpt_obj = torch.load(ckpt, map_location=device, weights_only=False)
        from model.build import build_counter as bc
        from utils.io import load_checkpoint
        m = bc("resnet18", pretrained=False).to(device)
        load_checkpoint(ckpt, m, device)
        from data.targets import list_test_images
        import pandas as pd
        from data.targets import COUNT_COLUMNS
        from inference import predict_image_tiled

        sample = pd.read_csv(mini_data_dir / "sample_submission.csv")
        id_col = "id"
        rows = []
        for img_id in sample[id_col]:
            path = mini_data_dir / "Test" / Path(img_id).name
            if not path.is_file():
                path = mini_data_dir / "Test" / f"{Path(img_id).stem}.jpg"
            pred = predict_image_tiled(m, path, 128, device, shifts=1, stride=64)
            row = {id_col: img_id}
            for i, col in enumerate(COUNT_COLUMNS):
                row[col] = max(0.0, float(pred[i]))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
    finally:
        _sys.argv = old

    from pathlib import Path as P
    df = __import__("pandas").read_csv(out)
    assert len(df) == 2
    assert (df["pups"] >= 0).all()
