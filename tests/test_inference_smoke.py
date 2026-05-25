"""Smoke test tiled prediction on mini fixture."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_predict_image_tiled_writes_csv(mini_data_dir, tmp_path):
    from model.build import build_counter
    from utils.io import save_checkpoint
    from data.predict import predict_image_tiled
    import pandas as pd
    from data.targets import COUNT_COLUMNS, LEGACY_COUNT_COLUMNS, pred_vector_to_submission_row

    ckpt = tmp_path / "fake_best.pth"
    model = build_counter("resnet18", pretrained=False)
    save_checkpoint(ckpt, model, None, 1, 0.0, {"backbone": "resnet18", "tile_size": 128})

    device = torch.device("cpu")
    from utils.io import load_checkpoint

    m = build_counter("resnet18", pretrained=False).to(device)
    load_checkpoint(ckpt, m, device)

    sample = pd.read_csv(mini_data_dir / "sample_submission.csv")
    id_col = "test_id"
    rows = []
    for img_id in sample[id_col]:
        path = mini_data_dir / "Test" / Path(str(img_id)).name
        if not path.is_file():
            path = mini_data_dir / "Test" / f"{Path(str(img_id)).stem}.jpg"
        pred = predict_image_tiled(m, path, 128, device, shifts=1, stride=64, batch_size=2)
        assert pred.shape == (5,)
        assert (pred >= 0).all()
        row = {id_col: img_id}
        row.update(pred_vector_to_submission_row(pred, source_columns=LEGACY_COUNT_COLUMNS))
        rows.append(row)

    out = tmp_path / "out.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.read_csv(out)
    assert len(df) == 2
    assert (df["pups"] >= 0).all()
