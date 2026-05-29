import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.targets import COUNT_COLUMNS, normalize_test_id
from inference import _plan_model_runs


def test_plan_model_runs_fails_when_sample_id_has_no_test_image():
    zero = {c: 0.0 for c in COUNT_COLUMNS}

    try:
        _plan_model_runs(["999.jpg"], {}, zero, max_images=None)
    except FileNotFoundError as exc:
        assert "999.jpg" in str(exc)
    else:
        raise AssertionError("missing sample ids must not be silently zero-filled")


def test_plan_model_runs_respects_max_images(mini_data_dir):
    from data.targets import list_test_images, submission_id_column
    import pandas as pd

    sample = pd.read_csv(mini_data_dir / "sample_submission.csv")
    test_paths = list_test_images(mini_data_dir, test_subdir="Test")
    path_by_id = {p.stem: p for p in test_paths}
    path_by_id.update({p.name: p for p in test_paths})
    zero = {c: 0.0 for c in COUNT_COLUMNS}
    ids = sample[submission_id_column(sample)].astype(str).tolist()

    predict_list, cache, n_zero = _plan_model_runs(ids, path_by_id, zero, max_images=1)

    assert len(predict_list) == 1
    assert len(cache) >= 0
    key, path = predict_list[0]
    assert path.is_file()
    assert key == normalize_test_id(key)
