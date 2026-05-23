import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup import (
    _find_extract_layout_root,
    _is_safe_zip_member,
    _safe_extract_zip,
    _validate_layout_root,
    check_dataset_exists,
)


def test_is_safe_zip_member_blocks_traversal():
    assert _is_safe_zip_member("Train/img.jpg") is True
    assert _is_safe_zip_member("../evil.jpg") is False
    assert _is_safe_zip_member("/etc/passwd") is False


def test_safe_extract_and_layout(tmp_path):
    zpath = tmp_path / "mini.zip"
    extract_to = tmp_path / "out"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("Train/a.jpg", b"x")
        zf.writestr("TrainDotted/a.jpg", b"x")
        zf.writestr("Test/b.jpg", b"y")
        zf.writestr("train.csv", "id,adult_males,adult_females,subadult_males,subadult_females,pups\n")
        zf.writestr("sample_submission.csv", "id,adult_males,adult_females,subadult_males,subadult_females,pups\n")
        zf.writestr("MismatchedTrainImages.txt", "")

    _safe_extract_zip(zpath, extract_to)
    root = _find_extract_layout_root(extract_to)
    assert root is not None
    _validate_layout_root(root)


def test_setup_no_module_level_tqdm_import():
    import ast

    tree = ast.parse((ROOT / "setup.py").read_text())
    imports = [
        n.name if isinstance(n, ast.Name) else getattr(n, "id", None)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "tqdm"
        for n in node.names
    ]
    assert imports == []
    assert "_tqdm" in (ROOT / "setup.py").read_text()
