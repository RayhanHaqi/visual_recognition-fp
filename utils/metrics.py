import numpy as np


def rmse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def rmse_per_class(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 1:
        names = [f"c{i}" for i in range(len(pred))]
        return {names[0]: float(np.abs(pred[0] - target[0]))}
    names = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
    if pred.shape[1] != 5:
        names = [f"c{i}" for i in range(pred.shape[1])]
    out = {}
    for i, name in enumerate(names[: pred.shape[1]]):
        diff = pred[:, i] - target[:, i]
        out[name] = float(np.sqrt(np.mean(diff ** 2)))
    return out
