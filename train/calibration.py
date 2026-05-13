from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Union

import pandas as pd
import torch


POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def _safe_probability(value: float, eps: float = 1e-6) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def _as_dataframe(data: Union[pd.DataFrame, Path, str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def compute_class_priors(
    data: Union[pd.DataFrame, Path, str],
    label_column: str = "label_idx",
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, object]:
    df = _as_dataframe(data)
    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    labels = df[label_column].astype(int)
    positive_count = int((labels == positive_label).sum())
    negative_count = int((labels != positive_label).sum())
    total_count = positive_count + negative_count
    if total_count == 0 or positive_count == 0 or negative_count == 0:
        raise ValueError("Class prior statistics require at least one sample from each class.")

    positive_prior = positive_count / total_count
    negative_prior = negative_count / total_count
    return {
        "counts": {
            str(NEGATIVE_LABEL): negative_count,
            str(POSITIVE_LABEL): positive_count,
        },
        "priors": {
            str(NEGATIVE_LABEL): negative_prior,
            str(POSITIVE_LABEL): positive_prior,
        },
        "negative_to_positive_ratio": negative_count / positive_count,
        "total_count": total_count,
    }


def compute_class_ratio_from_csv(
    csv_path: Union[Path, str],
    label_column: str = "label",
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    df = df.copy()
    df["label_idx"] = df[label_column].astype(int)
    return compute_class_priors(df, label_column="label_idx", positive_label=positive_label)


def build_target_priors_from_ratio(negative_to_positive_ratio: float) -> Dict[str, object]:
    if negative_to_positive_ratio <= 0:
        raise ValueError("Target negative/positive ratio must be positive.")

    positive_prior = 1.0 / (1.0 + float(negative_to_positive_ratio))
    negative_prior = 1.0 - positive_prior
    return {
        "priors": {
            str(NEGATIVE_LABEL): negative_prior,
            str(POSITIVE_LABEL): positive_prior,
        },
        "negative_to_positive_ratio": float(negative_to_positive_ratio),
    }


def build_class_weights_for_target_prior(
    train_priors: Mapping[str, object],
    target_priors: Mapping[str, object],
) -> torch.Tensor:
    train_probs = train_priors["priors"]
    target_probs = target_priors["priors"]

    neg_weight = _safe_probability(target_probs[str(NEGATIVE_LABEL)]) / _safe_probability(
        train_probs[str(NEGATIVE_LABEL)]
    )
    pos_weight = _safe_probability(target_probs[str(POSITIVE_LABEL)]) / _safe_probability(
        train_probs[str(POSITIVE_LABEL)]
    )

    mean_weight = (neg_weight + pos_weight) / 2.0
    return torch.tensor([neg_weight / mean_weight, pos_weight / mean_weight], dtype=torch.float32)


def apply_bayes_prior_correction(
    prob_pos: torch.Tensor,
    train_pos_prior: float,
    target_pos_prior: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob_pos = prob_pos.clamp(min=eps, max=1.0 - eps)
    train_pos_prior = _safe_probability(train_pos_prior, eps)
    target_pos_prior = _safe_probability(target_pos_prior, eps)

    prior_logit_delta = math.log(target_pos_prior / (1.0 - target_pos_prior)) - math.log(
        train_pos_prior / (1.0 - train_pos_prior)
    )
    corrected_logits = torch.logit(prob_pos) + prior_logit_delta
    return torch.sigmoid(corrected_logits)


def compute_binary_f1(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    predictions = (probabilities >= threshold).to(torch.long)
    labels = labels.to(torch.long)

    true_positive = int(((predictions == POSITIVE_LABEL) & (labels == POSITIVE_LABEL)).sum().item())
    false_positive = int(((predictions == POSITIVE_LABEL) & (labels == NEGATIVE_LABEL)).sum().item())
    false_negative = int(((predictions == NEGATIVE_LABEL) & (labels == POSITIVE_LABEL)).sum().item())

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def search_best_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "f1",
    candidate_thresholds: Iterable[float] | None = None,
) -> Dict[str, float]:
    if metric != "f1":
        raise ValueError(f"Unsupported threshold metric: {metric}")

    if candidate_thresholds is None:
        unique_scores = torch.unique(probabilities.detach().cpu()).tolist()
        candidate_thresholds = sorted({0.0, 0.5, 1.0, *[float(score) for score in unique_scores]})

    best_threshold = 0.5
    best_metric = float("-inf")
    for threshold in candidate_thresholds:
        metric_value = compute_binary_f1(probabilities, labels, float(threshold))
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = float(threshold)

    return {
        "threshold": best_threshold,
        "metric_value": best_metric,
    }


def summarize_priors(priors: Mapping[str, object]) -> Dict[str, object]:
    payload = {
        "priors": {key: float(value) for key, value in priors["priors"].items()},
        "negative_to_positive_ratio": float(priors["negative_to_positive_ratio"]),
    }
    if "counts" in priors:
        payload["counts"] = dict(priors["counts"])
    return payload
