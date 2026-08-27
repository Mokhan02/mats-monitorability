import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .model import encode_full_turn


def _fit_eval(X, y, test_size, seed, C: float = 1.0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=C, max_iter=2000).fit(X_train, y_train)
    auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return clf, auroc


def scan_layers(pos_acts: dict[int, torch.Tensor], neg_acts: dict[int, torch.Tensor],
                 test_size: float = 0.3, seed: int = 0, C: float = 1.0) -> dict[int, dict]:
    """Per-layer probe AUROC, for LAYER SELECTION only.

    Fit this on a subset disjoint from whatever data you'll report the final
    AUROC on: argmax over ~36 layers with a small n will find noise, so an
    AUROC read off the same set used to pick the layer is inflated.
    """
    results = {}
    for layer in pos_acts:
        X = torch.cat([pos_acts[layer], neg_acts[layer]]).numpy()
        y = np.array([1] * len(pos_acts[layer]) + [0] * len(neg_acts[layer]))
        clf, auroc = _fit_eval(X, y, test_size, seed, C)
        results[layer] = {"auroc": auroc, "probe": clf}
    return results


def best_layer(scan_results: dict[int, dict]) -> int:
    return max(scan_results, key=lambda l: scan_results[l]["auroc"])


def fit_final_probe(pos_acts: dict[int, torch.Tensor], neg_acts: dict[int, torch.Tensor],
                     layer: int, test_size: float = 0.3, seed: int = 0, C: float = 1.0):
    """Fit + eval at a layer already chosen from a *different* data split."""
    X = torch.cat([pos_acts[layer], neg_acts[layer]]).numpy()
    y = np.array([1] * len(pos_acts[layer]) + [0] * len(neg_acts[layer]))
    return _fit_eval(X, y, test_size, seed, C)


def shuffle_label_control(pos_acts: dict[int, torch.Tensor], neg_acts: dict[int, torch.Tensor],
                           layer: int, test_size: float = 0.3, seed: int = 0) -> float:
    """Refit on permuted labels. AUROC should collapse toward 0.5 -- if it
    doesn't, the probe is separating on sample count (d >> n), not the
    concept, regardless of what the real-label AUROC said.
    """
    _, auroc = fit_shuffled_label_probe(pos_acts, neg_acts, layer, test_size, seed)
    return auroc


def fit_shuffled_label_probe(pos_acts: dict[int, torch.Tensor], neg_acts: dict[int, torch.Tensor],
                              layer: int, test_size: float = 0.3, seed: int = 0, C: float = 1.0):
    """Same fit as shuffle_label_control, but returns the classifier too --
    used as a steering-direction control: its weight vector should carry no
    persistence signal specifically *about the concept*, since it was never
    fit to detect one.
    """
    X = torch.cat([pos_acts[layer], neg_acts[layer]]).numpy()
    y = np.array([1] * len(pos_acts[layer]) + [0] * len(neg_acts[layer]))
    y_shuffled = np.random.RandomState(seed).permutation(y)
    return _fit_eval(X, y_shuffled, test_size, seed, C)


@torch.no_grad()
def probe_response(model, tokenizer, probe, layer: int, prompt: str, completion: str) -> float:
    """Unsteered probe readout on a generated (prompt, completion) pair --
    what a deployed monitor would see. Caller must not have a steering hook
    registered when this runs: encode_full_turn does a clean forward pass, but
    a hook left active on `model` would still fire and contaminate the read,
    since hooks are a property of the model, not of this call.
    """
    encoded = encode_full_turn(tokenizer, prompt, completion, model.device)
    out = model(**encoded, output_hidden_states=True)
    act = out.hidden_states[layer][0, -1, :].float().cpu().numpy().reshape(1, -1)
    return float(probe.decision_function(act)[0])
