#!/usr/bin/env python3
"""Write random stress inputs for every exported booster, scored by Python LightGBM.

    python make_tree_fuzz.py MODEL.pkl.gz EXPORT_DIR [--rows 2000] [--seed 0]

Values include NaN, exact zeros, tiny values near zero, and exact split
thresholds taken from the trees, so every missing-value and `<=` branch rule
is exercised. Output goes to EXPORT_DIR/fuzz in the same layout as
EXPORT_DIR/reference; check with `check_trees EXPORT_DIR fuzz`.
"""
import argparse
import json
import os

import numpy as np


def thresholds_by_feature(booster, n_features):
    th = [[] for _ in range(n_features)]
    for t in booster.dump_model()["tree_info"]:
        stack = [t["tree_structure"]]
        while stack:
            n = stack.pop()
            if "split_feature" in n:
                th[n["split_feature"]].append(float(n["threshold"]))
                stack += [n["left_child"], n["right_child"]]
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("export_dir")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from zcor.classifier import load_zcor_classifier
    clf = load_zcor_classifier(a.model).CLASSIFIER
    models = dict(clf.BASE_MODELS)
    models["FINAL"] = clf.FINAL_MODEL
    manifest = json.load(open(os.path.join(a.export_dir, "manifest.json")))
    columns = {m["name"]: m["columns"] for m in manifest["base_models"]}
    columns["FINAL"] = manifest["final_model"]["columns"]

    rng = np.random.default_rng(a.seed)
    out = os.path.join(a.export_dir, "fuzz")
    os.makedirs(out, exist_ok=True)
    summary = {"calls": {}}
    for name, model in models.items():
        nf = model.booster_.num_feature()
        th = thresholds_by_feature(model.booster_, nf)
        X = np.zeros((a.rows, nf))
        for j in range(nf):
            pool = np.array(th[j]) if th[j] else np.array([0.0, 1.0])
            kind = rng.integers(0, 6, a.rows)
            v = rng.choice(pool, a.rows)
            v = np.where(kind == 1, v + rng.normal(0, 1e-9, a.rows), v)            # just off a threshold
            v = np.where(kind == 2, rng.uniform(pool.min() - 1, pool.max() + 1, a.rows), v)
            v = np.where(kind == 3, np.nan, v)                                       # missing
            v = np.where(kind == 4, 0.0, v)                                          # exact zero
            v = np.where(kind == 5, rng.choice([1e-36, -1e-36, 1e-30]), v)           # near zero
            X[:, j] = v
        p = model.booster_.predict(X, num_iteration=model.best_iteration_ or None)
        X.tofile(os.path.join(out, f"{name}.X.f64"))
        p.astype(np.float64).tofile(os.path.join(out, f"{name}.p.f64"))
        summary["calls"][name] = {"rows": a.rows, "cols": nf, "columns": columns[name]}
        print(name, nf, "features,", a.rows, "rows")
    json.dump(summary, open(os.path.join(out, "reference.json"), "w"))


if __name__ == "__main__":
    main()
