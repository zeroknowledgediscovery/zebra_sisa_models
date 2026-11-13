#!/usr/bin/env python3
"""Export a pickled ZeBRA/zcor EHRClassifier into plain files a native scorer can read.

Run with the zcor version the model was built with (zcor 1.1.10 for the ILD model).

    python export_model.py MODEL.pkl.gz OUT_DIR
    python export_model.py MODEL.pkl.gz OUT_DIR --reference PATIENTS.json

Writes
    OUT_DIR/manifest.json           model structure: base models, column order, final model
    OUT_DIR/trees/<NAME>.txt        LightGBM text dump of each booster (best iteration only)
    OUT_DIR/reference/...           (with --reference) exact model inputs and outputs captured
                                    from the Python pipeline, used to test the native code
"""
import argparse
import json
import os
import sys

import numpy as np


def export(predictor, out_dir):
    clf = predictor.CLASSIFIER
    os.makedirs(os.path.join(out_dir, "trees"), exist_ok=True)

    def dump(name, model):
        it = model.best_iteration_ or None
        path = os.path.join(out_dir, "trees", f"{name}.txt")
        model.booster_.save_model(path, num_iteration=it)
        params = model.get_params()
        objective = params.get("objective") or "binary"
        if objective != "binary":
            raise SystemExit(f"{name}: objective {objective!r} not supported (binary only)")
        # The text dump prints `sigmoid` with 6 significant digits. A pickled LightGBM
        # booster is rebuilt from that same text, so the model as deployed uses the
        # rounded value: the native scorer reads it from the tree file, not from here.
        # The training-time value is kept for reference only.
        return {"name": name, "file": f"trees/{name}.txt", "num_iteration": it,
                "objective": objective,
                "training_sigmoid": float(params.get("sigmoid", 1.0)),
                "num_trees": model.booster_.num_trees(),
                "num_features": model.booster_.num_feature()}

    base = []
    for name in clf.BASE_MODEL_NAMES:
        info = dump(name, clf.BASE_MODELS[name])
        cols = [c for c in clf.BASE_MODEL_COLUMNS[name] + clf.BASE_ADDITIONAL_FEATURES
                if c != "patient_id"]
        if "SEQ" in name:
            cols = [c.replace("_code_density", "_density") for c in cols]
        info["columns"] = cols
        base.append(info)

    final = dump("FINAL", clf.FINAL_MODEL)
    final["columns"] = list(clf.FINAL_COLUMNS)

    cfg = predictor.CONFIG
    manifest = {
        "format_version": 1,
        "model_name": predictor.NAME,
        "genesis_info": {k: str(v) for k, v in predictor.GENESIS_INFO.items()},
        "base_additional_features": list(clf.BASE_ADDITIONAL_FEATURES),
        "final_additional_features": list(clf.FINAL_ADDITIONAL_FEATURES),
        "sex_encoding": {"M": 1, "other": 0},
        "base_models": base,
        "final_model": final,
        "config": {k: v for k, v in vars(cfg).items()
                   if isinstance(v, (int, float, str, bool, list, dict)) or v is None},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


def capture_reference(predictor, patients_path, out_dir):
    """Run predict_json once, recording every LightGBM call's exact input matrix and output."""
    import lightgbm
    from zcor.classifier import predict_json

    clf = predictor.CLASSIFIER
    name_of = {id(m): n for n, m in clf.BASE_MODELS.items()}
    name_of[id(clf.FINAL_MODEL)] = "FINAL"
    calls = {}
    original = lightgbm.LGBMClassifier.predict_proba

    def recording(self, X, *a, **kw):
        out = original(self, X, *a, **kw)
        Xn = X.to_numpy().astype(np.float64) if hasattr(X, "to_numpy") else np.asarray(X, np.float64)
        calls[name_of.get(id(self), f"unknown_{id(self)}")] = {
            "columns": list(X.columns), "X": Xn, "p": out[:, 1].astype(np.float64)}
        return out

    lightgbm.LGBMClassifier.predict_proba = recording
    try:
        with open(patients_path) as f:
            patients = json.load(f)
        result = predict_json(patients, predictor)
    finally:
        lightgbm.LGBMClassifier.predict_proba = original

    ref = os.path.join(out_dir, "reference")
    os.makedirs(ref, exist_ok=True)
    summary = {"predict_json_output": result, "calls": {}}
    for name, c in calls.items():
        # Binary matrices: rows x cols float64, NaN for missing. Little-endian.
        c["X"].tofile(os.path.join(ref, f"{name}.X.f64"))
        c["p"].tofile(os.path.join(ref, f"{name}.p.f64"))
        summary["calls"][name] = {"rows": int(c["X"].shape[0]), "cols": int(c["X"].shape[1]),
                                  "columns": c["columns"], "p": c["p"].tolist()}
    with open(os.path.join(ref, "reference.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--reference", help="patient JSON to run through Python and record")
    a = ap.parse_args()

    from zcor.classifier import load_zcor_classifier
    predictor = load_zcor_classifier(a.model)
    m = export(predictor, a.out_dir)
    print(f"exported {len(m['base_models'])} base models + FINAL to {a.out_dir}", file=sys.stderr)
    if a.reference:
        s = capture_reference(predictor, a.reference, a.out_dir)
        print(json.dumps(s["predict_json_output"]), file=sys.stderr)


if __name__ == "__main__":
    main()
