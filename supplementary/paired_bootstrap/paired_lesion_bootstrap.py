"""Supplementary paired lesion-grouped macro-F1 analysis of saved predictions.

No model training or prediction generation is performed.
Run with Python 3 and NumPy/pandas installed.
"""
import argparse
import hashlib
import json
import platform
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
MODELS = ["CNN_300_deeper", "Sugeno_ANFIS", "Sugeno_ANFIS_coverage"]
MANIFEST_HASH = "14d91a82418a54ee285ce399b3ae45bfadc10a5b54f51eb82df96dbcc5e2cf2d"
EXPECTED = np.array([0.7080077499067052, 0.7077740981568713, 0.7102589871493918])
SEED = 42
REPETITIONS = 2000


def macro_f1(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    ).mean(axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original completed reports ZIP")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(args.archive) as archive:
        assert archive.testzip() is None, "Corrupt archive"
        manifest_bytes = archive.read("tables/generated_lesion_grouped_split.csv")
        assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_HASH
        manifest = pd.read_csv(BytesIO(manifest_bytes))
        assert len(manifest) == 10015 and manifest.image_id.is_unique
        assert manifest.groupby("split").size().to_dict() == {
            "train": 7168, "val": 1434, "test": 1413,
        }
        assert manifest.groupby("lesion_id").split.nunique().max() == 1
        marker = json.loads(archive.read("tables/sealed_test_evaluated.json"))
        assert marker["split_sha256"] == MANIFEST_HASH
        assert set(marker["models_evaluated"]) == set(MODELS)
        assert marker["test_evaluated"] is True
        assert marker["test_tuning_performed"] is False
        test = manifest.loc[manifest.split == "test"].set_index("image_id").sort_index()
        frames = [
            pd.read_csv(BytesIO(archive.read(f"tables/{model}_test_predictions.csv")))
            .set_index("image_id").sort_index()
            for model in MODELS
        ]

    for frame in frames:
        assert frame.index.is_unique and frame.index.equals(test.index)
        assert frame.lesion_id.equals(test.lesion_id)
        assert frame.true.equals(frames[0].true)
        assert set(frame.true) == set(CLASSES)
        assert set(frame.predicted).issubset(CLASSES)
        probabilities = frame[[f"prob_{c}" for c in CLASSES]].to_numpy()
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        assert np.array_equal(np.array(CLASSES)[probabilities.argmax(axis=1)], frame.predicted)

    lesions, inverse = np.unique(frames[0].lesion_id.to_numpy(), return_inverse=True)
    n_groups = len(lesions)
    assert n_groups == 1068
    true = np.array([CLASSES.index(v) for v in frames[0].true])
    # Store one confusion matrix per lesion and classifier.
    counts = np.zeros((len(MODELS), n_groups, len(CLASSES), len(CLASSES)), dtype=np.int32)
    for i, frame in enumerate(frames):
        predicted = np.array([CLASSES.index(v) for v in frame.predicted])
        np.add.at(counts[i], (inverse, true, predicted), 1)

    observed = macro_f1(counts.sum(axis=1))
    assert np.allclose(observed, EXPECTED, atol=1e-12, rtol=0)

    rng = np.random.default_rng(SEED)
    differences = np.empty((REPETITIONS, len(MODELS) - 1))
    for i in range(REPETITIONS):
        weights = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        # Same sampled lesions (with multiplicity) for every classifier.
        sample = np.einsum("g,mgij->mij", weights, counts)
        scores = macro_f1(sample)
        differences[i] = scores[1:] - scores[0]

    rows = []
    for i, model in enumerate(MODELS[1:]):
        lower, upper = np.percentile(differences[:, i], [2.5, 97.5])
        rows.append({
            "comparison": f"{model} minus CNN_300_deeper",
            "observed_macro_f1_difference": observed[i + 1] - observed[0],
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "bootstrap_repetitions": REPETITIONS,
            "lesion_groups": n_groups,
            "seed": SEED,
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "paired_lesion_bootstrap_macro_f1.csv",
                  index=False, float_format="%.8f")
    provenance = {
        "analysis": "Supplementary paired lesion-grouped bootstrap of saved predictions",
        "input_archive": args.archive.name,
        "input_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "split_manifest_sha256": MANIFEST_HASH,
        "original_evaluation_timestamp_utc": marker["evaluation_timestamp_utc"],
        "test_images": len(test),
        "lesion_groups": n_groups,
        "repetitions": REPETITIONS,
        "random_seed": SEED,
        "interval": "95% percentile",
        "class_order": CLASSES,
        "no_model_retraining": True,
        "no_test_based_model_selection": True,
        "supplementary_analysis_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.output_dir / "paired_lesion_bootstrap_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
