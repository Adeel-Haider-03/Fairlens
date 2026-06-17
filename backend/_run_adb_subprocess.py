"""
_run_adb_subprocess.py
======================
Runs AdversarialDebiasing in a completely fresh Python process.
Guarantees TF1 graph isolation between model runs — no variable leakage.

Returns both .labels (binary) AND .scores (probability) for each predict set
so that CEO can properly calibrate on ADB's debiased output.

Usage: python _run_adb_subprocess.py <input.pkl> <output.pkl>
"""

import sys, pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "rb") as f:
        p = pickle.load(f)

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.inprocessing import AdversarialDebiasing

    feat_names = [f"x{i}" for i in range(p["X_train"].shape[1])]
    protected  = p["protected_attribute"]
    target     = p["target_column"]

    def make_ds(X, prot, y, weights=None):
        df = pd.DataFrame(X, columns=feat_names)
        df[protected] = prot.astype(float)
        df[target]    = y.astype(float)
        ds = BinaryLabelDataset(
            df=df,
            label_names=[target],
            protected_attribute_names=[protected],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        if weights is not None:
            ds.instance_weights = weights
        return ds

    train_ds = make_ds(
        p["X_train"], p["prot_train"], p["y_train"], p["instance_weights"]
    )

    # ── Hyperparameters ────────────────────────────────────────────────────
    # 150 epochs gives ADB enough training iterations on the 45K dataset.
    # Paper reports high accuracy (~0.963) after ADB which requires convergence.
    # batch_size=512 → stable gradients, fewer but higher-quality updates.
    n_train     = len(p["y_train"])
    num_epochs  = 150 if n_train > 30000 else 50
    batch_size  = 512 if n_train > 30000 else 256

    sess = tf.compat.v1.Session()
    adb  = AdversarialDebiasing(
        privileged_groups=p["privileged_groups"],
        unprivileged_groups=p["unprivileged_groups"],
        scope_name=p["scope_name"],
        debias=True, sess=sess,
        num_epochs=num_epochs,
        batch_size=batch_size,
        classifier_num_hidden_units=200,
    )
    adb.fit(train_ds)

    # Return both binary labels AND probability scores so CEO can calibrate
    # on ADB's debiased output (not the reweighing model's scores)
    preds = []
    for X, prot, y in zip(p["predict_Xs"], p["predict_prots"], p["predict_ys"]):
        result_ds = adb.predict(make_ds(X, prot, y))
        labels = result_ds.labels.ravel().astype(int)
        # .scores holds the classifier's probability output from ADB
        # Clip to valid probability range for CEO calibration
        scores = np.clip(result_ds.scores.ravel(), 0.001, 0.999)
        preds.append({"labels": labels, "scores": scores})

    sess.close()

    with open(output_path, "wb") as f:
        pickle.dump(preds, f)


if __name__ == "__main__":
    main()
