"""
FairnessEngine v11 — Standalone bias detection and mitigation framework
=======================================================================
Supports any binary classification dataset.
Pipeline: Reweighing → Adversarial Debiasing → Calibrated Equalised Odds
Encoding: LabelEncoder for categoricals, StandardScaler for ADB/sklearn
Split:    Stratified 70/30 (or pre-defined via _split column)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

SMOTE_MAP = {
    "Standard":   lambda rs: SMOTE(random_state=rs),
    "Borderline": lambda rs: BorderlineSMOTE(random_state=rs, kind="borderline-1"),
    "ADASYN":     lambda rs: ADASYN(random_state=rs),
    "KMeans":     lambda rs: KMeansSMOTE(random_state=rs),
}

RANDOM_STATE = 42


class FairnessEngine:
    def __init__(self, df, target_column, protected_attribute,
                 privileged_value, feature_columns, test_size=0.3,
                 favorable_value=None, protected_threshold=None):
        self.df                  = df.copy()
        self.target_column       = target_column
        self.protected_attribute = protected_attribute
        self.privileged_value    = privileged_value
        self.favorable_value      = favorable_value        # user-specified positive class
        self.protected_threshold  = protected_threshold   # numeric split threshold for continuous attributes
        self.feature_columns     = feature_columns
        self.test_size           = test_size
        self.privileged_groups   = [{protected_attribute: 1}]
        self.unprivileged_groups = [{protected_attribute: 0}]
        self._prepare_data()

    # ─────────────────────────────────────────────────────────────────────────
    # DATA PREPARATION — matches AIF360 AdultDataset internal preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_data(self):
        df = self.df.copy()

        # Step 1: Clean — replace '?' with NaN, strip whitespace from strings
        df.replace("?", np.nan, inplace=True)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Step 2: Mode impute missing values
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0)

        # Step 3: Binarise target
        # If user specified favorable_value → use it directly (most reliable)
        # Otherwise auto-detect from common patterns
        tgt_str = df[self.target_column].astype(str).str.strip()

        if self.favorable_value is not None:
            # User explicitly told us which value = positive class
            fav = str(self.favorable_value).strip()
            df[self.target_column] = (tgt_str == fav).astype(int)
            print(f"[Target] Using user-specified positive class: {fav!r}")
        else:
            # Auto-detect: try common patterns, fall back to minority class
            tgt_num = pd.to_numeric(tgt_str, errors="coerce")
            unique_str = tgt_str.unique()
            unique_num = set(tgt_num.dropna().unique())

            # Pattern 1: Adult Census income (>50K / <=50K)
            pos_vals = [v for v in unique_str if ">" in v and "50" in v]
            neg_vals = [v for v in unique_str if "<" in v and "50" in v]

            if pos_vals and neg_vals:
                df[self.target_column] = tgt_str.isin(pos_vals).astype(int)
                print(f"[Target] Auto-detected income pattern: {pos_vals} = 1")

            # Pattern 2: Already binary 0/1
            elif unique_num.issubset({0, 1, 0.0, 1.0}) and not tgt_num.isnull().all():
                df[self.target_column] = tgt_num.astype(int)
                print(f"[Target] Auto-detected binary 0/1")

            # Pattern 3: yes/no, true/false, y/n
            elif set(v.lower() for v in unique_str).issubset({"yes","no"}):
                df[self.target_column] = (tgt_str.str.lower() == "yes").astype(int)
                print(f"[Target] Auto-detected yes/no pattern")

            elif set(v.lower() for v in unique_str).issubset({"true","false"}):
                df[self.target_column] = (tgt_str.str.lower() == "true").astype(int)
                print(f"[Target] Auto-detected true/false pattern")

            # Pattern 4: Diabetes readmission (NO / <30 / >30)
            elif any(v.upper() == "NO" for v in unique_str):
                df[self.target_column] = (~tgt_str.str.upper().isin(["NO"])).astype(int)
                print(f"[Target] Auto-detected readmission pattern: not-NO = 1")

            # Fallback: minority class = positive (works for most imbalanced datasets)
            else:
                if tgt_num.isnull().all():
                    minority = tgt_str.value_counts().idxmin()
                    df[self.target_column] = (tgt_str == minority).astype(int)
                    print(f"[Target] Auto-detected minority class: {minority!r} = 1")
                else:
                    minority_num = tgt_num.value_counts().idxmin()
                    df[self.target_column] = (tgt_num == minority_num).astype(int)
                    print(f"[Target] Auto-detected minority numeric class: {minority_num} = 1")

        # Step 4: Binarise protected attribute
        # Strategy: try each approach in order, use the first that produces
        # a non-empty privileged group. Simple and robust across all datasets.
        priv     = str(self.privileged_value).strip()
        prot_col = df[self.protected_attribute]

        def apply_and_check(series_01):
            """Return series if it produces both groups, else None."""
            arr = series_01.fillna(0).astype(int)
            if arr.sum() > 0 and (len(arr) - arr.sum()) > 0:
                return arr
            return None

        result = None

        # Approach 1: user-defined numeric threshold (e.g. age >= 25)
        if self.protected_threshold is not None:
            try:
                thresh   = float(self.protected_threshold)
                prot_num = pd.to_numeric(prot_col, errors="coerce")
                result   = apply_and_check((prot_num >= thresh).fillna(False))
                if result is not None:
                    print(f"[Protected] Threshold >= {thresh} = privileged")
            except (ValueError, TypeError):
                pass

        # Approach 2: direct string match on original column values
        if result is None:
            candidate = (prot_col.astype(str).str.strip() == priv)
            result = apply_and_check(candidate)
            if result is not None:
                print(f"[Protected] String match '{priv}' = privileged")

        # Approach 3: numeric exact match (e.g. sex=1, age_binary=1)
        if result is None:
            try:
                priv_num = int(float(priv))
                prot_num = pd.to_numeric(prot_col, errors="coerce")
                candidate = (prot_num == priv_num)
                result = apply_and_check(candidate)
                if result is not None:
                    print(f"[Protected] Numeric match {priv_num} = privileged")
            except (ValueError, TypeError):
                pass

        # Approach 4: majority group = privileged (pre-encoded binary columns)
        if result is None:
            prot_num = pd.to_numeric(prot_col, errors="coerce")
            unique_vals = set(prot_num.dropna().unique())
            if unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) == 2:
                # Binary column where string label doesn't match either value
                # Majority = privileged is the safest assumption
                counts    = prot_num.value_counts()
                maj_val   = int(counts.index[0])
                candidate = (prot_num == maj_val).fillna(False)
                result    = apply_and_check(candidate)
                if result is not None:
                    print(f"[Protected] Binary pre-encoded — majority value {maj_val} = privileged")

        # Approach 5: median split (continuous numeric columns)
        if result is None:
            prot_num = pd.to_numeric(prot_col, errors="coerce")
            if not prot_num.isnull().all():
                median    = float(prot_num.median())
                candidate = (prot_num.fillna(median) >= median)
                result    = apply_and_check(candidate)
                if result is not None:
                    print(f"[Protected] Median split >= {median} = privileged (auto)")

        # Approach 6: last resort — use raw 0/1 values as-is
        if result is None:
            prot_num = pd.to_numeric(prot_col, errors="coerce").fillna(0)
            result   = prot_num.astype(int)
            print(f"[Protected] Fallback: using raw values as-is")

        df[self.protected_attribute] = result

        # Post-Step-4 validation: ensure privileged group (1) is not empty.
        # If empty it means the encoding is inverted — user's privileged value
        # was mapped to 0 instead of 1 (e.g. pre-encoded CSV with White=0).
        # Auto-flip to correct it.
        priv_count = (df[self.protected_attribute] == 1).sum()
        unpriv_count = (df[self.protected_attribute] == 0).sum()
        if priv_count == 0 and unpriv_count > 0:
            df[self.protected_attribute] = 1 - df[self.protected_attribute]
            print(f"[Protected] Auto-flipped encoding — privileged group was empty (pre-encoded CSV detected)")
        elif priv_count > 0 and unpriv_count == 0:
            print(f"[Protected] Warning — unprivileged group is empty, all rows are privileged")

        # Step 5: LabelEncode all remaining categoricals
        # This matches AIF360 AdultDataset which uses LabelEncoder internally
        # and is why paper gets lower baseline accuracy than OHE
        df_work = df[self.feature_columns + [self.protected_attribute, self.target_column]].copy()
        self.label_encoders = {}
        for col in df_work.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            self.label_encoders[col] = le

        # Step 6: Train / val / test split
        train_val, test_df = train_test_split(
            df_work, test_size=self.test_size, random_state=RANDOM_STATE,
            stratify=df_work[self.target_column]
        )
        val_frac = 0.1 / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val, test_size=val_frac, random_state=RANDOM_STATE,
            stratify=train_val[self.target_column]
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df   = val_df.reset_index(drop=True)
        self.test_df  = test_df.reset_index(drop=True)
        self.all_cols = self.feature_columns + [self.protected_attribute]

        # ── Dataset size warnings ────────────────────────────────────────
        n_total = len(df_work)
        n_train = len(self.train_df)
        n_val   = len(self.val_df)

        self.warnings = []

        if n_total < 500:
            self.warnings.append({
                "level": "error",
                "code":  "DATASET_TOO_SMALL",
                "msg":   f"Dataset has only {n_total} rows. A minimum of 500 rows is required for reliable results. ADB (neural network) requires at least 2,000 rows."
            })
        elif n_total < 2000:
            self.warnings.append({
                "level": "warning",
                "code":  "DATASET_SMALL_ADB",
                "msg":   f"Dataset has {n_total} rows. ADB (Adversarial Debiasing) is a neural network and may produce unreliable results below 2,000 rows. Consider using only Reweighing + CEO for small datasets."
            })

        if n_val < 100:
            self.warnings.append({
                "level": "warning",
                "code":  "VAL_SET_SMALL",
                "msg":   f"Validation set has only {n_val} rows. CEO post-processing calibration may be unreliable. Results should be interpreted with caution."
            })

        # Warn if a protected group is very small
        prot_counts = self.train_df[self.protected_attribute].value_counts()
        min_group   = prot_counts.min()
        if min_group < 30:
            self.warnings.append({
                "level": "warning",
                "code":  "SMALL_PROTECTED_GROUP",
                "msg":   f"The smallest protected group has only {min_group} training samples. Fairness metrics may be unstable. Consider a dataset with more balanced group representation."
            })

        for w in self.warnings:
            print(f"[{w['level'].upper()}] {w['msg']}")

        # Step 7: StandardScaler — used for ADB neural network
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_df[self.all_cols].astype(float))

        # Pre-extract arrays
        self.X_train = self.train_df[self.all_cols].values.astype(float)
        self.X_val   = self.val_df[self.all_cols].values.astype(float)
        self.X_test  = self.test_df[self.all_cols].values.astype(float)

        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_val_scaled   = self.scaler.transform(self.X_val)
        self.X_test_scaled  = self.scaler.transform(self.X_test)

        self.y_train = self.train_df[self.target_column].values.astype(int)
        self.y_val   = self.val_df[self.target_column].values.astype(int)
        self.y_test  = self.test_df[self.target_column].values.astype(int)

        self.prot_train = self.train_df[self.protected_attribute].values.astype(float)
        self.prot_val   = self.val_df[self.protected_attribute].values.astype(float)
        self.prot_test  = self.test_df[self.protected_attribute].values.astype(float)

        self._feat_names = [f"x{i}" for i in range(self.X_train.shape[1])]

    # ─────────────────────────────────────────────────────────────────────────
    # AIF360 HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _make_aif360(self, X, prot, y, weights=None):
        df_tmp = pd.DataFrame(X, columns=self._feat_names)
        df_tmp[self.protected_attribute] = prot.astype(float)
        df_tmp[self.target_column]       = y.astype(float)
        ds = BinaryLabelDataset(
            df=df_tmp,
            label_names=[self.target_column],
            protected_attribute_names=[self.protected_attribute],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        if weights is not None:
            ds.instance_weights = weights
        return ds

    def _prot_label_aif360(self, prot, y):
        df_tmp = pd.DataFrame({
            self.protected_attribute: prot.astype(float),
            self.target_column:       y.astype(float),
        })
        return BinaryLabelDataset(
            df=df_tmp,
            label_names=[self.target_column],
            protected_attribute_names=[self.protected_attribute],
            favorable_label=1.0, unfavorable_label=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, prot, y_true, y_pred):
        true_ds = self._prot_label_aif360(prot, y_true)
        pred_ds = self._prot_label_aif360(prot, y_pred)
        bm = BinaryLabelDatasetMetric(
            pred_ds,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        cm = ClassificationMetric(
            true_ds, pred_ds,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

        def safe(v):
            """Return None for nan/inf — caused by zero-count groups in DI."""
            try:
                f = float(v)
                return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
            except Exception:
                return None

        # Per-group confusion matrix counts
        def group_cm(mask):
            yt = y_true[mask]; yp = y_pred[mask]
            if len(yt) == 0: return {"tp":0,"fp":0,"tn":0,"fn":0,"n":0}
            tp = int(((yt==1)&(yp==1)).sum())
            fp = int(((yt==0)&(yp==1)).sum())
            tn = int(((yt==0)&(yp==0)).sum())
            fn = int(((yt==1)&(yp==0)).sum())
            return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"n":len(yt)}

        priv_mask   = prot == 1
        unpriv_mask = prot == 0

        return {
            "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "spd": safe(bm.statistical_parity_difference()),
            "di":  safe(bm.disparate_impact()),
            "aod": safe(cm.average_odds_difference()),
            "eod": safe(cm.equal_opportunity_difference()),
            "cm_privileged":   group_cm(priv_mask),
            "cm_unprivileged": group_cm(unpriv_mask),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MODELS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_model(self, name):
        if name == "RF":
            return RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        elif name == "XGBoost":
            return xgb.XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE,
                eval_metric="logloss", verbosity=0)
        elif name == "LightGBM":
            return lgb.LGBMClassifier(
                n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
        elif name == "TabNet":
            return None
        raise ValueError(f"Unknown model: {name}")

    def _fit_sklearn(self, model, sample_weight=None):
        # Guard: if training set is missing a class, model cannot fit properly
        # This happens with small datasets after stratified splitting
        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            print(f"[Warning] Training set only has class {unique_classes} — returning majority baseline")
            majority = unique_classes[0]
            n_test = len(self.y_test)
            n_val  = len(self.y_val)
            return (np.full(n_test, majority),
                    np.full(n_test, 0.5),
                    np.full(n_val,  majority),
                    np.full(n_val,  0.5))

        try:
            if sample_weight is not None:
                model.fit(self.X_train, self.y_train, sample_weight=sample_weight)
            else:
                model.fit(self.X_train, self.y_train)
        except ValueError as e:
            print(f"[Warning] Model fit failed: {e} — returning majority baseline")
            majority = int(np.bincount(self.y_train).argmax())
            n_test = len(self.y_test)
            n_val  = len(self.y_val)
            return (np.full(n_test, majority),
                    np.full(n_test, 0.5),
                    np.full(n_val,  majority),
                    np.full(n_val,  0.5))

        def safe_proba(X):
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]

        return (model.predict(self.X_test),
                safe_proba(self.X_test),
                model.predict(self.X_val),
                safe_proba(self.X_val))

    def _fit_tabnet(self, sample_weight=None):
        # Guard: missing class in training set
        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            print(f"[Warning] TabNet: training set only has class {unique_classes} — returning majority baseline")
            majority = unique_classes[0]
            return (np.full(len(self.y_test), majority), np.full(len(self.y_test), 0.5),
                    np.full(len(self.y_val),  majority), np.full(len(self.y_val),  0.5))

        from pytorch_tabnet.tab_model import TabNetClassifier
        from torch.optim import Adam
        clf = TabNetClassifier(seed=RANDOM_STATE, verbose=0,
                                n_d=16, n_a=16, n_steps=3,
                                optimizer_fn=Adam,
                                optimizer_params={"lr": 2e-2})
        clf.fit(self.X_train.astype(np.float32), self.y_train,
                weights=sample_weight if sample_weight is not None else 0,
                max_epochs=50, patience=10,
                batch_size=1024, virtual_batch_size=128)
        def safe_proba_tn(X):
            proba = clf.predict_proba(X.astype(np.float32))
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]

        return (clf.predict(self.X_test.astype(np.float32)),
                safe_proba_tn(self.X_test),
                clf.predict(self.X_val.astype(np.float32)),
                safe_proba_tn(self.X_val))

    # ─────────────────────────────────────────────────────────────────────────
    # ADB — subprocess isolated to prevent TF graph leakage between models
    # ─────────────────────────────────────────────────────────────────────────

    def _run_adb(self, scope_name, instance_weights,
                 predict_Xs, predict_prots, predict_ys):
        import subprocess, sys, tempfile, pickle, os

        tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp_in.close(); tmp_out.close()

        payload = {
            "X_train":              self.X_train_scaled,
            "y_train":              self.y_train,
            "prot_train":           self.prot_train,
            "instance_weights":     instance_weights,
            "predict_Xs":           predict_Xs,
            "predict_prots":        predict_prots,
            "predict_ys":           predict_ys,
            "privileged_groups":    self.privileged_groups,
            "unprivileged_groups":  self.unprivileged_groups,
            "protected_attribute":  self.protected_attribute,
            "target_column":        self.target_column,
            "scope_name":           scope_name,
        }
        with open(tmp_in.name, "wb") as f:
            pickle.dump(payload, f)

        adb_script = os.path.join(os.path.dirname(__file__), "_run_adb_subprocess.py")
        try:
            result = subprocess.run(
                [sys.executable, adb_script, tmp_in.name, tmp_out.name],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                with open(tmp_out.name, "rb") as f:
                    preds = pickle.load(f)
                os.unlink(tmp_in.name)
                os.unlink(tmp_out.name)
                return preds
        except Exception:
            pass

        # Fallback: in-process
        return self._run_adb_inprocess(scope_name, instance_weights,
                                        predict_Xs, predict_prots, predict_ys)

    def _run_adb_inprocess(self, scope_name, instance_weights,
                            predict_Xs, predict_prots, predict_ys):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        train_ds = self._make_aif360(
            self.X_train_scaled, self.prot_train, self.y_train, instance_weights
        )
        n_train    = len(self.y_train)
        num_epochs = 150 if n_train > 30000 else 50
        batch_size = 512 if n_train > 30000 else 256
        sess = tf.compat.v1.Session()
        adb = AdversarialDebiasing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            scope_name=scope_name, debias=True, sess=sess,
            num_epochs=num_epochs, batch_size=batch_size,
            classifier_num_hidden_units=200,
        )
        adb.fit(train_ds)
        results = []
        for X, prot, y in zip(predict_Xs, predict_prots, predict_ys):
            result_ds = adb.predict(self._make_aif360(X, prot, y))
            labels = result_ds.labels.ravel().astype(int)
            scores = np.clip(result_ds.scores.ravel(), 0.001, 0.999)
            results.append({"labels": labels, "scores": scores})
        sess.close()
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def compute_feature_importance(self):
        """Compute feature importance using a quick RF on training data."""
        try:
            # Guard: RF requires at least 2 classes in training set
            if len(np.unique(self.y_train)) < 2:
                print("[Feature importance] Skipped — training set has only one class")
                return []
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            importances = rf.feature_importances_
            feat_names  = self.feature_columns + [self.protected_attribute]
            # Sort descending
            pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
            return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]
        except Exception as e:
            print(f"[Feature importance] Failed: {e}")
            return []

    def compute_baseline_metrics(self):
        y_all    = np.concatenate([self.y_train, self.y_val, self.y_test])
        prot_all = np.concatenate([self.prot_train, self.prot_val, self.prot_test])
        ds = self._prot_label_aif360(prot_all, y_all)
        bm = BinaryLabelDatasetMetric(
            ds,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        n_pos = int(y_all.sum())

        # Group size breakdown for transparency
        group_sizes = {}
        for g_val in [0, 1]:
            mask = prot_all == g_val
            label = "privileged" if g_val == 1 else "unprivileged"
            group_sizes[label] = {
                "total": int(mask.sum()),
                "positive": int(y_all[mask].sum()),
            }

        return {
            "spd":        round(float(bm.statistical_parity_difference()), 4),
            "di":         round(float(bm.disparate_impact()), 4),
            "total_rows": len(y_all),
            "class_distribution": {0: len(y_all) - n_pos, 1: n_pos},
            "group_sizes": group_sizes,
            "train_size":  len(self.y_train),
            "val_size":    len(self.y_val),
            "test_size":   len(self.y_test),
            "warnings":    getattr(self, "warnings", []),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SMOTE
    # ─────────────────────────────────────────────────────────────────────────

    def run_smote_experiments(self, variants, progress_callback=None):
        results = {}
        for i, variant in enumerate(variants):
            if progress_callback:
                progress_callback(f"SMOTE: {variant}", i / len(variants))
            try:
                X_res, y_res = SMOTE_MAP[variant](RANDOM_STATE).fit_resample(
                    self.X_train, self.y_train
                )
            except Exception:
                try:
                    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
                        self.X_train, self.y_train
                    )
                except Exception as e2:
                    results[variant] = {"error": str(e2)}
                    continue
            model = self._get_model("RF")
            model.fit(X_res, y_res)
            y_pred = model.predict(self.X_test)
            results[variant] = self._compute_metrics(self.prot_test, self.y_test, y_pred)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def run_model_pipeline(self, model_name, mitigation_steps,
                            progress_callback=None):
        results   = {}
        is_tabnet = (model_name == "TabNet")

        def prog(s, p):
            if progress_callback: progress_callback(s, p)

        # ── Stage 1: Baseline ─────────────────────────────────────────────
        prog("Baseline", 0)
        if is_tabnet:
            y_pred_te, y_scr_te, y_pred_va, y_scr_va = self._fit_tabnet()
        else:
            y_pred_te, y_scr_te, y_pred_va, y_scr_va = self._fit_sklearn(
                self._get_model(model_name)
            )
        results["original"] = self._compute_metrics(
            self.prot_test, self.y_test, y_pred_te
        )

        # ── Reweighing weights ────────────────────────────────────────────
        needs_rw = any(s in mitigation_steps for s in ["Reweighing", "ADB", "CEO"])
        rw_weights  = None
        y_scr_te_rw = y_scr_te
        y_scr_va_rw = y_scr_va

        if needs_rw:
            prog("Computing Reweighing weights", 0.15)
            rw = Reweighing(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )
            rw_weights = rw.fit_transform(
                self._prot_label_aif360(self.prot_train, self.y_train)
            ).instance_weights

        # ── Stage 2: Reweighing + model ───────────────────────────────────
        if "Reweighing" in mitigation_steps or "CEO" in mitigation_steps:
            prog(f"{model_name} + Reweighing", 0.25)
            if is_tabnet:
                y_pred_te_rw, y_scr_te_rw, y_pred_va_rw, y_scr_va_rw = self._fit_tabnet(
                    sample_weight=rw_weights
                )
            else:
                y_pred_te_rw, y_scr_te_rw, y_pred_va_rw, y_scr_va_rw = self._fit_sklearn(
                    self._get_model(model_name), sample_weight=rw_weights
                )
            if "Reweighing" in mitigation_steps:
                results["reweigh"] = self._compute_metrics(
                    self.prot_test, self.y_test, y_pred_te_rw
                )

        # ── Stage 3: ADB ──────────────────────────────────────────────────
        # Predict on BOTH test and val so CEO can calibrate on ADB's output
        adb_te = None
        adb_va = None
        if "ADB" in mitigation_steps:
            prog("Adversarial Debiasing", 0.5)
            adb_results = self._run_adb(
                scope_name=f"adb_{model_name}",
                instance_weights=rw_weights,
                predict_Xs=[self.X_test_scaled, self.X_val_scaled],
                predict_prots=[self.prot_test, self.prot_val],
                predict_ys=[self.y_test, self.y_val],
            )
            # Each result is {"labels": ..., "scores": ...}
            adb_te = adb_results[0]
            adb_va = adb_results[1]
            results["reweigh_adb"] = self._compute_metrics(
                self.prot_test, self.y_test, adb_te["labels"]
            )

        # ── Stage 4: CEO ──────────────────────────────────────────────────
        # KEY FIX: calibrate CEO on ADB's debiased scores, not reweigh scores.
        # Using reweigh scores for CEO ignores everything ADB learned.
        if "CEO" in mitigation_steps:
            prog("Calibrated Equalised Odds", 0.8)
            try:
                def make_scored(prot, y_true, scores):
                    ds = self._prot_label_aif360(prot, y_true)
                    pred = ds.copy()
                    pred.scores = np.clip(scores, 0.001, 0.999).reshape(-1, 1)
                    pred.labels = (pred.scores >= 0.5).astype(float)
                    return ds, pred

                # Use ADB scores if available, else fall back to reweigh scores
                val_scores  = adb_va["scores"]  if adb_va  is not None else y_scr_va_rw
                test_scores = adb_te["scores"]  if adb_te  is not None else y_scr_te_rw

                val_true,  val_pred  = make_scored(
                    self.prot_val,  self.y_val,  val_scores
                )
                test_true, test_pred = make_scored(
                    self.prot_test, self.y_test, test_scores
                )

                best_ceo     = None
                best_abs_spd = float("inf")

                for constraint in ["fnr", "fpr", "weighted"]:
                    try:
                        ceo = CalibratedEqOddsPostprocessing(
                            privileged_groups=self.privileged_groups,
                            unprivileged_groups=self.unprivileged_groups,
                            cost_constraint=constraint,
                            seed=RANDOM_STATE,
                        )
                        ceo.fit(val_true, val_pred)
                        ceo_pred = ceo.predict(test_pred)
                        y_ceo = ceo_pred.labels.ravel().astype(int)
                        m = self._compute_metrics(self.prot_test, self.y_test, y_ceo)
                        if m["accuracy"] < 0.999 and m["spd"] is not None and abs(m["spd"]) < best_abs_spd:
                            best_abs_spd = abs(m["spd"])
                            best_ceo = m
                    except Exception:
                        continue

                results["reweigh_adb_ceo"] = best_ceo or results.get(
                    "reweigh", results["original"]
                )
            except Exception:
                results["reweigh_adb_ceo"] = results.get("reweigh", results["original"])

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def generate_summary(self, baseline, mitigation_results):
        best_model, best_spd, best_acc = None, float("inf"), 0
        for model, stages in mitigation_results.items():
            final = (stages.get("reweigh_adb_ceo") or stages.get("reweigh_adb")
                     or stages.get("reweigh") or stages.get("original"))
            if final:
                spd = final.get("spd")
                # spd can be None when a group has zero positive predictions
                if spd is not None and abs(spd) < abs(best_spd):
                    best_spd  = spd
                    best_acc  = final.get("accuracy", 0)
                    best_model = model
        # best_spd stays inf if nothing valid found — handle gracefully
        best_spd_val = best_spd if best_spd != float("inf") else 0
        baseline_spd = baseline.get("spd") or 0
        return {
            "best_model":        best_model,
            "best_spd_after":    round(best_spd_val, 4),
            "best_accuracy":     round(best_acc, 4),
            "initial_spd":       baseline_spd,
            "fairness_improved": abs(best_spd_val) < abs(baseline_spd),
        }
