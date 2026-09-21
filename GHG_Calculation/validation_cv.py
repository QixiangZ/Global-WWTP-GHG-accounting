"""Ten-fold cross-validation of the classifier and Monte Carlo EF validation."""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
import time
import os
import json
import random
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from datetime import datetime
import multiprocessing

from paths import OUTPUT_DIR, inp

warnings.filterwarnings('ignore')

B             = 1000
N_SIMULATIONS = 10000
N_FOLDS       = 10

# --- input ---------------------------------------------------------------
INPUT_FILE  = inp('combined_wwtp.xlsx')
N2O_EF_FILE = inp('N2O_EF.xlsx')
CH4_EF_FILE = inp('CH4_EF.xlsx')
FB_FILE     = inp('FB.xlsx')

N2O_EF_COL = 'N2OEF'
CH4_EF_COL = 'CH4EF'
FB_COL     = 'FB'

# --- output --------------------------------------------------------------
STAGE1_DIR = OUTPUT_DIR
STAGE2_DIR = OUTPUT_DIR

PROGRESS_FILE = os.path.join(STAGE1_DIR, 'validation_progress.json')

AVAILABLE_CORES = int(os.environ.get('NCPUS', 0)) or cpu_count()
print(f"{AVAILABLE_CORES} cores detected")
if AVAILABLE_CORES >= 16:
    N_JOBS_BOOTSTRAP = 8;  N_JOBS_XGB = 2
elif AVAILABLE_CORES >= 8:
    N_JOBS_BOOTSTRAP = 4;  N_JOBS_XGB = 2
else:
    N_JOBS_BOOTSTRAP = 2;  N_JOBS_XGB = 1
N_CORES_MC = min(16, AVAILABLE_CORES)
print(f"config: bootstrap={N_JOBS_BOOTSTRAP}, xgb threads={N_JOBS_XGB}, mc cores={N_CORES_MC}")

LABEL_COL = 'Biotreat Type'
POP_COL   = 'Current Served Population (estimate)'

FEATURE_COLS = [
    'Current Served Population (estimate)', 'Population_Density', 'Urban_cluster_share',
    'GDP_PerCapita_PPP', 'GNI_per_capita', 'Extreme Monthly Average Temperature (High)',
    'Decadal Average Precipitation', 'Agriculture_Area', 'Nitrogen_fertilizer',
    'Pasture_Area', 'Building_Area', 'Water_Area', 'Aridity_Index', 'HDI',
    'Social_progress_index', 'Developing_country', 'City_Population', 'Region_Area',
    'Rural_share', 'Income_level', 'Small_island_developing_state',
    'Extreme Monthly Average Temperature (Low)', 'Decadal Average Temperature',
    'Extreme Monthly Average Precipitation (High)',
    'Extreme Monthly Average Precipitation (Low)', 'Average Atmospheric Pressure'
]

TREATMENT_METHODS = {
    'A2O', 'AO', 'Anaerobic', 'BAF', 'BNR', 'CAS', 'EA', 'Lagoon',
    'MBR', 'MLE', 'OD', 'Pond', 'RBC', 'SBR', 'TF', 'Wetland'
}

CH4_DEFAULT_VALUES      = {'Pond': 0.102, 'Lagoon': 0.102, 'Anaerobic': 0.408}
CH4_DEFAULT_UNCERTAINTY = 0.30
N2O_DEFAULT_VALUES      = {'Anaerobic': 0}
ADVANCED_TYPES          = {'AO', 'BAF', 'BNR', 'EA', 'MBR', 'MLE', 'A2O', 'SBR', 'OD'}


def load_progress():
    """Read which folds are already done."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
            completed = set(data.get('completed_folds', []))
            print(f"  resuming, folds already done: {sorted(completed)}")
            return completed
        except Exception:
            pass
    return set()


def save_progress(completed_folds):
    """Persist the completed fold list."""
    data = {
        'completed_folds': sorted(list(completed_folds)),
        'last_update': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_population_bucket(pop):
    """Assign a plant to a served-population bucket."""
    if pd.isna(pop):
        return 'medium: 2000-10000'
    if pop <= 2000:
        return 'small: 0-2000'
    elif pop <= 10000:
        return 'medium: 2000-10000'
    else:
        return 'large: 10000+'


def get_bucket_params(bucket):
    """Return the XGBoost hyperparameters for one bucket."""
    base = {
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 1
    }
    bucket = bucket.lower()
    if 'small' in bucket:
        base.update({
            'max_depth': 8, 'n_estimators': 100, 'learning_rate': 0.10,
            'min_child_weight': 5, 'reg_lambda': 5
        })
    elif 'medium' in bucket:
        base.update({
            'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.10,
            'min_child_weight': 1, 'reg_lambda': 1
        })
    else:
        base.update({
            'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.10,
            'min_child_weight': 1, 'reg_lambda': 1
        })
    return base


IQR_K = 1.5
MIN_GROUP_N = 4
FB_NORM_FLOOR = 0.10

FB_SAMPLING_FLOOR = None

TYPE_MEAN_NONF_RULE = '!=F'


def iqr_clean_by_type(df, screen_col, group_col='Type', k=IQR_K, min_n=MIN_GROUP_N,
                      label='', verbose=True):
    """Remove NaN and IQR outliers within each Type group."""
    n_input = len(df)

    n_group_nan = int(df[group_col].isna().sum())
    if n_group_nan and verbose:
        print(f"  ! '{group_col}': {n_group_nan} missing, cannot group, dropped")
    df = df.dropna(subset=[group_col])

    nan_mask = df[screen_col].isna()
    n_nan = int(nan_mask.sum())
    if n_nan and verbose:
        print(f"  ! '{screen_col}': {n_nan} NaN, dropped")
        print(f"     by Type: {df[nan_mask].groupby(group_col).size().to_dict()}")
    d = df[~nan_mask]

    rows, bad = [], []
    for t, grp in d.groupby(group_col):
        vals = grp[screen_col]
        n = len(grp)
        rec = {'Type': t, 'N_eligible': n,
               'Mean_before': vals.mean(), 'Std_before': vals.std()}

        if n < min_n:
            rec.update({'Method': 'insufficient_data', 'Q1': np.nan, 'Q3': np.nan,
                        'IQR': np.nan, 'Lower': np.nan, 'Upper': np.nan,
                        'N_removed': 0, 'Pct_removed': 0.0, 'N_kept': n,
                        'Mean_after': vals.mean()})
            rows.append(rec)
            continue

        Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            rec.update({'Method': 'iqr_zero_skipped', 'Q1': Q1, 'Q3': Q3, 'IQR': 0.0,
                        'Lower': np.nan, 'Upper': np.nan,
                        'N_removed': 0, 'Pct_removed': 0.0, 'N_kept': n,
                        'Mean_after': vals.mean()})
            rows.append(rec)
            continue

        lo, hi = Q1 - k * IQR, Q3 + k * IQR
        mask = ((vals < lo) | (vals > hi)).values
        bad.extend(grp.index[mask])
        kept = vals[~mask]

        rec.update({'Method': 'IQR', 'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'Lower': lo, 'Upper': hi,
                    'N_removed': int(mask.sum()), 'Pct_removed': mask.sum() / n * 100,
                    'N_kept': len(kept),
                    'Mean_after': kept.mean() if len(kept) else np.nan})
        rows.append(rec)

    cleaned = d.drop(bad)
    summary = pd.DataFrame(rows)

    if verbose:
        n_out = n_input - len(cleaned)
        print(f"  IQR {label or screen_col}: {n_input} -> {len(cleaned)}"
              f" (NaN {n_nan + n_group_nan} + outliers {len(bad)},"
              f" {n_out/max(n_input,1):.2%})")
        if len(summary):
            show = ['Type', 'Method', 'N_eligible', 'Q1', 'Q3', 'IQR', 'Lower', 'Upper',
                    'N_removed', 'Pct_removed', 'Mean_before', 'Mean_after']
            print(summary[show].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
            skipped = summary[summary['Method'] != 'IQR']
            if len(skipped):
                print(f"  {len(skipped)} Type groups were not cleaned: "
                      f"{skipped[['Type', 'Method', 'N_eligible']].to_dict('records')}")
            empty = summary[summary['N_kept'] == 0]
            if len(empty):
                print(f"  ! emptied Type: {sorted(empty['Type'].tolist())}")
            high = summary[summary['Pct_removed'] > 50]
            if len(high):
                print(f"  ! >50% removed for Type: {sorted(high['Type'].tolist())}")

    return cleaned, summary


def _ratio_means(fb_by_type, floor=FB_NORM_FLOOR):
    """Per-type and global mean of the F/B ratio pool."""
    def _m(vals):
        """Mean of a sample array."""
        a = np.asarray(list(vals), dtype=float)
        a = a[~np.isnan(a)]
        if floor is not None:
            a = np.clip(a, floor, None)
        return float(a.mean()) if len(a) else np.nan

    per_type = {t: _m(v) for t, v in fb_by_type.items()}
    all_flat = [v for vv in fb_by_type.values() for v in vv]
    return per_type, (_m(all_flat) if all_flat else 1.0)


def load_ef_data():
    """Load and clean the EF tables and derive per-type reference means."""
    print("\nloading EF data...")

    ratio_df = pd.read_excel(FB_FILE)
    fb_by_type = {}
    for _, row in ratio_df.iterrows():
        fb_by_type.setdefault(row['Type'], []).append(row[FB_COL])
    print(f"  FB types: {list(fb_by_type.keys())}")

    ratio_mean_by_type, ratio_global_mean = _ratio_means(fb_by_type)
    print(f"  FB normalising means: {len(ratio_mean_by_type)} Types, "
          f"global fallback = {ratio_global_mean:.4g} (floor {FB_NORM_FLOOR})")

    all_flat = np.asarray([v for vv in fb_by_type.values() for v in vv], dtype=float)
    n_low = int((all_flat < (FB_NORM_FLOOR or 0)).sum())
    if n_low:
        print(f"  ! {n_low} FB values below {FB_NORM_FLOOR}. The normalising mean is clipped,")
        print("    but sample_n2o_ef divides by a randomly drawn FB value, so a draw near zero")
        print("    inflates the EF. See FB_SAMPLING_FLOOR.")

    if FB_SAMPLING_FLOOR is not None:
        fb_by_type = {t: [max(float(v), FB_SAMPLING_FLOOR) for v in vv]
                         for t, vv in fb_by_type.items()}
        print(f"  sampling FB values clipped to {FB_SAMPLING_FLOOR}")

    print("\n  --- N2O ---")
    n2o_df = pd.read_excel(N2O_EF_FILE)

    if 'F/B' not in n2o_df.columns:
        print("  ! no 'F/B' column, every record treated as non-F")
        n2o_df['F/B'] = np.nan

    is_F = (n2o_df['F/B'] == 'F')
    r = n2o_df['Type'].map(lambda t: ratio_mean_by_type.get(t, ratio_global_mean))
    r = r.replace(0, np.nan).fillna(ratio_global_mean)
    n2o_df['_screen'] = np.where(is_F, n2o_df[N2O_EF_COL], n2o_df[N2O_EF_COL] / r)

    mixed = [t for t, g in n2o_df.groupby('Type')
             if (g['F/B'] == 'F').any() and (~(g['F/B'] == 'F')).any()]
    print(f"  F/non-F split: F={int(is_F.sum())}, non-F={int((~is_F).sum())} | "
          f"{len(mixed)} mixed Types {sorted(mixed)}")

    n2o_df, _ = iqr_clean_by_type(n2o_df, '_screen', label=f'{N2O_EF_COL} (normalised scale)')
    n2o_df = n2o_df.drop(columns=['_screen'])
    n2o_raw = n2o_df.to_dict('records')
    print(f"  N2O EF: {len(n2o_raw)} records")

    print("\n  --- CH4 ---")
    ch4_df = pd.read_excel(CH4_EF_FILE)
    ch4_df, _ = iqr_clean_by_type(ch4_df, CH4_EF_COL, label="CH4_EF_COL")
    ch4_raw = ch4_df.to_dict('records')
    print(f"  CH4 EF: {len(ch4_raw)} records")

    n_rule_diff = sum(1 for rec in n2o_raw
                      if (rec.get('F/B') != 'F') != (rec.get('F/B') == 'B'))
    if n_rule_diff:
        print(f"  ! {n_rule_diff} N2O records have an F/B value that is neither 'F' nor 'B'; "
              f"the two rules disagree on them. Currently using "
              f"TYPE_MEAN_NONF_RULE='{TYPE_MEAN_NONF_RULE}'.")

    def _is_nonF(rec):
        """True when a record is not on the F scale."""
        return (rec.get('F/B') != 'F') if TYPE_MEAN_NONF_RULE == '!=F' \
            else (rec.get('F/B') == 'B')

    n2o_type_mean = {}
    for rec in n2o_raw:
        t, ef = rec['Type'], rec[N2O_EF_COL]
        if _is_nonF(rec):
            ef = ef / ratio_mean_by_type.get(t, ratio_global_mean)
        n2o_type_mean.setdefault(t, []).append(ef)
    n2o_type_mean = {t: float(np.mean(v)) for t, v in n2o_type_mean.items()}
    for t, v in N2O_DEFAULT_VALUES.items():
        n2o_type_mean[t] = float(v)

    ch4_type_mean = {}
    for rec in ch4_raw:
        t = rec['Type']
        if t in CH4_DEFAULT_VALUES:
            continue
        ch4_type_mean.setdefault(t, []).append(rec[CH4_EF_COL])
    ch4_type_mean = {t: float(np.mean(v)) for t, v in ch4_type_mean.items()}
    for t, v in CH4_DEFAULT_VALUES.items():
        ch4_type_mean[t] = float(v)

    print(f"  N2O per-type mean EF: { {k: round(v,4) for k,v in n2o_type_mean.items()} }")
    print(f"  CH4 per-type mean EF: { {k: round(v,4) for k,v in ch4_type_mean.items()} }")
    return n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean


def sample_n2o_ef(treatment_type, n2o_raw, fb_by_type):
    """Draw one N2O EF for a treatment type."""
    if treatment_type in N2O_DEFAULT_VALUES:
        return N2O_DEFAULT_VALUES[treatment_type]
    records = [r for r in n2o_raw if r['Type'] == treatment_type]
    if not records:
        return None
    rec = random.choice(records)
    ef = rec[N2O_EF_COL]
    if rec.get('F/B') != 'F':
        ratios = fb_by_type.get(treatment_type)
        if ratios:
            ef = ef / random.choice(ratios)
        else:
            all_r = [v for vals in fb_by_type.values() for v in vals]
            ef = ef / (np.mean(all_r) if all_r else 1.0)
    return ef


def sample_ch4_ef(treatment_type, ch4_raw):
    """Draw one CH4 EF for a treatment type."""
    if treatment_type in CH4_DEFAULT_VALUES:
        default = CH4_DEFAULT_VALUES[treatment_type]
        std = default * CH4_DEFAULT_UNCERTAINTY
        for _ in range(10):
            v = np.random.normal(default, std)
            if v >= 0:
                return v
        return abs(np.random.normal(default, std))
    records = [r for r in ch4_raw if r['Type'] == treatment_type]
    if not records:
        return None
    return random.choice(records)[CH4_EF_COL]


def bootstrap_worker(args):
    """Fit one bootstrap replicate and return class probabilities."""
    b_idx, X_train_rec, y_enc, X_pred_rec, unique_cats, bucket_params, treatment_levels = args
    try:
        X_train = pd.DataFrame(X_train_rec)
        X_pred  = pd.DataFrame(X_pred_rec)
        np.random.seed(b_idx + 42)
        n = len(X_train)
        idx = np.random.choice(n, size=n, replace=True)
        model = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            n_estimators=bucket_params['n_estimators'],
            max_depth=bucket_params['max_depth'],
            learning_rate=bucket_params['learning_rate'],
            subsample=bucket_params['subsample'],
            colsample_bytree=bucket_params['colsample_bytree'],
            n_jobs=N_JOBS_XGB, random_state=b_idx+42,
            verbosity=0, tree_method='hist', max_bin=256
        )
        model.fit(X_train.iloc[idx], y_enc[idx])
        proba = model.predict_proba(X_pred)

        if treatment_levels is not None:
            allowed = np.array([c in ADVANCED_TYPES for c in unique_cats])
            if np.any(allowed):
                for i, lvl in enumerate(treatment_levels):
                    if pd.notna(lvl) and str(lvl).strip() == 'Advanced':
                        proba[i, ~allowed] = 0
                        s = proba[i].sum()
                        proba[i] = proba[i]/s if s > 0 else (
                            np.where(allowed, 1.0/np.sum(allowed), 0))
        return b_idx, proba
    except Exception as e:
        print(f"  bootstrap {b_idx} failed: {e}")
        return b_idx, np.zeros((len(X_pred_rec), len(unique_cats)))


def run_bootstrap(X_train, y_enc, X_pred, unique_cats, bucket_params, treatment_levels):
    """Run B bootstrap replicates and return (B, n_pred, n_classes)."""
    X_tr_rec  = X_train.to_dict('records')
    X_pr_rec  = X_pred.to_dict('records')
    args_list = [
        (b, X_tr_rec, y_enc, X_pr_rec, unique_cats, bucket_params, treatment_levels)
        for b in range(B)
    ]
    results = np.zeros((B, len(X_pred), len(unique_cats)))
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=N_JOBS_BOOTSTRAP) as ex:
            futures = {ex.submit(bootstrap_worker, a): a[0] for a in args_list}
            for fut in as_completed(futures):
                b_idx = futures[fut]
                try:
                    _, proba = fut.result()
                    results[b_idx] = proba
                    completed += 1
                    if completed % 200 == 0:
                        print(f"    Bootstrap: {completed}/{B}")
                except Exception as e:
                    print(f"    bootstrap {b_idx} result failed: {e}")
    except Exception as e:
        print(f"  parallel failed, running serially: {e}")
        for a in args_list:
            b_idx, proba = bootstrap_worker(a)
            results[b_idx] = proba
    return results


def compute_proba_stats(boot_results, unique_cats):
    """Summarise bootstrap probabilities into per-plant statistics."""
    d = {}
    for ci, cat in enumerate(unique_cats):
        cp = boot_results[:, :, ci]
        d[f'{cat}_mean']     = np.mean(cp, axis=0)
        d[f'{cat}_std']      = np.std(cp, axis=0)
        d[f'{cat}_ci_lower'] = np.percentile(cp, 2.5,  axis=0)
        d[f'{cat}_ci_upper'] = np.percentile(cp, 97.5, axis=0)
        d[f'{cat}_median']   = np.median(cp, axis=0)

    means = np.column_stack([d[f'{c}_mean'] for c in unique_cats])
    pred_idx = np.argmax(means, axis=1)
    d['predicted_biotreat']  = [unique_cats[i] for i in pred_idx]
    d['confidence']          = np.max(means, axis=1)
    eps = 1e-10
    safe = np.clip(means, eps, 1-eps)
    d['prediction_entropy']  = -np.sum(safe * np.log(safe), axis=1)
    return pd.DataFrame(d)


def stage1_cv_proba(df, available_features):
    """Stage 1: stratified 10-fold CV producing out-of-fold class probabilities."""
    print("\n" + "="*70)
    print("Stage 1: 10-fold CV class probabilities")
    print("="*70)

    completed_folds = load_progress()

    train_df = df.dropna(subset=[LABEL_COL]).copy()
    train_df['population_bucket'] = train_df[POP_COL].apply(get_population_bucket)
    train_df['serial_number'] = train_df.index

    print(f"  labelled samples: {len(train_df)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_all = train_df[LABEL_COL].values

    fold_files = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_all)):
        fold_num = fold_idx + 1
        fold_file = os.path.join(STAGE1_DIR, f'validation_fold_{fold_num:02d}_proba.csv')
        fold_files.append(fold_file)

        if fold_num in completed_folds:
            print(f"\n  Fold {fold_num}/{N_FOLDS}: already done, skipped")
            continue

        print(f"\n  {'─'*60}")
        print(f"  Fold {fold_num}/{N_FOLDS}  start {datetime.now().strftime('%H:%M:%S')}")
        print(f"  {'─'*60}")
        fold_start = time.time()

        fold_train = train_df.iloc[train_idx].copy()
        fold_val   = train_df.iloc[val_idx].copy()
        print(f"    train {len(fold_train)}  validate {len(fold_val)}")

        fold_rows = []

        for bucket in sorted(fold_train['population_bucket'].unique()):
            tr_b  = fold_train[fold_train['population_bucket'] == bucket]
            val_b = fold_val[fold_val['population_bucket'] == bucket]

            if len(tr_b) < 10 or len(val_b) == 0:
                print(f"    [{bucket}] too few samples, skipped")
                continue

            print(f"    [{bucket}] train={len(tr_b)}, validate={len(val_b)}")

            X_train = tr_b[available_features].copy()
            y_train = tr_b[LABEL_COL].values
            X_val   = val_b[available_features].copy()

            for col in X_train.select_dtypes(include=np.number).columns:
                med = X_train[col].median()
                X_train[col] = X_train[col].fillna(med)
                X_val[col]   = X_val[col].fillna(med)

            for col in X_train.select_dtypes(include=['object','category']).columns:
                all_cats = pd.Categorical(
                    pd.concat([X_train[col], X_val[col]], ignore_index=True)
                ).categories
                X_train[col] = pd.Categorical(X_train[col], categories=all_cats).codes
                X_val[col]   = pd.Categorical(X_val[col], categories=all_cats).codes

            unique_cats = sorted(set(y_train))
            cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}

            val_mask = np.array([t in cat_to_idx for t in val_b[LABEL_COL].values])
            if not np.all(val_mask):
                print(f"    [{bucket}] dropped {np.sum(~val_mask)} validation rows with a class absent from training")
                val_b = val_b.iloc[val_mask]
                X_val = X_val.iloc[val_mask]
            if len(val_b) == 0:
                continue

            y_enc = np.array([cat_to_idx[c] for c in y_train])

            treatment_levels = (val_b['Treatment Level'].tolist()
                                if 'Treatment Level' in val_b.columns else None)

            print(f"    [{bucket}] bootstrap ({B} replicates)...")
            boot_results = run_bootstrap(
                X_train, y_enc, X_val, unique_cats,
                get_bucket_params(bucket), treatment_levels
            )

            stats_df = compute_proba_stats(boot_results, unique_cats)
            stats_df.index = val_b.index

            stats_df.insert(0, 'serial_number',    val_b['serial_number'].values)
            stats_df.insert(1, 'fold',             fold_num)
            stats_df.insert(2, 'true_biotreat',    val_b[LABEL_COL].values)
            stats_df.insert(3, 'population_bucket', bucket)
            stats_df.insert(4, 'served_population', val_b[POP_COL].values)
            if 'Treatment Level' in val_b.columns:
                stats_df['treatment_level'] = val_b['Treatment Level'].values

            fold_rows.append(stats_df)

        if fold_rows:
            fold_df = pd.concat(fold_rows, ignore_index=True)
            fold_df.to_csv(fold_file, index=False, encoding='utf-8')
            elapsed = (time.time() - fold_start) / 60
            print(f"\n  Fold {fold_num} done: {len(fold_df)} rows -> {fold_file}  ({elapsed:.1f} min)")

            completed_folds.add(fold_num)
            save_progress(completed_folds)
        else:
            print(f"  ! Fold {fold_num} produced no result")

    existing = [f for f in fold_files if os.path.exists(f)]
    print(f"\n  stage 1 done: {len(existing)}/{N_FOLDS} fold files ready")
    return existing


def sample_beta_prob(mean_val, std_val):
    """Draw a class probability from a Beta matched to mean and std."""
    if pd.isna(mean_val) or mean_val <= 0:
        return 0.0
    if pd.isna(std_val) or std_val <= 0:
        return float(mean_val)
    mean_val = np.clip(float(mean_val), 0.001, 0.999)
    var = min(float(std_val)**2, mean_val*(1-mean_val)*0.99)
    if var <= 0:
        return mean_val
    try:
        cf = mean_val*(1-mean_val)/var - 1
        if cf <= 0:
            return mean_val
        a, b = mean_val*cf, (1-mean_val)*cf
        return float(np.random.beta(a, b)) if (a>0 and b>0) else mean_val
    except Exception:
        return float(np.clip(np.random.normal(mean_val, std_val), 0, 1))


def mc_single_wwtp(row, unique_cats, n2o_raw, ch4_raw, fb_by_type):
    """Propagate classification and EF uncertainty for one plant."""
    n2o_s = np.zeros(N_SIMULATIONS, dtype=np.float32)
    ch4_s = np.zeros(N_SIMULATIONS, dtype=np.float32)

    for sim in range(N_SIMULATIONS):
        probs = {}
        for cat in unique_cats:
            m = row.get(f'{cat}_mean', np.nan)
            s = row.get(f'{cat}_std',  np.nan)
            p = sample_beta_prob(m, s)
            if p > 0:
                probs[cat] = p
        total = sum(probs.values())
        if total <= 0:
            continue
        probs = {k: v/total for k, v in probs.items()}

        n2o_ef = ch4_ef = 0.0
        for cat, prob in probs.items():
            nv = sample_n2o_ef(cat, n2o_raw, fb_by_type)
            cv = sample_ch4_ef(cat, ch4_raw)
            if nv is not None:
                n2o_ef += prob * nv
            if cv is not None:
                ch4_ef += prob * cv
        n2o_s[sim] = n2o_ef
        ch4_s[sim] = ch4_ef

    return n2o_s, ch4_s


def process_mc_chunk(args):
    """Run the plant-level Monte Carlo for a chunk of plants."""
    chunk, unique_cats, n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean = args

    np.random.seed(int(time.time()*1000) % (2**31) + os.getpid())
    random.seed(int(time.time()*1000)  % (2**31) + os.getpid())

    rows = []
    for _, row in chunk.iterrows():
        n2o_s, ch4_s = mc_single_wwtp(row, unique_cats, n2o_raw, ch4_raw, fb_by_type)

        true_type = row.get('true_biotreat', '')
        n2o_true  = n2o_type_mean.get(true_type, np.nan)
        ch4_true  = ch4_type_mean.get(true_type, np.nan)

        n2o_pred  = float(np.mean(n2o_s))
        ch4_pred  = float(np.mean(ch4_s))

        n2o_p25, n2o_p975 = np.percentile(n2o_s, 2.5), np.percentile(n2o_s, 97.5)
        ch4_p25, ch4_p975 = np.percentile(ch4_s, 2.5), np.percentile(ch4_s, 97.5)

        rows.append({
            'serial_number':      row.get('serial_number', row.name),
            'fold':               row.get('fold', -1),
            'true_biotreat':      true_type,
            'predicted_biotreat': row.get('predicted_biotreat', ''),
            'confidence':         row.get('confidence', np.nan),
            'served_population':  row.get('served_population', np.nan),
            'population_bucket':  row.get('population_bucket', ''),
            'N2O_true_EF':          n2o_true,
            'N2O_pred_EF_mean':     n2o_pred,
            'N2O_pred_EF_p2.5':     float(n2o_p25),
            'N2O_pred_EF_p97.5':    float(n2o_p975),
            'CH4_true_EF':          ch4_true,
            'CH4_pred_EF_mean':     ch4_pred,
            'CH4_pred_EF_p2.5':     float(ch4_p25),
            'CH4_pred_EF_p97.5':    float(ch4_p975),
        })
    return rows


import re

N_SIM_REGION = 10000
REGION_SEED = 42
REGION_POOL_SIZE = 20000
INCLUDE_PROB_UNCERTAINTY = True
STRATA_MODE = 'pool_only'

REGION_POP_COL = 'served_population'
REGION_BUCKET_COL = 'population_bucket'
REGION_TYPE_COL = 'true_biotreat'


def _records_by_type(records, value_key, keep_fb=False):
    """Group cleaned EF records by treatment type."""
    out = {}
    for r in records:
        t = r.get('Type')
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        v = r.get(value_key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out.setdefault(t, []).append(
            (float(v), str(r.get('F/B', ''))) if keep_fb else float(v))
    return out


def sample_n2o_type(t, n2o_by_type, fb_by_type, gmr, rng):
    """Draw one N2O EF for a treatment type using a shared generator."""
    if t in N2O_DEFAULT_VALUES:
        return float(N2O_DEFAULT_VALUES[t])
    recs = n2o_by_type.get(t)
    if not recs:
        return np.nan
    ef, fb = recs[rng.randrange(len(recs))]
    if fb != 'F':
        ratios = fb_by_type.get(t)
        r = float(ratios[rng.randrange(len(ratios))]) if ratios else gmr
        if r > 0:
            ef = ef / r
    return float(ef)


def sample_ch4_type(t, ch4_by_type, rng):
    """Draw one CH4 EF for a treatment type using a shared generator."""
    if t in CH4_DEFAULT_VALUES:
        d = CH4_DEFAULT_VALUES[t]
        return float(abs(np.random.normal(d, d * CH4_DEFAULT_UNCERTAINTY)))
    recs = ch4_by_type.get(t)
    if not recs:
        return np.nan
    return float(recs[rng.randrange(len(recs))])


def build_ef_pools(cats, gas, ef_by_type, fb_by_type, gmr, seed,
                   pool_size=REGION_POOL_SIZE):
    """Pre-resolve one sampling pool of emission factors per treatment type."""
    rng = random.Random(seed)
    pools = {}
    for c in cats:
        if gas == 'N2O':
            vals = [sample_n2o_type(c, ef_by_type, fb_by_type, gmr, rng)
                    for _ in range(pool_size)]
        else:
            vals = [sample_ch4_type(c, ef_by_type, rng) for _ in range(pool_size)]
        vals = np.array(vals, dtype=float)
        pools[c] = vals if np.isfinite(vals).any() else None
    return pools


def region_beta_params(mean_mat, std_mat):
    """Convert class probability mean and std into Beta parameters."""
    m = np.clip(mean_mat, 0.001, 0.999)
    var = np.minimum(std_mat ** 2, m * (1 - m) * 0.99)
    with np.errstate(divide='ignore', invalid='ignore'):
        cf = m * (1 - m) / var - 1
    ok = (var > 0) & (cf > 0) & np.isfinite(cf)
    a = np.where(ok, m * cf, 1.0)
    b = np.where(ok, (1 - m) * cf, 1.0)
    return a, b, ok, m


def region_mc(sub, cats, ef_pools, n_sim, seed, use_prob_unc=True):
    """Draw one EF per plant and aggregate to a single population-weighted value."""
    rng_np = np.random.default_rng(seed)

    pop = pd.to_numeric(sub[REGION_POP_COL], errors='coerce').fillna(1.0).values
    w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop)) / len(pop)
    n = len(sub)

    mean_mat = np.nan_to_num(sub[[f'{c}_mean' for c in cats]].to_numpy(dtype=float))
    std_mat = np.nan_to_num(sub[[f'{c}_std' for c in cats]].to_numpy(dtype=float))

    pool_list = [ef_pools.get(c) for c in cats]
    C = len(cats)
    out = np.empty(n_sim, dtype=float)

    P_fixed = None
    if not use_prob_unc:
        p = np.where(mean_mat > 0, mean_mat, 0.0)
        tot = p.sum(axis=1, keepdims=True)
        P_fixed = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

    for k in range(n_sim):
        if P_fixed is not None:
            P = P_fixed
        else:
            a, b, ok, m = region_beta_params(mean_mat, std_mat)
            draw = rng_np.beta(a, b)
            p = np.where(ok, draw, m)
            p = np.where(mean_mat > 0, p, 0.0)
            tot = p.sum(axis=1, keepdims=True)
            P = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

        cdf = np.cumsum(P, axis=1)
        u = rng_np.random(size=(n, 1))
        cat_idx = np.clip((u > cdf).sum(axis=1), 0, C - 1)

        ef_i = np.full(n, np.nan)
        for ci in range(C):
            mask = cat_idx == ci
            n_need = int(mask.sum())
            if n_need == 0:
                continue
            pool = pool_list[ci]
            if pool is None:
                continue
            ef_i[mask] = pool[rng_np.integers(0, len(pool), size=n_need)]

        ok_i = ~np.isnan(ef_i)
        if not ok_i.any():
            out[k] = np.nan
            continue
        ww = w[ok_i] / w[ok_i].sum()
        out[k] = float(np.sum(ww * ef_i[ok_i]))

    return out


def run_region_stage(proba_df, unique_cats, n2o_raw, ch4_raw, fb_by_type,
                     n2o_type_mean, ch4_type_mean):
    """Treat each fold as one region and return the region-level tables."""
    print("\n" + "=" * 70)
    print("Stage 3: region-level EF prediction intervals, one region per fold")
    print("=" * 70)

    np.random.seed(REGION_SEED)
    random.seed(REGION_SEED)

    n2o_by_type = _records_by_type(n2o_raw, N2O_EF_COL, keep_fb=True)
    ch4_by_type = _records_by_type(ch4_raw, CH4_EF_COL, keep_fb=False)
    all_r = [float(v) for vals in fb_by_type.values() for v in vals]
    gmr = float(np.mean(all_r)) if all_r else 1.0

    if STRATA_MODE == 'all' and REGION_BUCKET_COL in proba_df.columns \
            and proba_df[REGION_BUCKET_COL].notna().any():
        found = [str(x) for x in proba_df[REGION_BUCKET_COL].dropna().unique()]

        def lower_edge(lab):
            """Lower edge of the population bucket."""
            nums = re.findall(r'\d+', lab)
            return int(nums[0]) if nums else 10 ** 12

        strata = sorted(found, key=lower_edge) + ['Pool']
    else:
        strata = ['Pool']
    print(f"  strata mode: {STRATA_MODE} -> {strata}")

    GASES = [('N2O', n2o_by_type, n2o_type_mean),
             ('CH4', ch4_by_type, ch4_type_mean)]

    rows = []

    for gas, ef_by_type, true_map in GASES:
        print(f"\n  building the {gas} EF pool...")
        ef_pools = build_ef_pools(unique_cats, gas, ef_by_type,
                                  fb_by_type, gmr, REGION_SEED)

        for fold_id in sorted(proba_df['fold'].unique()):
            fold_all = proba_df[proba_df['fold'] == fold_id]

            for si, stratum in enumerate(strata):
                sub = fold_all if stratum == 'Pool' else \
                    fold_all[fold_all[REGION_BUCKET_COL] == stratum]
                if len(sub) == 0:
                    continue

                pop = pd.to_numeric(sub[REGION_POP_COL], errors='coerce').fillna(1.0).values
                w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop)) / len(pop)

                true_vals = sub[REGION_TYPE_COL].map(true_map).astype(float).values
                ok = ~np.isnan(true_vals)
                if ok.sum() == 0:
                    continue
                w_ok = w[ok] / w[ok].sum()
                true_region = float(np.sum(w_ok * true_vals[ok]))

                seed = REGION_SEED + int(fold_id) * 100 + si

                samples = region_mc(sub, unique_cats, ef_pools, N_SIM_REGION, seed,
                                    INCLUDE_PROB_UNCERTAINTY)
                s = samples[~np.isnan(samples)]
                if len(s) == 0:
                    continue

                p25, p975 = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
                mean_v = float(np.mean(s))

                rows.append({
                    'Gas': gas, 'Fold': int(fold_id), 'Stratum': stratum,
                    'N_plants': len(sub), 'Pop_total': float(pop.sum()),
                    'Pop_share_%': round(pop.sum() /
                                         pd.to_numeric(fold_all[REGION_POP_COL], errors='coerce')
                                         .fillna(1.0).sum() * 100, 2),
                    'True_EF_region': round(true_region, 6),
                    'Pred_EF_region_mean': round(mean_v, 6),
                    'Pred_EF_region_median': round(float(np.median(s)), 6),
                    'Pred_EF_region_std': round(float(np.std(s)), 6),
                    'Pred_EF_region_p2.5': round(p25, 6),
                    'Pred_EF_region_p97.5': round(p975, 6),
                    'PI_width': round(p975 - p25, 6),
                    'PI_width_rel_%': round((p975 - p25) / mean_v * 100, 1) if mean_v else np.nan,
                    'RelDiff_%': round((mean_v - true_region) / true_region * 100, 2)
                                 if true_region else np.nan,
                    'In_PI': int(p25 <= true_region <= p975),
                })

            print(f"    {gas} Fold {fold_id} done")

    region_df = pd.DataFrame(rows)
    if len(region_df):
        region_df['Stratum'] = pd.Categorical(region_df['Stratum'],
                                              categories=strata, ordered=True)
        region_df = region_df.sort_values(['Gas', 'Fold', 'Stratum']).reset_index(drop=True)

    return region_df


def stage2_monte_carlo(fold_files, n2o_raw, ch4_raw, fb_by_type,
                       n2o_type_mean, ch4_type_mean):
    """Stage 2: Monte Carlo EF estimation and validation against reference values."""
    print("\n" + "="*70)
    print("Stage 2: Monte Carlo EF estimation and validation")
    print("="*70)

    plant_csv   = os.path.join(STAGE2_DIR, 'validation_plant_level_results.csv')
    summary_xls = os.path.join(STAGE2_DIR, 'validation_summary.xlsx')

    all_proba = []
    for f in fold_files:
        if os.path.exists(f):
            all_proba.append(pd.read_csv(f))
    if not all_proba:
        print("  no stage 1 probability files found")
        return

    proba_df = pd.concat(all_proba, ignore_index=True)
    print(f"  probabilities read: {len(proba_df)} rows from {len(fold_files)} folds")

    unique_cats = sorted([
        col.replace('_mean', '') for col in proba_df.columns
        if col.endswith('_mean') and col.replace('_mean', '') in TREATMENT_METHODS
    ])
    print(f"  treatment types: {unique_cats}")

    chunk_size = max(1, len(proba_df) // N_CORES_MC)
    chunks = [proba_df.iloc[i:i+chunk_size]
              for i in range(0, len(proba_df), chunk_size)]

    args_list = [
        (chunk, unique_cats, n2o_raw, ch4_raw, fb_by_type,
         n2o_type_mean, ch4_type_mean)
        for chunk in chunks
    ]

    print(f"\n  starting the Monte Carlo on {N_CORES_MC} cores in {len(chunks)} chunks...")
    mc_start = time.time()

    all_plant_rows = []
    with Pool(N_CORES_MC) as pool:
        for i, batch in enumerate(pool.imap(process_mc_chunk, args_list)):
            all_plant_rows.extend(batch)
            print(f"  chunk {i+1}/{len(chunks)} done, {len(all_plant_rows)} rows so far")
            gc.collect()

    print(f"  Monte Carlo took {(time.time()-mc_start)/60:.1f} min")

    plant_df = pd.DataFrame(all_plant_rows)
    plant_df.to_csv(plant_csv, index=False, encoding='utf-8')
    print(f"\n  plant-level results saved: {plant_csv}  ({len(plant_df)} rows)")

    fold_rows = []
    for gas, true_col, pred_col in [
        ('N2O', 'N2O_true_EF', 'N2O_pred_EF_mean'),
        ('CH4', 'CH4_true_EF', 'CH4_pred_EF_mean'),
    ]:
        for fold_id in sorted(plant_df['fold'].unique()):
            fd = plant_df[plant_df['fold'] == fold_id].dropna(subset=[true_col, pred_col])
            if len(fd) == 0:
                continue

            pop = fd['served_population'].fillna(1.0).values
            true_arr = fd[true_col].values
            pred_arr = fd[pred_col].values

            true_mean_simple = float(np.mean(true_arr))
            pred_mean_simple = float(np.mean(pred_arr))
            rel_err_simple   = (pred_mean_simple - true_mean_simple) / true_mean_simple * 100 \
                               if true_mean_simple != 0 else np.nan

            w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop))/len(pop)
            true_mean_wtd = float(np.sum(w * true_arr))
            pred_mean_wtd = float(np.sum(w * pred_arr))
            rel_err_wtd   = (pred_mean_wtd - true_mean_wtd) / true_mean_wtd * 100 \
                            if true_mean_wtd != 0 else np.nan

            residuals = pred_arr - true_arr
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae  = float(np.mean(np.abs(residuals)))

            fold_rows.append({
                'Gas':                     gas,
                'Fold':                    int(fold_id),
                'N_samples':               len(fd),
                'True_EF_simple_mean':     round(true_mean_simple, 6),
                'Pred_EF_simple_mean':     round(pred_mean_simple, 6),
                'Relative_Error_simple_%': round(rel_err_simple, 2),
                'True_EF_wtd_mean':        round(true_mean_wtd, 6),
                'Pred_EF_wtd_mean':        round(pred_mean_wtd, 6),
                'Relative_Error_wtd_%':    round(rel_err_wtd, 2),
                'RMSE':                    round(rmse, 6),
                'MAE':                     round(mae, 6),
            })

    fold_df = pd.DataFrame(fold_rows)

    summary_rows = []
    for gas in ['N2O', 'CH4']:
        gd = fold_df[fold_df['Gas'] == gas]
        if len(gd) == 0:
            continue
        for metric, label in [
            ('Relative_Error_simple_%', 'RelError_simple_%'),
            ('Relative_Error_wtd_%',    'RelError_wtd_%'),
            ('RMSE',                    'RMSE'),
            ('MAE',                     'MAE'),
        ]:
            summary_rows.append({
                'Gas':    gas,
                'Metric': label,
                'Mean':   round(gd[metric].mean(), 6),
                'Min':    round(gd[metric].min(),  6),
                'Max':    round(gd[metric].max(),  6),
                'Std':    round(gd[metric].std(),  6),
            })

    summary_df = pd.DataFrame(summary_rows)

    region_df = run_region_stage(
        proba_df, unique_cats, n2o_raw, ch4_raw, fb_by_type,
        n2o_type_mean, ch4_type_mean
    )

    with pd.ExcelWriter(summary_xls, engine='openpyxl') as writer:
        fold_df.to_excel(          writer, sheet_name='Fold_Results',        index=False)
        summary_df.to_excel(       writer, sheet_name='Summary',             index=False)
        plant_df.to_excel(         writer, sheet_name='Plant_Details',       index=False)
        if len(region_df):
            region_df.to_excel(    writer, sheet_name='Region_Fold_Results', index=False)

    print(f"  validation report saved: {summary_xls}")

    print("\n" + "=" * 70)
    print("validation summary")
    print("=" * 70)
    print("\n-- Fold results, plant level --")
    print(fold_df.to_string(index=False))
    print("\n-- Summary across folds --")
    print(summary_df.to_string(index=False))

    if len(region_df):
        show = ['Fold', 'Stratum', 'N_plants', 'Pop_share_%', 'True_EF_region',
                'Pred_EF_region_mean', 'Pred_EF_region_p2.5', 'Pred_EF_region_p97.5',
                'PI_width_rel_%', 'RelDiff_%', 'In_PI']
        for gas in ['N2O', 'CH4']:
            gd = region_df[region_df['Gas'] == gas]
            if len(gd) == 0:
                continue
            print(f"\n-- {gas} region level, one region per fold --")
            print(gd[show].to_string(index=False))
        print(f"\nCoverage is computed over only {region_df['Fold'].nunique()} folds, "
              f"so the coverage figure itself is highly uncertain.")


def main():
    """Run stage 1 and stage 2 end to end."""
    t0 = time.time()
    print("="*70)
    print("Ten-fold cross-validation and Monte Carlo EF validation")
    print(f"start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    df = pd.read_excel(INPUT_FILE)
    print(f"\nrows read: {len(df)}")
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"features available: {len(available_features)}")
    print(f"labelled samples: {df[LABEL_COL].notna().sum()}")

    n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean = load_ef_data()

    fold_files = stage1_cv_proba(df, available_features)

    existing = [f for f in fold_files if os.path.exists(f)]
    total_rows = sum(len(pd.read_csv(f)) for f in existing)
    print(f"\nstage 1 check: {len(existing)}/{N_FOLDS} folds, {total_rows} rows")
    print("moving on to stage 2\n")

    stage2_monte_carlo(
        existing, n2o_raw, ch4_raw, fb_by_type,
        n2o_type_mean, ch4_type_mean
    )

    h, rem = divmod(int(time.time()-t0), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nall done in {h}h{m}m{s}s")
    print(f"  stage 1: {STAGE1_DIR}/validation_fold_XX_proba.csv")
    print(f"  stage 2: {STAGE2_DIR}/validation_plant_level_results.csv")
    print(f"  stage 2: {STAGE2_DIR}/validation_summary.xlsx")


if __name__ == '__main__':
    import platform
    method = 'fork' if platform.system() != 'Windows' else 'spawn'
    try:
        multiprocessing.set_start_method(method, force=True)
    except RuntimeError:
        pass
    main()