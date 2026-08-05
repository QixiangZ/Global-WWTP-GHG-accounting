"""Hyperparameter grid search for Random Forest."""

import os
import gc
import json
import time
import zlib
import random
import shutil
import hashlib
import warnings
import platform
import itertools
import multiprocessing
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from paths import OUTPUT_DIR, inp, out_dir

warnings.filterwarnings('ignore')

FAMILY = 'RF'

SHARED_CONFIG = {
    'B':                    1000,
    'N_SIMULATIONS':        10000,
    'N_FOLDS':              10,
    'RANDOM_SEED':          42,
    'IQR_K':                1.5,
    'MIN_GROUP_N':          4,
    'FB_NORM_FLOOR':        0.10,
    'FB_SAMPLING_FLOOR':    None,
    'SUBSAMPLE':            0.8,
    'COLSAMPLE':            0.8,
    'REG_ALPHA':            1.0,
    'LGBM_FIXED_LAMBDA':    3.0,
    'BUCKET_EDGES':         [2000, 10000],
    'CH4_DEFAULT_VALUES':   {'Pond': 0.102, 'Lagoon': 0.102, 'Anaerobic': 0.408},
    'CH4_DEFAULT_UNCERTAINTY': 0.30,
    'N2O_DEFAULT_VALUES':   {'Anaerobic': 0},
    'ADVANCED_TYPES':       ['A2O', 'AO', 'BAF', 'BNR', 'EA', 'MBR', 'MLE', 'OD', 'SBR'],
    'TREATMENT_METHODS':    ['A2O', 'AO', 'Anaerobic', 'BAF', 'BNR', 'CAS', 'EA',
                             'Lagoon', 'MBR', 'MLE', 'OD', 'Pond', 'RBC', 'SBR',
                             'TF', 'Wetland'],
    'FEATURE_COLS': [
        'Current Served Population (estimate)', 'Population_Density', 'Urban_cluster_share',
        'GDP_PerCapita_PPP', 'GNI_per_capita', 'Extreme Monthly Average Temperature (High)',
        'Decadal Average Precipitation', 'Agriculture_Area', 'Nitrogen_fertilizer',
        'Pasture_Area', 'Building_Area', 'Water_Area', 'Aridity_Index', 'HDI',
        'Social_progress_index', 'Developing_country', 'City_Population', 'Region_Area',
        'Rural_share', 'Income_level', 'Small_island_developing_state',
        'Extreme Monthly Average Temperature (Low)', 'Decadal Average Temperature',
        'Extreme Monthly Average Precipitation (High)',
        'Extreme Monthly Average Precipitation (Low)', 'Average Atmospheric Pressure'
    ],
}

CONFIG_FINGERPRINT = hashlib.md5(
    json.dumps(SHARED_CONFIG, sort_keys=True, default=str).encode()
).hexdigest()[:12]

B                    = SHARED_CONFIG['B']
N_SIMULATIONS        = SHARED_CONFIG['N_SIMULATIONS']
N_FOLDS              = SHARED_CONFIG['N_FOLDS']
RANDOM_SEED          = SHARED_CONFIG['RANDOM_SEED']
IQR_K                = SHARED_CONFIG['IQR_K']
MIN_GROUP_N          = SHARED_CONFIG['MIN_GROUP_N']
FB_NORM_FLOOR        = SHARED_CONFIG['FB_NORM_FLOOR']
FB_SAMPLING_FLOOR    = SHARED_CONFIG['FB_SAMPLING_FLOOR']
SUBSAMPLE            = SHARED_CONFIG['SUBSAMPLE']
COLSAMPLE            = SHARED_CONFIG['COLSAMPLE']
REG_ALPHA            = SHARED_CONFIG['REG_ALPHA']
LGBM_FIXED_LAMBDA    = SHARED_CONFIG['LGBM_FIXED_LAMBDA']
SMALL_EDGE, MED_EDGE = SHARED_CONFIG['BUCKET_EDGES']

FEATURE_COLS            = SHARED_CONFIG['FEATURE_COLS']
TREATMENT_METHODS       = set(SHARED_CONFIG['TREATMENT_METHODS'])
ADVANCED_TYPES          = set(SHARED_CONFIG['ADVANCED_TYPES'])
CH4_DEFAULT_VALUES      = SHARED_CONFIG['CH4_DEFAULT_VALUES']
CH4_DEFAULT_UNCERTAINTY = SHARED_CONFIG['CH4_DEFAULT_UNCERTAINTY']
N2O_DEFAULT_VALUES      = SHARED_CONFIG['N2O_DEFAULT_VALUES']

# --- input ---------------------------------------------------------------
INPUT_FILE  = inp('combined_wwtp.xlsx')
N2O_EF_FILE = inp('N2O_EF.xlsx')
CH4_EF_FILE = inp('CH4_EF.xlsx')
FB_FILE     = inp('FB.xlsx')

N2O_EF_COL = 'N2OEF'
CH4_EF_COL = 'CH4EF'
FB_COL     = 'FB'

# --- output --------------------------------------------------------------
# every result file carries the family name so that the three searches can share
# a single flat output/ directory without overwriting each other
RESULT_DIR = OUTPUT_DIR
TEMP_DIR   = out_dir('_temp_RF')

AVAILABLE_CORES = int(os.environ.get('NCPUS', 0)) or cpu_count()
if AVAILABLE_CORES >= 16:
    N_JOBS_BOOTSTRAP, N_JOBS_MODEL = 8, 2
elif AVAILABLE_CORES >= 8:
    N_JOBS_BOOTSTRAP, N_JOBS_MODEL = 4, 2
else:
    N_JOBS_BOOTSTRAP, N_JOBS_MODEL = 2, 1
N_CORES_MC = min(16, AVAILABLE_CORES)

LABEL_COL = 'Biotreat Type'
POP_COL   = 'Current Served Population (estimate)'
BUCKETS   = ['small: 0-2000', 'medium: 2000-10000', 'large: 10000+']
N_KEEP    = 1

PARAM_KEYS = ['max_depth', 'n_estimators', 'min_samples_leaf', 'max_features']

PARAM_GRID = {
    'max_depth':        [8, 12, 20, None],
    'n_estimators':     [100, 200, 300],
    'min_samples_leaf': [1, 3, 5],
    'max_features':     [0.5, 'sqrt', 'log2'],
}

BUCKET_SLUG = {
    'small: 0-2000':      'small',
    'medium: 2000-10000': 'medium',
    'large: 10000+':      'large',
}


def expand_grid():
    """Full parameter grid, last key varying fastest."""
    values = [PARAM_GRID[k] for k in PARAM_KEYS]
    return [dict(zip(PARAM_KEYS, v)) for v in itertools.product(*values)]


def param_key(params):
    """Stable identifier for one parameter set."""
    return '|'.join(str(params[k]) for k in PARAM_KEYS)


def done_path(bucket):
    """Path of the resume log for one bucket."""
    return os.path.join(RESULT_DIR, f'done_{FAMILY}_{BUCKET_SLUG[bucket]}.txt')


def top_path(bucket):
    """Path of the retained fold-level table for one bucket."""
    return os.path.join(RESULT_DIR, f'best_{FAMILY}_{BUCKET_SLUG[bucket]}.csv')


def get_population_bucket(pop):
    """Assign a plant to a served-population bucket."""
    if pd.isna(pop):
        return 'medium: 2000-10000'
    if pop <= SMALL_EDGE:
        return 'small: 0-2000'
    if pop <= MED_EDGE:
        return 'medium: 2000-10000'
    return 'large: 10000+'


def _clip_floor(vals, floor=FB_NORM_FLOOR):
    """Drop NaN and clip to a lower bound."""
    a = np.asarray(list(vals), dtype=float)
    a = a[~np.isnan(a)]
    if floor is not None:
        a = np.clip(a, floor, None)
    return a


def iqr_clean_by_type(df, screen_col, group_col='Type', k=IQR_K,
                      min_n=MIN_GROUP_N, label='', verbose=True):
    """Remove NaN and IQR outliers within each Type group."""
    n_input = len(df)

    n_group_nan = int(df[group_col].isna().sum())
    if n_group_nan and verbose:
        print(f"  ! {group_col}: {n_group_nan} missing, dropped")
    df = df.dropna(subset=[group_col])

    nan_mask = df[screen_col].isna()
    n_nan = int(nan_mask.sum())
    if n_nan and verbose:
        print(f"  ! {screen_col}: {n_nan} NaN, dropped")
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

        rec.update({'Method': 'IQR', 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                    'Lower': lo, 'Upper': hi,
                    'N_removed': int(mask.sum()),
                    'Pct_removed': mask.sum() / n * 100,
                    'N_kept': len(kept),
                    'Mean_after': kept.mean() if len(kept) else np.nan})
        rows.append(rec)

    cleaned = d.drop(bad)
    summary = pd.DataFrame(rows)

    if verbose:
        n_out = n_input - len(cleaned)
        print(f"  IQR {label or screen_col}: {n_input} -> {len(cleaned)} "
              f"(NaN {n_nan + n_group_nan} + outliers {len(bad)}, "
              f"{n_out / max(n_input, 1):.2%})")
        if len(summary):
            empty = summary[summary['N_kept'] == 0]
            if len(empty):
                print(f"  ! emptied Type: {sorted(empty['Type'].tolist())}")
            high = summary[summary['Pct_removed'] > 50]
            if len(high):
                print(f"  ! >50% removed: {sorted(high['Type'].tolist())}")

    return cleaned, summary


_EF_CACHE = {}


def load_cleaned_ef_frames(verbose=True):
    """Load and IQR-clean the N2O and CH4 emission factor tables."""
    if 'frames' in _EF_CACHE:
        return _EF_CACHE['frames']

    fb_df = pd.read_excel(FB_FILE)
    fb_by_type = {}
    for _, r in fb_df.iterrows():
        fb_by_type.setdefault(r['Type'], []).append(float(r[FB_COL]))

    fb_mean_by_type = {
        t: float(_clip_floor(v).mean()) if len(_clip_floor(v)) else np.nan
        for t, v in fb_by_type.items()
    }
    all_flat = [v for vv in fb_by_type.values() for v in vv]
    gmr = float(_clip_floor(all_flat).mean()) if all_flat else 1.0

    n_low = int((np.asarray(all_flat, dtype=float) < (FB_NORM_FLOOR or 0)).sum())
    if n_low and verbose:
        print(f'  ! {n_low} FB values below {FB_NORM_FLOOR}; '
              f'sampling floor is {FB_SAMPLING_FLOOR}')

    if FB_SAMPLING_FLOOR is not None:
        fb_by_type = {t: [max(float(v), FB_SAMPLING_FLOOR) for v in vv]
                      for t, vv in fb_by_type.items()}

    n2o_df = pd.read_excel(N2O_EF_FILE)
    if 'F/B' not in n2o_df.columns:
        if verbose:
            print("  ! no 'F/B' column, all records treated as non-F")
        n2o_df['F/B'] = np.nan

    is_F = n2o_df['F/B'].astype(str).str.strip().eq('F')
    r = n2o_df['Type'].map(lambda t: fb_mean_by_type.get(t, gmr))
    r = r.replace(0, np.nan).fillna(gmr)
    n2o_df['_screen'] = np.where(is_F, n2o_df[N2O_EF_COL], n2o_df[N2O_EF_COL] / r)

    n2o_df, _ = iqr_clean_by_type(n2o_df, '_screen',
                                  label='N2OEF (normalised scale)', verbose=verbose)
    n2o_df = n2o_df.drop(columns=['_screen'])

    ch4_df = pd.read_excel(CH4_EF_FILE)
    ch4_df, _ = iqr_clean_by_type(ch4_df, CH4_EF_COL, label='CH4EF', verbose=verbose)

    if verbose:
        print(f'  N2O EF {len(n2o_df)} | CH4 EF {len(ch4_df)} | '
              f'FB {len(all_flat)}, global mean {gmr:.4f}')

    _EF_CACHE['frames'] = (n2o_df, ch4_df, fb_by_type, fb_mean_by_type, gmr)
    return _EF_CACHE['frames']


def n2o_true_ef(ef, fb, t, fb_by_type, gmr):
    """Rescale one N2O record to the F scale, matching the sampling side."""
    ef = float(ef)
    if str(fb).strip() == 'F':
        return ef
    rs = _clip_floor([x for x in fb_by_type.get(t, []) if x > 0])
    r = float(rs.mean()) if len(rs) else gmr
    return ef / r if r > 0 else ef


def load_ef_data():
    """Return EF records and per-Type mean EFs used as ground truth."""
    n2o_df, ch4_df, fb_by_type, fb_mean_by_type, gmr = load_cleaned_ef_frames()

    n2o_raw = n2o_df.to_dict('records')
    ch4_raw = ch4_df.to_dict('records')

    n2o_type_mean = {}
    for rec in n2o_raw:
        t = rec['Type']
        n2o_type_mean.setdefault(t, []).append(
            n2o_true_ef(rec[N2O_EF_COL], rec.get('F/B', ''), t, fb_by_type, gmr))
    n2o_type_mean = {t: float(np.mean(v)) for t, v in n2o_type_mean.items()}
    for t, v in N2O_DEFAULT_VALUES.items():
        n2o_type_mean[t] = float(v)

    ch4_type_mean = {}
    for rec in ch4_raw:
        ch4_type_mean.setdefault(rec['Type'], []).append(rec[CH4_EF_COL])
    ch4_type_mean = {t: float(np.mean(v)) for t, v in ch4_type_mean.items()}
    for t, v in CH4_DEFAULT_VALUES.items():
        ch4_type_mean[t] = float(v)

    return n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean


def sample_n2o_ef(t, n2o_raw, fb_by_type):
    """Draw one N2O EF for a treatment type."""
    if t in N2O_DEFAULT_VALUES:
        return N2O_DEFAULT_VALUES[t]
    records = [r for r in n2o_raw if r['Type'] == t]
    if not records:
        return None
    rec = random.choice(records)
    ef = rec[N2O_EF_COL]
    if str(rec.get('F/B', '')).strip() != 'F':
        ratios = fb_by_type.get(t)
        if ratios:
            ef = ef / random.choice(ratios)
        else:
            all_r = [v for vals in fb_by_type.values() for v in vals]
            ef = ef / (np.mean(all_r) if all_r else 1.0)
    return ef


def sample_ch4_ef(t, ch4_raw):
    """Draw one CH4 EF for a treatment type."""
    if t in CH4_DEFAULT_VALUES:
        d = CH4_DEFAULT_VALUES[t]
        std = d * CH4_DEFAULT_UNCERTAINTY
        for _ in range(10):
            v = np.random.normal(d, std)
            if v >= 0:
                return v
        return abs(np.random.normal(d, std))
    records = [r for r in ch4_raw if r['Type'] == t]
    if not records:
        return None
    return random.choice(records)[CH4_EF_COL]


def build_model(params, seed):
    """Instantiate the classifier for one parameter set."""
    return RandomForestClassifier(
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        n_jobs=N_JOBS_MODEL, random_state=seed,
    )


def bootstrap_worker(args):
    """Fit one bootstrap replicate and return class probabilities."""
    (b_idx, params, X_train_rec, y_enc, X_pred_rec,
     unique_cats, treatment_levels) = args
    try:
        X_train = pd.DataFrame(X_train_rec)
        X_pred = pd.DataFrame(X_pred_rec)
        seed = b_idx + RANDOM_SEED
        rng = np.random.default_rng(seed)

        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        y_b = y_enc[idx]
        model = build_model(params, seed)
        model.fit(X_train.iloc[idx], y_b)

        proba = model.predict_proba(X_pred)

        if treatment_levels is not None:
            allowed = np.array([c in ADVANCED_TYPES for c in unique_cats])
            if np.any(allowed):
                for i, lvl in enumerate(treatment_levels):
                    if pd.notna(lvl) and str(lvl).strip() == 'Advanced':
                        proba[i, ~allowed] = 0
                        s = proba[i].sum()
                        proba[i] = (proba[i] / s if s > 0
                                    else np.where(allowed, 1.0 / np.sum(allowed), 0))
        return b_idx, proba
    except Exception as e:
        print(f"  bootstrap {b_idx} failed: {e}")
        return b_idx, np.full((len(X_pred_rec), len(unique_cats)),
                              1.0 / len(unique_cats))


def run_bootstrap(params, X_train, y_enc, X_pred, unique_cats, treatment_levels):
    """Run B bootstrap replicates in parallel, falling back to serial on failure."""
    X_tr_rec = X_train.to_dict('records')
    X_pr_rec = X_pred.to_dict('records')
    args_list = [(b, params, X_tr_rec, y_enc, X_pr_rec,
                  unique_cats, treatment_levels) for b in range(B)]
    results = np.zeros((B, len(X_pred), len(unique_cats)))
    try:
        with ProcessPoolExecutor(max_workers=N_JOBS_BOOTSTRAP) as ex:
            futures = {ex.submit(bootstrap_worker, a): a[0] for a in args_list}
            for fut in as_completed(futures):
                b_idx = futures[fut]
                try:
                    _, proba = fut.result()
                    results[b_idx] = proba
                except Exception as e:
                    print(f"    bootstrap {b_idx} result failed: {e}")
    except Exception as e:
        print(f"  parallel failed, running serially: {e}")
        for a in args_list:
            b_idx, proba = bootstrap_worker(a)
            results[b_idx] = proba
    return results


def compute_proba_stats(boot_results, unique_cats):
    """Summarise bootstrap probabilities into mean, std and 95% interval."""
    d = {}
    for ci, cat in enumerate(unique_cats):
        cp = boot_results[:, :, ci]
        d[f'{cat}_mean'] = np.mean(cp, axis=0)
        d[f'{cat}_std'] = np.std(cp, axis=0)
        d[f'{cat}_ci_lower'] = np.percentile(cp, 2.5, axis=0)
        d[f'{cat}_ci_upper'] = np.percentile(cp, 97.5, axis=0)
    means = np.column_stack([d[f'{c}_mean'] for c in unique_cats])
    pred_idx = np.argmax(means, axis=1)
    d['predicted_biotreat'] = [unique_cats[i] for i in pred_idx]
    d['confidence'] = np.max(means, axis=1)
    return pd.DataFrame(d)


def sample_beta_prob(mean_val, std_val):
    """Draw a class probability from a Beta matched to mean and std."""
    if pd.isna(mean_val) or mean_val <= 0:
        return 0.0
    if pd.isna(std_val) or std_val <= 0:
        return float(mean_val)
    mean_val = np.clip(float(mean_val), 0.001, 0.999)
    var = min(float(std_val) ** 2, mean_val * (1 - mean_val) * 0.99)
    if var <= 0:
        return mean_val
    try:
        cf = mean_val * (1 - mean_val) / var - 1
        if cf <= 0:
            return mean_val
        a, b = mean_val * cf, (1 - mean_val) * cf
        return float(np.random.beta(a, b)) if (a > 0 and b > 0) else mean_val
    except Exception:
        return float(np.clip(np.random.normal(mean_val, std_val), 0, 1))


def mc_single_wwtp(row, unique_cats, n2o_raw, ch4_raw, fb_by_type):
    """Propagate classification and EF uncertainty for one plant."""
    n2o_s = np.zeros(N_SIMULATIONS, dtype=np.float32)
    ch4_s = np.zeros(N_SIMULATIONS, dtype=np.float32)
    for sim in range(N_SIMULATIONS):
        probs = {}
        for cat in unique_cats:
            p = sample_beta_prob(row.get(f'{cat}_mean', np.nan),
                                 row.get(f'{cat}_std', np.nan))
            if p > 0:
                probs[cat] = p
        total = sum(probs.values())
        if total <= 0:
            continue
        probs = {k: v / total for k, v in probs.items()}
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
    """Run the Monte Carlo for a chunk of plants under a deterministic seed."""
    (chunk, chunk_seed, unique_cats, n2o_raw, ch4_raw,
     fb_by_type, n2o_type_mean, ch4_type_mean) = args
    np.random.seed(chunk_seed % (2 ** 31))
    random.seed(chunk_seed % (2 ** 31))
    rows = []
    for _, row in chunk.iterrows():
        n2o_s, ch4_s = mc_single_wwtp(row, unique_cats, n2o_raw, ch4_raw, fb_by_type)
        true_type = row.get('true_biotreat', '')
        rows.append({
            'true_biotreat':     true_type,
            'served_population': row.get('served_population', np.nan),
            'population_bucket': row.get('population_bucket', ''),
            'fold':              row.get('fold', -1),
            'N2O_true_EF':       n2o_type_mean.get(true_type, np.nan),
            'N2O_pred_EF_mean':  float(np.mean(n2o_s)),
            'CH4_true_EF':       ch4_type_mean.get(true_type, np.nan),
            'CH4_pred_EF_mean':  float(np.mean(ch4_s)),
        })
    return rows


def compute_fold_metrics(plant_df, bucket, params):
    """Per fold and gas: RMSE, MAE, simple and population-weighted relative error."""
    rows = []
    for gas, true_col, pred_col in [
        ('N2O', 'N2O_true_EF', 'N2O_pred_EF_mean'),
        ('CH4', 'CH4_true_EF', 'CH4_pred_EF_mean'),
    ]:
        for fold_id in sorted(plant_df['fold'].unique()):
            fd = plant_df[(plant_df['fold'] == fold_id) &
                          (plant_df['population_bucket'] == bucket)] \
                .dropna(subset=[true_col, pred_col]).copy()
            if len(fd) == 0:
                continue

            pop = fd['served_population'].fillna(1.0).values
            true_arr = fd[true_col].values
            pred_arr = fd[pred_col].values

            true_s = float(np.mean(true_arr))
            pred_s = float(np.mean(pred_arr))
            rel_s = (pred_s - true_s) / true_s * 100 if true_s != 0 else np.nan

            w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop)) / len(pop)
            true_w = float(np.sum(w * true_arr))
            pred_w = float(np.sum(w * pred_arr))
            rel_w = (pred_w - true_w) / true_w * 100 if true_w != 0 else np.nan

            res = pred_arr - true_arr

            row = {'Family': FAMILY, 'Bucket': bucket, 'ParamKey': param_key(params),
                   'Gas': gas, 'Fold': int(fold_id), 'N_samples': len(fd)}
            row.update({k: params.get(k) for k in PARAM_KEYS})
            row.update({
                'RMSE': round(float(np.sqrt(np.mean(res ** 2))), 6),
                'MAE':  round(float(np.mean(np.abs(res))), 6),
                'Relative_Error_simple_%': round(rel_s, 2),
                'Relative_Error_wtd_%':    round(rel_w, 2),
            })
            rows.append(row)
    return rows


def run_one_bucket(bucket, params, train_df, available_features,
                   n2o_raw, ch4_raw, fb_by_type,
                   n2o_type_mean, ch4_type_mean):
    """Run 10-fold CV, bootstrap and Monte Carlo, and return 20 fold-level rows."""
    t0 = time.time()

    bucket_df = train_df[train_df['population_bucket'] == bucket].copy()
    if len(bucket_df) < 30:
        print(f"  ! {bucket}: too few samples, skipped")
        return []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_all = bucket_df[LABEL_COL].values

    tmp_prefix = os.path.join(TEMP_DIR, BUCKET_SLUG[bucket])
    os.makedirs(tmp_prefix, exist_ok=True)
    fold_proba_files = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(bucket_df, y_all)):
        fold_num = fold_idx + 1
        fold_file = os.path.join(tmp_prefix, f'fold_{fold_num:02d}_proba.csv')
        fold_proba_files.append(fold_file)

        fold_train = bucket_df.iloc[tr_idx].copy()
        fold_val = bucket_df.iloc[val_idx].copy()

        X_train = fold_train[available_features].copy()
        y_train = fold_train[LABEL_COL].values
        X_val = fold_val[available_features].copy()

        for col in X_train.select_dtypes(include=np.number).columns:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)

        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            all_cats = pd.Categorical(
                pd.concat([X_train[col], X_val[col]], ignore_index=True)).categories
            X_train[col] = pd.Categorical(X_train[col], categories=all_cats).codes
            X_val[col] = pd.Categorical(X_val[col], categories=all_cats).codes

        unique_cats = sorted(set(y_train))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}

        val_mask = np.array([t in cat_to_idx for t in fold_val[LABEL_COL].values])
        if not np.all(val_mask):
            fold_val = fold_val.iloc[val_mask]
            X_val = X_val.iloc[val_mask]
        if len(fold_val) == 0:
            continue

        y_enc = np.array([cat_to_idx[c] for c in y_train])
        treatment_levels = (fold_val['Treatment Level'].tolist()
                            if 'Treatment Level' in fold_val.columns else None)

        boot_results = run_bootstrap(params, X_train, y_enc, X_val,
                                     unique_cats, treatment_levels)

        stats_df = compute_proba_stats(boot_results, unique_cats)
        stats_df.index = fold_val.index
        stats_df.insert(0, 'fold', fold_num)
        stats_df.insert(1, 'true_biotreat', fold_val[LABEL_COL].values)
        stats_df.insert(2, 'population_bucket', bucket)
        stats_df.insert(3, 'served_population', fold_val[POP_COL].values)
        stats_df.to_csv(fold_file, index=False, encoding='utf-8')

    all_proba = [pd.read_csv(f) for f in fold_proba_files if os.path.exists(f)]
    if not all_proba:
        shutil.rmtree(tmp_prefix, ignore_errors=True)
        return []

    proba_df = pd.concat(all_proba, ignore_index=True)
    unique_cats_mc = sorted([
        c.replace('_mean', '') for c in proba_df.columns
        if c.endswith('_mean') and c.replace('_mean', '') in TREATMENT_METHODS
    ])

    chunk_size = max(1, len(proba_df) // N_CORES_MC)
    chunks = [proba_df.iloc[i:i + chunk_size]
              for i in range(0, len(proba_df), chunk_size)]

    base = zlib.crc32(f'{RANDOM_SEED}_{FAMILY}_{bucket}'.encode()) % (2 ** 31)
    args_list = [(chunk, base + 7919 * i, unique_cats_mc, n2o_raw, ch4_raw,
                  fb_by_type, n2o_type_mean, ch4_type_mean)
                 for i, chunk in enumerate(chunks)]

    all_rows = []
    with Pool(N_CORES_MC) as pool:
        for batch in pool.imap(process_mc_chunk, args_list):
            all_rows.extend(batch)
    gc.collect()

    plant_df = pd.DataFrame(all_rows)
    fold_metric_rows = compute_fold_metrics(plant_df, bucket, params)

    shutil.rmtree(tmp_prefix, ignore_errors=True)
    del proba_df, plant_df, all_rows
    gc.collect()

    fold_metric_rows_elapsed = time.time() - t0
    print(f'    {fold_metric_rows_elapsed / 60:.1f} min', end='')
    return fold_metric_rows


def joint_rmse(rows):
    """Sum of the two per-gas 10-fold mean RMSE values. Lower is better."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    total = 0.0
    for gas in ('N2O', 'CH4'):
        vals = df[df['Gas'] == gas]['RMSE'].dropna()
        if len(vals) == 0:
            return np.inf
        total += float(vals.mean())
    return total


def load_done(bucket):
    """Parameter sets already evaluated for one bucket."""
    p = done_path(bucket)
    if not os.path.exists(p):
        return set()
    with open(p, 'r', encoding='utf-8') as f:
        return {ln.strip() for ln in f if ln.strip()}


def append_done(bucket, key):
    """Mark one parameter set as evaluated."""
    with open(done_path(bucket), 'a', encoding='utf-8') as f:
        f.write(key + '\n')


def load_top(bucket):
    """Recover the retained parameter sets, best first."""
    p = top_path(bucket)
    if not os.path.exists(p):
        return []
    try:
        df = pd.read_csv(p).drop(columns=['Rank'], errors='ignore')
    except Exception:
        return []
    top = [{'key': k, 'score': joint_rmse(g), 'rows': g.to_dict('records')}
           for k, g in df.groupby('ParamKey', sort=False)]
    top.sort(key=lambda e: e['score'])
    return top[:N_KEEP]


def save_top(bucket, top):
    """Overwrite the retained table atomically, ranked best first."""
    frames = []
    for rank, e in enumerate(top, 1):
        g = pd.DataFrame(e['rows'])
        g.insert(0, 'Rank', rank)
        frames.append(g)
    p = top_path(bucket)
    tmp = p + '.tmp'
    pd.concat(frames, ignore_index=True).to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, p)


def main():
    """Search every parameter set on every bucket, keeping the best three."""
    t0 = time.time()
    grid = expand_grid()

    print('=' * 70)
    print(f'RF.py   family = {FAMILY}')
    print(f'CONFIG_FINGERPRINT = {CONFIG_FINGERPRINT}')
    print(f'grid {len(grid)} x {len(BUCKETS)} buckets = {len(grid) * len(BUCKETS)} runs')
    print(f'start {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'B={B}, MC={N_SIMULATIONS}, folds={N_FOLDS}, seed={RANDOM_SEED}')
    print(f'cores {AVAILABLE_CORES} | bootstrap={N_JOBS_BOOTSTRAP}, '
          f'model={N_JOBS_MODEL}, mc={N_CORES_MC}')
    print('=' * 70)

    df = pd.read_excel(INPUT_FILE)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    print(f'rows {len(df)} | features {len(available_features)}/{len(FEATURE_COLS)}')
    if missing:
        print(f'  ! missing features: {missing}')

    train_df = df.dropna(subset=[LABEL_COL]).copy()
    train_df['population_bucket'] = train_df[POP_COL].apply(get_population_bucket)
    print(f'labelled samples {len(train_df)}')
    print(train_df['population_bucket'].value_counts().to_string())

    n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean = load_ef_data()

    for bucket in BUCKETS:
        done = load_done(bucket)
        top = load_top(bucket)

        print('\n' + '=' * 70)
        print(f'{bucket}   done {len(done)}/{len(grid)}')
        for rank, e in enumerate(top, 1):
            print(f'  {rank}. joint RMSE = {e["score"]:.6f}   {e["key"]}')

        for i, params in enumerate(grid, 1):
            key = param_key(params)
            if key in done:
                continue

            print(f'  [{i}/{len(grid)}] {key}', end='', flush=True)
            rows = run_one_bucket(bucket, params, train_df, available_features,
                                  n2o_raw, ch4_raw, fb_by_type,
                                  n2o_type_mean, ch4_type_mean)

            if rows:
                score = joint_rmse(rows)
                cutoff = top[-1]['score'] if len(top) == N_KEEP else np.inf
                rank = None
                if score < cutoff:
                    top.append({'key': key, 'score': score, 'rows': rows})
                    top.sort(key=lambda e: e['score'])
                    del top[N_KEEP:]
                    save_top(bucket, top)
                    rank = next(i for i, e in enumerate(top, 1) if e['key'] == key)
                print(f'  joint RMSE = {score:.6f}'
                      f'{f"   <- rank {rank}" if rank else ""}')
            else:
                print()

            append_done(bucket, key)

    print('\n' + '=' * 70)
    print('retained parameter sets')
    for bucket in BUCKETS:
        top = load_top(bucket)
        print(f'  {bucket}')
        if not top:
            print('    no result')
        for rank, e in enumerate(top, 1):
            print(f'    {rank}. joint RMSE = {e["score"]:.6f}   {e["key"]}')

    h, rem = divmod(int(time.time() - t0), 3600)
    m, s = divmod(rem, 60)
    print(f'\ndone in {h}h{m}m{s}s')


if __name__ == '__main__':
    _method = 'fork' if platform.system() != 'Windows' else 'spawn'
    try:
        multiprocessing.set_start_method(_method, force=True)
    except RuntimeError:
        pass
    main()