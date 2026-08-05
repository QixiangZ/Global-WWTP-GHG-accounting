"""Leave-country-out validation: bootstrap training and country-level EF prediction intervals."""
import os, re, json, glob, time, random, getpass, datetime, warnings, hashlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import multiprocessing

from paths import OUTPUT_DIR, inp, out

warnings.filterwarnings('ignore')

LEAVE_OUT_COUNTRIES = ['South Korea', 'China', 'Mexico', 'Canada', 'Japan', 'India',
    'Thailand', 'Tunisia', 'Jamaica', 'Jordan', 'Vietnam','Bolivia',
    'Singapore', 'Nepal', 'Bahrain', 'Kuwait','Morocco', 'Botswana', 'Yemen', 'Bhutan',
    'Virgin Islands, U.S.', 'Saint-Martin', 'Reunion', 'Puerto Rico', 'Northern Mariana Islands',
    'Mayotte', 'Martinique', 'Guam', 'Guadeloupe', 'French Guiana', 'American Samoa','France','United States']

STAGE_A_MODE = 'auto'

RUN_STAGE_A = (STAGE_A_MODE != 'off')
RUN_STAGE_B = True
RUN_PLANT_LEVEL_MC = False

B             = 1000
N_SIMULATIONS = 10000

# --- input ---------------------------------------------------------------
INPUT_FILE  = inp('combined_wwtp.xlsx')
N2O_EF_FILE = inp('N2O_EF.xlsx')
CH4_EF_FILE = inp('CH4_EF.xlsx')
FB_FILE     = inp('FB.xlsx')

N2O_EF_COL = 'N2OEF'
CH4_EF_COL = 'CH4EF'
FB_COL     = 'FB'

# --- output --------------------------------------------------------------
OUT_DIR       = OUTPUT_DIR
PROGRESS_FILE = out('loco_progress.json')
MANIFEST_FILE = out('loco_stage_a_manifest.json')

# optional intermediate file from Validation.py, read back from output/.
# when it is absent the reference EFs are recomputed here, see build_true_maps().
CORRECTED_XLSX  = out('validation_summary_recomputed.xlsx')
CORRECTED_SHEET = 'Plant_Details'

TAG = f'{getpass.getuser()}_{datetime.datetime.now():%m%d_%H%M}'

N_SIM_COUNTRY = 5000
POOL_SIZE     = 20000
IQR_K         = 1.5
RANDOM_SEED   = 42
MIN_PLANTS_WARN = 50
INCLUDE_PROB_UNCERTAINTY = True

HOLDOUT_MODE      = True
LARGE_COUNTRY_MIN = 1000
SMALL_COUNTRY_MIN = 20
LARGE_TEST_FRACS  = [0.2]
SMALL_TEST_FRACS  = [0.5]
HOLDOUT_SEED      = 42
BELOW_MIN_ACTION  = 'skip'

COUNTRY_COL = 'NAME_0'
LABEL_COL   = 'Biotreat Type'
POP_COL     = 'Current Served Population (estimate)'
P_COUNTRY_COL = 'country'
P_POP_COL     = 'served_population'
P_TYPE_COL    = 'true_biotreat'

AVAILABLE_CORES = int(os.environ.get('NCPUS', 0)) or cpu_count()
if AVAILABLE_CORES >= 16:
    N_JOBS_BOOTSTRAP = 8; N_JOBS_XGB = 2
elif AVAILABLE_CORES >= 8:
    N_JOBS_BOOTSTRAP = 4; N_JOBS_XGB = 2
else:
    N_JOBS_BOOTSTRAP = 2; N_JOBS_XGB = 1
N_CORES_MC = min(16, AVAILABLE_CORES)

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
    'A2O','AO','Anaerobic','BAF','BNR','CAS','EA','Lagoon',
    'MBR','MLE','OD','Pond','RBC','SBR','TF','Wetland'
}
CH4_DEFAULT_VALUES      = {'Pond': 0.102, 'Lagoon': 0.102, 'Anaerobic': 0.408}
CH4_DEFAULT_UNCERTAINTY = 0.30
N2O_DEFAULT_VALUES      = {'Anaerobic': 0}
ADVANCED_TYPES          = {'AO','BAF','BNR','EA','MBR','MLE','A2O','SBR','OD'}


def safe_name(country):
    """Turn a country name into a filesystem-safe stem."""
    return re.sub(r'[^\w\-]+', '_', str(country)).strip('_')


def proba_path_of(country, pct=None):
    """Path of the probability file for one country and holdout fraction."""
    base = safe_name(country)
    if pct is None:
        return os.path.join(OUT_DIR, f'{base}_proba.csv')
    return os.path.join(OUT_DIR, f'{base}_holdout{int(pct)}_proba.csv')


def get_bucket_params(bucket):
    """Return the XGBoost hyperparameters for one bucket."""
    base = {'subsample': 0.8, 'colsample_bytree': 0.8}
    if 'small' in bucket:
        base.update({'max_depth': 8, 'n_estimators': 100, 'learning_rate': 0.10,
                     'min_child_weight': 5, 'reg_lambda': 5})
    elif 'medium' in bucket:
        base.update({'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.05,
                     'min_child_weight': 3, 'reg_lambda': 1})
    else:
        base.update({'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.10,
                     'min_child_weight': 3, 'reg_lambda': 3})
    return base


def get_population_bucket(pop):
    """Assign a plant to a served-population bucket."""
    if pd.isna(pop): return 'medium: 2000-10000'
    if pop <= 2000:  return 'small: 0-2000'
    if pop <= 10000: return 'medium: 2000-10000'
    return 'large: 10000+'


MIN_GROUP_N      = 4
FB_NORM_FLOOR = 0.10

FB_SAMPLING_FLOOR = None

_EF_CACHE = {}


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


def _clip_floor(vals, floor=FB_NORM_FLOOR):
    """Drop NaN and clip to a lower bound."""
    a = np.asarray(list(vals), dtype=float)
    a = a[~np.isnan(a)]
    if floor is not None:
        a = np.clip(a, floor, None)
    return a


def load_cleaned_ef_frames(verbose=True):
    """Load and IQR-clean the EF tables once per run, then cache."""
    if 'frames' in _EF_CACHE:
        return _EF_CACHE['frames']

    if verbose:
        print('loading EF data, shared by stage A and stage B...')

    ratio_df = pd.read_excel(FB_FILE)
    fb_by_type = {}
    for _, r in ratio_df.iterrows():
        fb_by_type.setdefault(r['Type'], []).append(float(r[FB_COL]))

    ratio_mean_by_type = {t: float(_clip_floor(v).mean()) if len(_clip_floor(v)) else np.nan
                          for t, v in fb_by_type.items()}
    all_flat = [v for vv in fb_by_type.values() for v in vv]
    gmr = float(_clip_floor(all_flat).mean()) if all_flat else 1.0

    n_low = int((np.asarray(all_flat, dtype=float) < (FB_NORM_FLOOR or 0)).sum())
    if n_low and verbose:
        print(f'  ! {n_low} FB values below {FB_NORM_FLOOR}. The normalising mean is clipped,')
        print('    but sample_n2o_ef and sample_n2o_type divide by a randomly drawn FB value,')
        print('    so a draw near zero inflates the EF. See FB_SAMPLING_FLOOR.')

    if FB_SAMPLING_FLOOR is not None:
        fb_by_type = {t: [max(float(v), FB_SAMPLING_FLOOR) for v in vv]
                         for t, vv in fb_by_type.items()}
        if verbose:
            print(f'  sampling FB values clipped to {FB_SAMPLING_FLOOR}')

    if verbose:
        print('\n  --- N2O ---')
    n2o_df = pd.read_excel(N2O_EF_FILE)
    if 'F/B' not in n2o_df.columns:
        if verbose:
            print("  ! no 'F/B' column, every record treated as non-F")
        n2o_df['F/B'] = np.nan

    is_F = n2o_df['F/B'].astype(str).str.strip().eq('F')
    r = n2o_df['Type'].map(lambda t: ratio_mean_by_type.get(t, gmr))
    r = r.replace(0, np.nan).fillna(gmr)
    n2o_df['_screen'] = np.where(is_F, n2o_df[N2O_EF_COL], n2o_df[N2O_EF_COL] / r)

    mixed = [t for t, g in n2o_df.groupby('Type')
             if is_F.loc[g.index].any() and (~is_F.loc[g.index]).any()]
    if verbose:
        print(f'  F/non-F split: F={int(is_F.sum())}, non-F={int((~is_F).sum())} | '
              f'{len(mixed)} mixed Types {sorted(mixed)}')

    n2o_df, _ = iqr_clean_by_type(n2o_df, '_screen', label=f'{N2O_EF_COL} (normalised scale)',
                                  verbose=verbose)
    n2o_df = n2o_df.drop(columns=['_screen'])

    if verbose:
        print('\n  --- CH4 ---')
    ch4_df = pd.read_excel(CH4_EF_FILE)
    ch4_df, _ = iqr_clean_by_type(ch4_df, CH4_EF_COL, label="CH4_EF_COL", verbose=verbose)

    if verbose:
        print(f'\n  N2O EF: {len(n2o_df)} | CH4 EF: {len(ch4_df)} | '
              f'FB: {len(all_flat)}, global mean {gmr:.4f}')

    _EF_CACHE['frames'] = (n2o_df, ch4_df, fb_by_type, ratio_mean_by_type, gmr)
    return _EF_CACHE['frames']


def n2o_true_ef(ef, fb, t, fb_by_type, gmr):
    """Rescale one N2O record to the F scale, matching the sampling side."""
    ef = float(ef)
    if str(fb).strip() == 'F':
        return ef
    rs = _clip_floor([r for r in fb_by_type.get(t, []) if r > 0])
    r = float(rs.mean()) if len(rs) else gmr
    return ef / r if r > 0 else ef


def load_ef_data():
    """Return EF records and per-type means for the plant-level Monte Carlo."""
    n2o_df, ch4_df, fb_by_type, ratio_mean_by_type, gmr = load_cleaned_ef_frames()

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
    if t in N2O_DEFAULT_VALUES: return N2O_DEFAULT_VALUES[t]
    records = [r for r in n2o_raw if r['Type'] == t]
    if not records: return None
    rec = random.choice(records); ef = rec[N2O_EF_COL]
    if str(rec.get('F/B', '')).strip() != 'F':
        ratios = fb_by_type.get(t)
        if ratios: ef = ef / random.choice(ratios)
        else:
            all_r = [v for vals in fb_by_type.values() for v in vals]
            ef = ef / (np.mean(all_r) if all_r else 1.0)
    return ef


def sample_ch4_ef(t, ch4_raw):
    """Draw one CH4 EF for a treatment type."""
    if t in CH4_DEFAULT_VALUES:
        d = CH4_DEFAULT_VALUES[t]; std = d * CH4_DEFAULT_UNCERTAINTY
        for _ in range(10):
            v = np.random.normal(d, std)
            if v >= 0: return v
        return abs(np.random.normal(d, std))
    records = [r for r in ch4_raw if r['Type'] == t]
    if not records: return None
    return random.choice(records)[CH4_EF_COL]


def bootstrap_worker(args):
    """Fit one bootstrap replicate and return class probabilities."""
    b_idx, X_tr_rec, y_enc, X_pr_rec, unique_cats, bp, treatment_levels = args
    try:
        X_train = pd.DataFrame(X_tr_rec); X_pred = pd.DataFrame(X_pr_rec)
        np.random.seed(b_idx + 42)
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        model = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            n_estimators=bp['n_estimators'], max_depth=bp['max_depth'],
            learning_rate=bp['learning_rate'], subsample=bp['subsample'],
            colsample_bytree=bp['colsample_bytree'],
            min_child_weight=bp.get('min_child_weight', 1),
            reg_lambda=bp.get('reg_lambda', 1),
            n_jobs=N_JOBS_XGB, random_state=b_idx+42,
            verbosity=0, tree_method='hist', max_bin=256)
        model.fit(X_train.iloc[idx], y_enc[idx])
        proba = model.predict_proba(X_pred)
        if treatment_levels is not None:
            allowed = np.array([c in ADVANCED_TYPES for c in unique_cats])
            if np.any(allowed):
                for i, lvl in enumerate(treatment_levels):
                    if pd.notna(lvl) and str(lvl).strip() == 'Advanced':
                        proba[i, ~allowed] = 0
                        s = proba[i].sum()
                        proba[i] = proba[i]/s if s > 0 else np.where(allowed, 1.0/np.sum(allowed), 0)
        return b_idx, proba
    except Exception as e:
        print(f"  bootstrap {b_idx} failed: {e}")
        return b_idx, np.zeros((len(X_pr_rec), len(unique_cats)))


def run_bootstrap(X_train, y_enc, X_pred, unique_cats, bp, treatment_levels):
    """Run B bootstrap replicates and return (B, n_pred, n_classes)."""
    X_tr_rec = X_train.to_dict('records'); X_pr_rec = X_pred.to_dict('records')
    args_list = [(b, X_tr_rec, y_enc, X_pr_rec, unique_cats, bp, treatment_levels)
                 for b in range(B)]
    results = np.zeros((B, len(X_pred), len(unique_cats)))
    try:
        with ProcessPoolExecutor(max_workers=N_JOBS_BOOTSTRAP) as ex:
            futures = {ex.submit(bootstrap_worker, a): a[0] for a in args_list}
            for fut in as_completed(futures):
                b_idx = futures[fut]
                _, proba = fut.result()
                results[b_idx] = proba
    except Exception as e:
        print(f"  parallel failed, running serially: {e}")
        for a in args_list:
            b_idx, proba = bootstrap_worker(a); results[b_idx] = proba
    return results


def compute_proba_stats(boot, unique_cats):
    """Summarise bootstrap probabilities into per-plant statistics."""
    d = {}
    for ci, cat in enumerate(unique_cats):
        cp = boot[:, :, ci]
        d[f'{cat}_mean'] = np.mean(cp, axis=0)
        d[f'{cat}_std']  = np.std(cp, axis=0)
    means = np.column_stack([d[f'{c}_mean'] for c in unique_cats])
    pred_idx = np.argmax(means, axis=1)
    d['predicted_biotreat'] = [unique_cats[i] for i in pred_idx]
    d['confidence'] = np.max(means, axis=1)
    eps = 1e-10; safe = np.clip(means, eps, 1-eps)
    d['prediction_entropy'] = -np.sum(safe*np.log(safe), axis=1)
    return pd.DataFrame(d)


def sample_beta_prob(m, s):
    """Draw a class probability from a Beta matched to mean and std."""
    if pd.isna(m) or m <= 0: return 0.0
    if pd.isna(s) or s <= 0: return float(m)
    m = np.clip(float(m), 0.001, 0.999)
    var = min(float(s)**2, m*(1-m)*0.99)
    if var <= 0: return m
    try:
        cf = m*(1-m)/var - 1
        if cf <= 0: return m
        a, b = m*cf, (1-m)*cf
        return float(np.random.beta(a, b)) if (a>0 and b>0) else m
    except Exception:
        return float(np.clip(np.random.normal(m, s), 0, 1))


def mc_single(row, unique_cats, n2o_raw, ch4_raw, fb_by_type):
    """Propagate classification and EF uncertainty for one plant."""
    n2o_s = np.zeros(N_SIMULATIONS, dtype=np.float32)
    ch4_s = np.zeros(N_SIMULATIONS, dtype=np.float32)
    for sim in range(N_SIMULATIONS):
        probs = {}
        for cat in unique_cats:
            p = sample_beta_prob(row.get(f'{cat}_mean', np.nan), row.get(f'{cat}_std', np.nan))
            if p > 0: probs[cat] = p
        total = sum(probs.values())
        if total <= 0: continue
        probs = {k: v/total for k, v in probs.items()}
        n2o_ef = ch4_ef = 0.0
        for cat, prob in probs.items():
            nv = sample_n2o_ef(cat, n2o_raw, fb_by_type)
            cv = sample_ch4_ef(cat, ch4_raw)
            if nv is not None: n2o_ef += prob*nv
            if cv is not None: ch4_ef += prob*cv
        n2o_s[sim] = n2o_ef; ch4_s[sim] = ch4_ef
    return n2o_s, ch4_s


def process_mc_chunk(args):
    """Run the plant-level Monte Carlo for a chunk of plants."""
    chunk, unique_cats, n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean = args
    np.random.seed(int(time.time()*1000) % (2**31) + os.getpid())
    random.seed(int(time.time()*1000) % (2**31) + os.getpid())
    rows = []
    for _, row in chunk.iterrows():
        n2o_s, ch4_s = mc_single(row, unique_cats, n2o_raw, ch4_raw, fb_by_type)
        tt = row.get('true_biotreat', '')
        rows.append({
            'serial_number': row.get('serial_number', row.name),
            'country': row.get('country', ''),
            'true_biotreat': tt,
            'predicted_biotreat': row.get('predicted_biotreat', ''),
            'confidence': row.get('confidence', np.nan),
            'served_population': row.get('served_population', np.nan),
            'population_bucket': row.get('population_bucket', ''),
            'N2O_true_EF': n2o_type_mean.get(tt, np.nan),
            'N2O_pred_EF_mean': float(np.mean(n2o_s)),
            'CH4_true_EF': ch4_type_mean.get(tt, np.nan),
            'CH4_pred_EF_mean': float(np.mean(ch4_s)),
        })
    return rows


def load_progress():
    """Read which country and bucket jobs are already done."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f: return json.load(f)
        except Exception: pass
    return {'done_buckets': {}}


def save_progress(p):
    """Persist the completed job list."""
    with open(PROGRESS_FILE, 'w') as f: json.dump(p, f, indent=2, ensure_ascii=False)


def _file_sig(path):
    """Size and hash signature of one input file."""
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return f'{st.st_size}_{int(st.st_mtime)}'


def stage_a_fingerprint(feats):
    """Fingerprint of everything that affects the stored probabilities."""
    payload = {
        'input_file': _file_sig(INPUT_FILE),
        'features': list(feats),
        'label_col': LABEL_COL, 'country_col': COUNTRY_COL, 'pop_col': POP_COL,
        'B': B,
        'holdout_mode': HOLDOUT_MODE, 'holdout_seed': HOLDOUT_SEED,
        'large_country_min': LARGE_COUNTRY_MIN, 'small_country_min': SMALL_COUNTRY_MIN,
        'large_test_fracs': list(LARGE_TEST_FRACS),
        'small_test_fracs': list(SMALL_TEST_FRACS),
        'below_min_action': BELOW_MIN_ACTION,
        'advanced_types': sorted(ADVANCED_TYPES),
        'bucket_params': {b: get_bucket_params(b) for b in
                          ('small: 0-2000', 'medium: 2000-10000', 'large: 10000+')},
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(s.encode('utf-8')).hexdigest(), payload


def check_stage_a_manifest(feats):
    """Stop if the configuration changed since the stored probabilities were written."""
    fp, payload = stage_a_fingerprint(feats)
    old = None
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, encoding='utf-8') as f:
                old = json.load(f)
        except Exception:
            old = None

    if old is None or STAGE_A_MODE == 'force':
        with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump({'fingerprint': fp, 'payload': payload, 'tag': TAG},
                      f, indent=2, ensure_ascii=False)
        print('stage A fingerprint written')
        return

    if old.get('fingerprint') == fp:
        print('stage A fingerprint matches, stored probabilities can be reused')
        return

    oldp = old.get('payload', {})
    changed = [k for k in payload
               if json.dumps(payload[k], sort_keys=True, default=str)
               != json.dumps(oldp.get(k), sort_keys=True, default=str)]
    print('\n' + '!' * 70)
    print('stage A configuration changed, stored probabilities cannot be reused. Changes:')
    for k in changed:
        print(f'   {k}: {oldp.get(k)}  ->  {payload[k]}')
    print("Use a different OUT_DIR, or set STAGE_A_MODE to 'force'")
    print('!' * 70 + '\n')
    raise SystemExit(1)


def _atomic_write_csv(df, path):
    """Write to a temporary file and replace, so an interrupt leaves no partial CSV."""
    keys = [k for k in ('serial_number', 'holdout_pct') if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep='last')
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stratified_holdout(cdf, frac, seed):
    """Draw a stratified holdout fold within population bucket and true type."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in cdf.groupby(['population_bucket', LABEL_COL]):
        idx = grp.index.to_numpy()
        k = min(max(int(round(len(idx) * frac)), 0), len(idx))
        if k > 0:
            test_idx.extend(rng.choice(idx, size=k, replace=False).tolist())
    m = cdf.index.isin(test_idx)
    return cdf[m].copy(), cdf[~m].copy()


def build_holdout_jobs(df):
    """Decide the holdout fraction per country from its plant count."""
    counts = (df[df[COUNTRY_COL].isin(LEAVE_OUT_COUNTRIES)]
              .groupby(COUNTRY_COL).size().to_dict())
    jobs, skipped = [], []
    for c in LEAVE_OUT_COUNTRIES:
        n = int(counts.get(c, 0))
        if not HOLDOUT_MODE:
            jobs.append((c, None, None)); continue
        if n == 0:
            skipped.append((c, 'name not found in the data')); continue
        if n > LARGE_COUNTRY_MIN:
            fracs = LARGE_TEST_FRACS
        elif n >= SMALL_COUNTRY_MIN:
            fracs = SMALL_TEST_FRACS
        else:
            if BELOW_MIN_ACTION == 'skip':
                skipped.append((c, f'{n} plants < {SMALL_COUNTRY_MIN}')); continue
            if BELOW_MIN_ACTION == 'lco':
                jobs.append((c, None, None)); continue
            fracs = SMALL_TEST_FRACS
        for fr in fracs:
            jobs.append((c, float(fr), int(round(fr * 100))))
    if skipped:
        print(f"\nskipped (BELOW_MIN_ACTION='{BELOW_MIN_ACTION}'):")
        for c, why in skipped:
            print(f"   {c}（{why}）")
    return jobs


def run_one_country(country, frac, pct, df, feats, ef, progress):
    """Train on everything else and predict one country or one holdout fold."""
    n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean = ef
    proba_path = proba_path_of(country, pct)
    prog_key   = f'{safe_name(country)}|{"LCO" if pct is None else pct}'
    tag        = 'full leave-country-out' if pct is None else f'{pct}% holdout fold'

    country_df = df[df[COUNTRY_COL] == country].copy()
    others_df  = df[df[COUNTRY_COL] != country].copy()
    if len(country_df) == 0:
        print(f"\n{'='*70}\n{country}: no labelled plants in "
              f"{os.path.basename(INPUT_FILE)}, skipped\n{'='*70}")
        return None

    for d in (country_df, others_df):
        d['population_bucket'] = d[POP_COL].apply(get_population_bucket)
        d['serial_number'] = d.index

    if frac is None:
        test_df, retain_df = country_df, country_df.iloc[0:0]
    else:
        test_df, retain_df = stratified_holdout(country_df, frac, HOLDOUT_SEED)

    train_df = pd.concat([others_df, retain_df], ignore_index=False)

    print(f"\n{'='*70}\nheld out: {country} ({tag})  test={len(test_df)}  "
          f"train={len(train_df)} (other countries {len(others_df)} + retained {len(retain_df)})"
          f"\n{'='*70}")
    if len(test_df) == 0:
        print("  ! empty test set, skipped"); return None

    miss = set(test_df[LABEL_COL].unique()) - set(train_df[LABEL_COL].unique())
    if miss:
        print(f"  ! test set has types absent from training, they will be skipped: {miss}")

    if STAGE_A_MODE == 'force':
        done, fold_rows = set(), []
        if os.path.exists(proba_path):
            os.replace(proba_path, proba_path + '.bak')
            print(f"  force mode: previous probabilities backed up as {os.path.basename(proba_path)}.bak")
        progress['done_buckets'].pop(prog_key, None)
        save_progress(progress)
    else:
        done = set(progress['done_buckets'].get(prog_key, []))
        fold_rows = []
        if os.path.exists(proba_path):
            prev = pd.read_csv(proba_path)
            fold_rows.append(prev)
            if 'population_bucket' in prev.columns:
                done |= set(prev['population_bucket'].dropna().astype(str).unique())
            progress['done_buckets'][prog_key] = sorted(done)
            save_progress(progress)

        expected = set(test_df['population_bucket'].astype(str).unique())
        if expected and expected.issubset(done):
            print(f'  cache hit ({len(expected)} buckets complete), training skipped')
            if not RUN_PLANT_LEVEL_MC:
                return None

    for bucket in sorted(test_df['population_bucket'].unique()):
        if bucket in done:
            print(f"  [{bucket}] already done, skipped"); continue
        tr_b = train_df[train_df['population_bucket'] == bucket]
        te_b = test_df[test_df['population_bucket'] == bucket]
        if len(tr_b) < 10 or len(te_b) == 0:
            print(f"  [{bucket}] too few samples, skipped"); continue
        print(f"  [{bucket}] train={len(tr_b)} test={len(te_b)}")

        X_train = tr_b[feats].copy(); y_train = tr_b[LABEL_COL].values
        X_test  = te_b[feats].copy()
        for col in X_train.select_dtypes(include=np.number).columns:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med); X_test[col] = X_test[col].fillna(med)
        for col in X_train.select_dtypes(include=['object','category']).columns:
            cats = pd.Categorical(pd.concat([X_train[col], X_test[col]], ignore_index=True)).categories
            X_train[col] = pd.Categorical(X_train[col], categories=cats).codes
            X_test[col]  = pd.Categorical(X_test[col],  categories=cats).codes

        unique_cats = sorted(set(y_train)); c2i = {c:i for i,c in enumerate(unique_cats)}
        mask = np.array([t in c2i for t in te_b[LABEL_COL].values])
        if not np.all(mask):
            te_b = te_b.iloc[mask]; X_test = X_test.iloc[mask]
        if len(te_b) == 0: continue
        y_enc = np.array([c2i[c] for c in y_train])
        tl = te_b['Treatment Level'].tolist() if 'Treatment Level' in te_b.columns else None

        boot = run_bootstrap(X_train, y_enc, X_test, unique_cats, get_bucket_params(bucket), tl)
        sdf = compute_proba_stats(boot, unique_cats); sdf.index = te_b.index
        sdf.insert(0, 'serial_number', te_b['serial_number'].values)
        sdf.insert(1, 'country', country)
        sdf.insert(2, 'true_biotreat', te_b[LABEL_COL].values)
        sdf.insert(3, 'population_bucket', bucket)
        sdf.insert(4, 'served_population', te_b[POP_COL].values)
        sdf.insert(5, 'holdout_pct', 'LCO' if pct is None else pct)
        if 'Treatment Level' in te_b.columns:
            sdf['treatment_level'] = te_b['Treatment Level'].values
        fold_rows.append(sdf)

        _atomic_write_csv(pd.concat(fold_rows, ignore_index=True), proba_path)
        done.add(bucket); progress['done_buckets'][prog_key] = sorted(done)
        save_progress(progress)
        print(f"    [{bucket}] done and saved")

    if not fold_rows:
        print("  ! no bucket produced a result for this country")
        return None
    if not RUN_PLANT_LEVEL_MC:
        return None

    proba_df = pd.concat(fold_rows, ignore_index=True)
    unique_cats = sorted([c.replace('_mean','') for c in proba_df.columns
                          if c.endswith('_mean') and c.replace('_mean','') in TREATMENT_METHODS])
    cs = max(1, len(proba_df)//N_CORES_MC)
    chunks = [proba_df.iloc[i:i+cs] for i in range(0, len(proba_df), cs)]
    args = [(c, unique_cats, n2o_raw, ch4_raw, fb_by_type, n2o_type_mean, ch4_type_mean)
            for c in chunks]
    all_rows = []
    with Pool(N_CORES_MC) as pool:
        for batch in pool.imap(process_mc_chunk, args): all_rows.extend(batch)
    plant_df = pd.DataFrame(all_rows)
    plant_df.to_csv(os.path.join(OUT_DIR, f'{safe_name(country)}_plant_level.csv'), index=False)
    return plant_df


def wmean(v, w):
    """Population-weighted mean."""
    v = np.asarray(v, float); w = np.asarray(w, float)
    if w.sum() <= 0: w = np.ones_like(w)
    return float(np.sum(w/w.sum()*v))


def metrics(plant_df, country):
    """Plant-level RMSE, MAE and weighted error for one country."""
    d = plant_df.dropna(subset=['CH4_true_EF','CH4_pred_EF_mean',
                                'N2O_true_EF','N2O_pred_EF_mean']).copy()
    pop = d['served_population'].fillna(1.0).values
    out = {'country': country, 'N': len(d)}
    for gas, t, p, unit in [
        ('N2O', 'N2O_true_EF', 'N2O_pred_EF_mean', 'N2OEF'),
        ('CH4', 'CH4_true_EF', 'CH4_pred_EF_mean', 'CH4EF')]:
        err = d[p].values - d[t].values
        out[f'{gas}_RMSE({unit})'] = round(float(np.sqrt(np.mean(err**2))), 4)
        out[f'{gas}_MAE({unit})']  = round(float(np.mean(np.abs(err))), 4)
        tw, pw = wmean(d[t], pop), wmean(d[p], pop)
        out[f'{gas}_true_wtd'] = round(tw, 4)
        out[f'{gas}_pred_wtd'] = round(pw, 4)
        out[f'{gas}_wtd_err_%'] = round((pw-tw)/tw*100 if tw else np.nan, 2)
    return out


def load_ef_pools_source():
    """Load the cleaned EF records used to build the sampling pools."""
    n2o_df, ch4_df, fb_by_type, ratio_mean_by_type, gmr = load_cleaned_ef_frames()

    n2o_by_type, ch4_by_type = {}, {}
    for _, r in n2o_df.iterrows():
        n2o_by_type.setdefault(r['Type'], []).append(
            (float(r[N2O_EF_COL]), str(r.get('F/B', ''))))
    for _, r in ch4_df.iterrows():
        ch4_by_type.setdefault(r['Type'], []).append(float(r[CH4_EF_COL]))

    print(f'  stage B: N2O {len(n2o_df)} records / {len(n2o_by_type)} types, '
          f'CH4 {len(ch4_df)} records / {len(ch4_by_type)} types')
    return n2o_by_type, ch4_by_type, fb_by_type, gmr


def sample_n2o_type(t, n2o_by_type, fb_by_type, gmr, rng):
    """Draw one N2O EF for a treatment type using a shared generator."""
    if t in N2O_DEFAULT_VALUES:
        return float(N2O_DEFAULT_VALUES[t])
    recs = n2o_by_type.get(t)
    if not recs:
        return np.nan
    ef, fb = recs[rng.randrange(len(recs))]
    if str(fb).strip() != 'F':
        ratios = fb_by_type.get(t)
        r = ratios[rng.randrange(len(ratios))] if ratios else gmr
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


def build_ef_pools(cats, gas, ef_by_type, fb_by_type, gmr, seed, pool_size=POOL_SIZE):
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


def build_true_maps(n2o_by_type, ch4_by_type, fb_by_type, gmr):
    """Map each treatment type to its reference N2O and CH4 EF."""
    n2o_map, ch4_map = None, {}
    if os.path.exists(CORRECTED_XLSX):
        d = pd.read_excel(CORRECTED_XLSX, sheet_name=CORRECTED_SHEET, engine='openpyxl')
        if P_TYPE_COL in d.columns and 'N2O_true_EF' in d.columns:
            g = d.dropna(subset=[P_TYPE_COL, 'N2O_true_EF']).groupby(P_TYPE_COL)['N2O_true_EF']
            n2o_map = g.first().to_dict()
            multi = g.nunique()
            if (multi > 1).any():
                print(f'  ! types with several reference EFs, first one used: {list(multi[multi>1].index)}')
            print(f'  N2O reference values read for {len(n2o_map)} types')
        if P_TYPE_COL in d.columns and 'CH4_true_EF' in d.columns:
            ch4_map = d.dropna(subset=[P_TYPE_COL, 'CH4_true_EF']).groupby(
                P_TYPE_COL)['CH4_true_EF'].first().to_dict()

    if n2o_map is None:
        print(f'  ! {CORRECTED_XLSX} not found, N2O reference values computed here')
        n2o_map = {}
        for t, recs in n2o_by_type.items():
            n2o_map[t] = float(np.mean(
                [n2o_true_ef(ef, fb, t, fb_by_type, gmr) for ef, fb in recs]))
        for t, v in N2O_DEFAULT_VALUES.items():
            n2o_map[t] = float(v)

    if not ch4_map:
        for t, vals in ch4_by_type.items():
            if t not in CH4_DEFAULT_VALUES:
                ch4_map[t] = float(np.mean(vals))
        for t, v in CH4_DEFAULT_VALUES.items():
            ch4_map[t] = float(v)
    return n2o_map, ch4_map


def load_proba_for(countries):
    """Read every stored probability file for the requested countries."""
    frames = []
    for c in countries:
        base = safe_name(c)
        paths = []
        lco = os.path.join(OUT_DIR, f'{base}_proba.csv')
        if os.path.exists(lco):
            paths.append(lco)
        paths.extend(sorted(glob.glob(os.path.join(OUT_DIR, f'{base}_holdout*_proba.csv'))))
        if not paths:
            print(f'  ! no probability file for {base}')
            continue
        for f in paths:
            d = pd.read_csv(f)
            if P_COUNTRY_COL not in d.columns or d[P_COUNTRY_COL].isna().all():
                d[P_COUNTRY_COL] = c
            if 'holdout_pct' not in d.columns:
                d['holdout_pct'] = 'LCO'
            frames.append(d)
            print(f'  {os.path.basename(f)}: {len(d)} rows')
    if not frames:
        raise FileNotFoundError(
            f'no probability file found for this batch. Files present: '
            f'{sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT_DIR, "*_proba.csv")))}')
    df = pd.concat(frames, ignore_index=True)
    cats = sorted([c[:-5] for c in df.columns if c.endswith('_mean')
                   and f'{c[:-5]}_std' in df.columns])
    ncombo = df.groupby([P_COUNTRY_COL, 'holdout_pct']).ngroups
    print(f'  {len(df)} rows, {ncombo} country x holdout combinations, {len(cats)} types')
    return df, cats


def beta_params(mean_mat, std_mat):
    """Convert class probability mean and std into Beta parameters."""
    m = np.clip(mean_mat, 0.001, 0.999)
    var = np.minimum(std_mat ** 2, m * (1 - m) * 0.99)
    with np.errstate(divide='ignore', invalid='ignore'):
        cf = m * (1 - m) / var - 1
    ok = (var > 0) & (cf > 0) & np.isfinite(cf)
    a = np.where(ok, m * cf, 1.0)
    b = np.where(ok, (1 - m) * cf, 1.0)
    return a, b, ok, m


def country_mc(sub, cats, n2o_pools, ch4_pools, n_sim, seed):
    """Draw one type per plant, one EF per plant, and aggregate by population weight."""
    rng_np = np.random.default_rng(seed)

    pop = pd.to_numeric(sub[P_POP_COL], errors='coerce').fillna(1.0).values
    w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop)) / len(pop)
    n = len(sub); C = len(cats)

    mean_mat = np.nan_to_num(sub[[f'{c}_mean' for c in cats]].to_numpy(dtype=float))
    std_mat = np.nan_to_num(sub[[f'{c}_std' for c in cats]].to_numpy(dtype=float))

    n_pools = [n2o_pools.get(c) for c in cats]
    c_pools = [ch4_pools.get(c) for c in cats]

    n2o_out = np.empty(n_sim); ch4_out = np.empty(n_sim)

    P_fixed = None
    if not INCLUDE_PROB_UNCERTAINTY:
        p = np.where(mean_mat > 0, mean_mat, 0.0)
        tot = p.sum(axis=1, keepdims=True)
        P_fixed = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

    for k in range(n_sim):
        if P_fixed is not None:
            P = P_fixed
        else:
            a, b, ok, m = beta_params(mean_mat, std_mat)
            draw = rng_np.beta(a, b)
            p = np.where(ok, draw, m)
            p = np.where(mean_mat > 0, p, 0.0)
            tot = p.sum(axis=1, keepdims=True)
            P = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

        cdf = np.cumsum(P, axis=1)
        u = rng_np.random(size=(n, 1))
        cat_idx = np.clip((u > cdf).sum(axis=1), 0, C - 1)

        nef = np.full(n, np.nan); cef = np.full(n, np.nan)
        for ci in range(C):
            mask = cat_idx == ci
            n_need = int(mask.sum())
            if n_need == 0:
                continue
            np_pool = n_pools[ci]
            if np_pool is not None:
                nef[mask] = np_pool[rng_np.integers(0, len(np_pool), size=n_need)]
            cp_pool = c_pools[ci]
            if cp_pool is not None:
                cef[mask] = cp_pool[rng_np.integers(0, len(cp_pool), size=n_need)]

        for arr, out in ((nef, n2o_out), (cef, ch4_out)):
            ok2 = ~np.isnan(arr)
            if ok2.any():
                ww = w[ok2] / w[ok2].sum()
                out[k] = float(np.sum(ww * arr[ok2]))
            else:
                out[k] = np.nan

    return n2o_out, ch4_out


OUT_COLS = ['Country', 'Holdout', 'N_plants', 'Pop_total', 'N_true_types', 'Gas',
            'True_EF', 'Pred_mean', 'Pred_median', 'Pred_std', 'Pred_p2.5', 'Pred_p97.5',
            'PI_width', 'PI_width_rel_%', 'RelDiff_%', 'In_PI']


def summarize_samples(samples, true_val):
    """Mean, median, std, 95% interval and coverage for one sample set."""
    s = samples[~np.isnan(samples)]
    if len(s) == 0:
        return {}
    mean_v = float(np.mean(s))
    p25, p975 = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
    return {
        'True_EF': round(true_val, 6) if true_val is not None else np.nan,
        'Pred_mean': round(mean_v, 6),
        'Pred_median': round(float(np.median(s)), 6),
        'Pred_std': round(float(np.std(s)), 6),
        'Pred_p2.5': round(p25, 6),
        'Pred_p97.5': round(p975, 6),
        'PI_width': round(p975 - p25, 6),
        'PI_width_rel_%': round((p975 - p25) / mean_v * 100, 1) if mean_v else np.nan,
        'RelDiff_%': round((mean_v - true_val) / true_val * 100, 2) if true_val else np.nan,
        'In_PI': int(p25 <= true_val <= p975) if true_val is not None else np.nan,
    }


def make_out_path(countries):
    """Build the output workbook path for this batch of countries."""
    if len(countries) == 1:
        stem = f'country_level_PI_{safe_name(countries[0])}_{TAG}'
    elif len(countries) <= 4:
        stem = f'country_level_PI_{"-".join(safe_name(c) for c in countries)}_{TAG}'
    else:
        stem = f'country_level_PI_overseas_{TAG}'
    return os.path.join(OUT_DIR, f'{stem}.xlsx')


def stage_b(countries):
    """Run the country-level Monte Carlo and write the prediction intervals."""
    np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

    n2o_by_type, ch4_by_type, fb_by_type, gmr = load_ef_pools_source()
    print('\nbuilding reference value maps...')
    n2o_map, ch4_map = build_true_maps(n2o_by_type, ch4_by_type, fb_by_type, gmr)

    print('\nreading stored probabilities...')
    proba_df, cats = load_proba_for(countries)

    print('\nbuilding EF pools...')
    n2o_pools = build_ef_pools(cats, 'N2O', n2o_by_type, fb_by_type, gmr, RANDOM_SEED)
    ch4_pools = build_ef_pools(cats, 'CH4', ch4_by_type, fb_by_type, gmr, RANDOM_SEED + 1)

    rows = []
    got = sorted(proba_df[P_COUNTRY_COL].unique())
    combos = (proba_df[[P_COUNTRY_COL, 'holdout_pct']]
              .drop_duplicates()
              .sort_values([P_COUNTRY_COL, 'holdout_pct'])
              .itertuples(index=False, name=None))
    combos = list(combos)
    print(f'\ncountry-level MC (N_SIM={N_SIM_COUNTRY}) over {len(combos)} combinations')

    for c, hp in combos:
        sub = proba_df[(proba_df[P_COUNTRY_COL] == c) & (proba_df['holdout_pct'] == hp)]
        if len(sub) == 0:
            continue

        pop = pd.to_numeric(sub[P_POP_COL], errors='coerce').fillna(1.0).values
        w = pop / pop.sum() if pop.sum() > 0 else np.ones(len(pop)) / len(pop)

        def true_region(mp):
            """Population-weighted reference EF for the current subset."""
            v = sub[P_TYPE_COL].map(mp).astype(float).values
            ok = ~np.isnan(v)
            if not ok.any():
                return None
            ww = w[ok] / w[ok].sum()
            return float(np.sum(ww * v[ok]))

        n2o_true = true_region(n2o_map)
        ch4_true = true_region(ch4_map)

        seed = RANDOM_SEED + abs(hash((c, str(hp)))) % 10000
        n2o_s, ch4_s = country_mc(sub, cats, n2o_pools, ch4_pools, N_SIM_COUNTRY, seed)

        base = {
            'Country': c,
            'Holdout': hp,
            'N_plants': len(sub),
            'Pop_total': float(pop.sum()),
            'N_true_types': int(sub[P_TYPE_COL].nunique()),
        }
        for gas, s, tv in [('N2O', n2o_s, n2o_true), ('CH4', ch4_s, ch4_true)]:
            rows.append({**base, 'Gas': gas, **summarize_samples(s, tv)})

        print(f'  {c}[{hp}]: n={len(sub)}, N2O {n2o_s.mean():.4f} '
              f'[{np.percentile(n2o_s,2.5):.4f}, {np.percentile(n2o_s,97.5):.4f}]')

    res = pd.DataFrame(rows)[OUT_COLS]

    summ = []
    for (hp, gas), gd in res.groupby(['Holdout', 'Gas']):
        summ.append({
            'Holdout': hp,
            'Gas': gas,
            'N_countries': len(gd),
            'RelDiff_mean_%': round(gd['RelDiff_%'].mean(), 2),
            'RelDiff_std_%': round(gd['RelDiff_%'].std(ddof=1), 2),
            'RelDiff_min_%': round(gd['RelDiff_%'].min(), 2),
            'RelDiff_max_%': round(gd['RelDiff_%'].max(), 2),
            'MedAbs_RelDiff_%': round(gd['RelDiff_%'].abs().median(), 2),
            'PI_width_rel_mean_%': round(gd['PI_width_rel_%'].mean(), 1),
            'PI_coverage_%': round(gd['In_PI'].mean() * 100, 1),
            'N_covered': f"{int(gd['In_PI'].sum())}/{len(gd)}",
        })
    summary_df = pd.DataFrame(summ)

    out_xlsx = make_out_path(got)
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w_:
        res.to_excel(w_, sheet_name='Country_Level_PI', index=False)
        summary_df.to_excel(w_, sheet_name='Summary', index=False)
    print(f'\nsaved: {out_xlsx}')

    show = ['Country', 'Holdout', 'N_plants', 'N_true_types', 'True_EF', 'Pred_mean',
            'Pred_p2.5', 'Pred_p97.5', 'PI_width_rel_%', 'RelDiff_%', 'In_PI']
    for gas in ['N2O', 'CH4']:
        gd = res[res['Gas'] == gas].sort_values(['Country', 'Holdout'])
        if len(gd) == 0:
            continue
        print(f'\n{"="*104}\n{gas}  country level, one row per country and holdout\n{"="*104}')
        print(gd[show].to_string(index=False))

    print(f'\n{"="*104}\nsummary across countries\n{"="*104}')
    print(summary_df.to_string(index=False))

    small = res[(res['Gas'] == 'N2O') & (res['N_plants'] < MIN_PLANTS_WARN)]
    if len(small):
        print(f'\n! countries with fewer than {MIN_PLANTS_WARN} plants, weighted EF driven by few plants:')
        for _, r in small.iterrows():
            print(f'   {r["Country"]}[{r["Holdout"]}]: n={r["N_plants"]}')

    print('\nSampling matches the main pipeline: one type per plant, one EF per plant, '
          'aggregated by population weight.')
    return res


def main():
    """Run stage A and stage B end to end."""
    t0 = time.time()
    print(f"stage A mode: STAGE_A_MODE={STAGE_A_MODE!r}  (RUN_STAGE_A={RUN_STAGE_A}, "
          f"RUN_STAGE_B={RUN_STAGE_B}, RUN_PLANT_LEVEL_MC={RUN_PLANT_LEVEL_MC})")
    if not RUN_STAGE_A:
        print("  skipping training, running the Monte Carlo on probabilities already in %s" % OUT_DIR)
    print("countries held out:", LEAVE_OUT_COUNTRIES)

    if RUN_STAGE_A:
        df = pd.read_excel(INPUT_FILE)
        df = df.dropna(subset=[LABEL_COL, COUNTRY_COL]).copy()

        have = set(df[COUNTRY_COL].unique())
        missing = [c for c in LEAVE_OUT_COUNTRIES if c not in have]
        if missing:
            print(f"\n! these names are not present in {COUNTRY_COL}, check the spelling: {missing}")
            for m in missing:
                key = m.split(',')[0].split()[0].lower()
                cand = [h for h in have if key in str(h).lower()]
                if cand:
                    print(f'   closest matches for "{m}": {cand}')

        feats = [c for c in FEATURE_COLS if c in df.columns]
        check_stage_a_manifest(feats)

        ef = load_ef_data() if RUN_PLANT_LEVEL_MC else (None, None, None, None, None)
        progress = load_progress()

        jobs = build_holdout_jobs(df)
        print("\njobs in this batch (country x holdout):")
        for c, fr, pct in jobs:
            print(f"   {c}: {'full leave-country-out' if pct is None else f'{pct}% fold'}")

        all_metrics = []
        for country, frac, pct in jobs:
            plant_df = run_one_country(country, frac, pct, df, feats, ef, progress)
            if plant_df is not None and len(plant_df):
                m = metrics(plant_df, country)
                m['holdout_pct'] = 'LCO' if pct is None else pct
                all_metrics.append(m)
                print(f"\n-- {country} ({'LCO' if pct is None else str(pct)+'%'}) plant-level metrics --")
                for k, v in m.items(): print(f"   {k}: {v}")

        if all_metrics:
            res = pd.DataFrame(all_metrics)
            out_xlsx = os.path.join(OUT_DIR, f'leave_country_out_metrics_{TAG}.xlsx')
            res.to_excel(out_xlsx, index=False)
            print(f"\nplant-level summary saved: {out_xlsx}")
            print(res.to_string(index=False))

    if RUN_STAGE_B:
        print(f"\n\n{'#'*70}\nStage B: country-level EF prediction intervals\n{'#'*70}")
        stage_b(LEAVE_OUT_COUNTRIES)

    h, rem = divmod(int(time.time()-t0), 3600); m_, s = divmod(rem, 60)
    print(f"\ntotal time {h}h{m_}m{s}s")


if __name__ == '__main__':
    method = 'fork' if os.name != 'nt' else 'spawn'
    try: multiprocessing.set_start_method(method, force=True)
    except RuntimeError: pass
    main()