"""Bootstrap classification, Monte Carlo emission simulation, and multi-level 95% PI."""

import pandas as pd
import numpy as np
import os
import gc
import glob
import pickle
import time
import platform
import warnings
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from xgboost import XGBClassifier

from paths import inp, out

warnings.filterwarnings('ignore')

N_SIMULATIONS = 10000
B = 1000
POOL_SIZE = 20000
RANDOM_SEED = None

AVAILABLE_CORES = int(os.environ.get('NCPUS', 0)) or os.cpu_count()
if AVAILABLE_CORES >= 16:
    N_JOBS_BOOTSTRAP, N_JOBS_XGB = 8, 2
elif AVAILABLE_CORES >= 8:
    N_JOBS_BOOTSTRAP, N_JOBS_XGB = 4, 2
else:
    N_JOBS_BOOTSTRAP, N_JOBS_XGB = 2, 1
print(f"{AVAILABLE_CORES} cores | bootstrap={N_JOBS_BOOTSTRAP}, xgb threads={N_JOBS_XGB}")

IQR_K = 1.5
MIN_GROUP_N = 4
FB_NORM_FLOOR = 0.10
FB_SAMPLING_FLOOR = None

GWP_CH4 = 27
GWP_N2O = 273

CITY_SUBSAMPLE = 1
CITY_PCTL_CHUNK = 2000

# --- input ---------------------------------------------------------------
COMBINED_WWTP_FILE = inp('combined_wwtp.xlsx')
CITY_DATA_FILE = inp('citytreat.xlsx')
# to use the city table produced by AD_Sewercode.py, which already carries the
# 'Anaerobic_digestion_ratio' column, point CITY_DATA_FILE at that output instead:
# CITY_DATA_FILE = out('AD_sewer', 'city_database_with_anaerobic_sewer.xlsx')
N2O_EF_FILE = inp('N2O_EF.xlsx')
CH4_EF_FILE = inp('CH4_EF.xlsx')
FB_FILE = inp('FB.xlsx')

N2O_EF_COL = 'N2OEF'
CH4_EF_COL = 'CH4EF'
FB_COL = 'FB'

# --- output --------------------------------------------------------------
MATRIX_OUTPUT_FILE = out('wwtp_full_proba_matrix_main.xlsx')
CITY_MAPPING_OUTPUT = out('city_mapping_v2.pkl')
PROBA_OUTPUT_FILE = out('predicted_proba.csv')
BUCKET_PROBA_PREFIX = out('proba')

CITY_POPULATION_COL = 'Population'

REGION_COL = 'Region'

LABEL_COL = 'Biotreat Type'
ID_COL = 'serial number'
POP_COL = 'Current Served Population (estimate)'

TREATMENT_METHODS = [
    'A2O', 'AO', 'Anaerobic', 'BAF', 'BNR', 'CAS', 'EA', 'Lagoon',
    'MBR', 'MLE', 'OD', 'Pond', 'RBC', 'SBR', 'TF', 'Wetland'
]

KEEP_COLS = ['serial number', 'NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4',
             'Region', POP_COL, LABEL_COL]

ADVANCED_TYPES = {'AO', 'BAF', 'BNR', 'EA', 'MBR', 'MLE', 'A2O', 'SBR', 'OD'}

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

CH4_DEFAULT_VALUES = {'Pond': 0.102, 'Lagoon': 0.102, 'Anaerobic': 0.408}
CH4_DEFAULT_UNCERTAINTY = 0.30
CH4_MEAN_UNCERTAINTY_TYPES = {}
N2O_DEFAULT_VALUES = {'Anaerobic': 0.0}

CH4_LEAK_MIN = 0.0
CH4_LEAK_MAX = 0.1


def draw_ch4_leak_rate():
    """Draw one CH4_AD leak rate, shared by all cities in the iteration."""
    return float(np.random.uniform(CH4_LEAK_MIN, CH4_LEAK_MAX))


CH4_GEN_MEAN = 0.55
CH4_GEN_UNCERTAINTY = 0.30


def draw_ch4_gen_rate():
    """Draw one CH4 generation rate, shared by all cities in the iteration."""
    val = np.random.normal(CH4_GEN_MEAN, CH4_GEN_MEAN * CH4_GEN_UNCERTAINTY)
    return float(np.clip(val, 0.0, 1.0))

SLUDGE_RATIO = {
    'A2O': 0.6219171728, 'AO': 0.6219171728, 'BAF': 0.6219171728,
    'BNR': 0.6219171728, 'MBR': 0.6219171728, 'MLE': 0.6219171728,
    'OD': 0.6219171728, 'SBR': 0.6219171728, 'CAS': 0.6219171728,
    'EA': 0.499485,
    'TF': 0.591157644, 'RBC': 0.591157644,
    'Lagoon': 0.14058, 'Pond': 0.14058, 'Wetland': 0.14058,
    'Anaerobic': 0.12141,
}

CH4_REDUCTION = {
    'A2O': 0.9, 'AO': 0.9, 'BAF': 0.9, 'BNR': 0.9, 'EA': 0.9,
    'MBR': 0.9, 'MLE': 0.9, 'OD': 0.9, 'SBR': 0.9,
    'CAS': 0.85, 'TF': 0.85, 'RBC': 0.85,
    'Lagoon': 0.85, 'Pond': 0.85, 'Wetland': 0.85, 'Anaerobic': 0.85,
}
N2O_REDUCTION = {
    'A2O': 0.8, 'AO': 0.8, 'BAF': 0.8, 'BNR': 0.8, 'EA': 0.8,
    'MBR': 0.8, 'MLE': 0.8, 'OD': 0.8, 'SBR': 0.8,
    'CAS': 0.4, 'TF': 0.4, 'RBC': 0.4,
    'Lagoon': 0.4, 'Pond': 0.4, 'Wetland': 0.4, 'Anaerobic': 0.4,
}

CITY_COLS = dict(
    pop='Population', bod_value='BOD_PerCapita', protein='Protein_PerCapita',
    f_noncon='F-NONCON', n_hh='N-HH', protein_consumed='Protein consumed',
    pct_sewer='% Generated via Sewers', pct_septic='% Generated via Septic Tanks',
    pct_sewer_treated='% Sewer WW Safely Treated',
    pct_septic_treated='% Septic Tank WW Safely Treated',
    ad_ratio='Anaerobic_digestion_ratio',
)


_COUNTRY_TABLE = """
Afghanistan|Asia|Low income
Akrotiri and Dhekelia|Europe|High income
Åland|Europe|High income
Albania|Europe|Upper middle income
Algeria|Africa|Upper middle income
American Samoa|Oceania|High income
Andorra|Europe|High income
Angola|Africa|Lower middle income
Anguilla|North America|High income
Antigua and Barbuda|North America|High income
Argentina|South America|Upper middle income
Armenia|Asia|Lower middle income
Aruba|North America|High income
Australia|Oceania|High income
Austria|Europe|High income
Azerbaijan|Asia|Upper middle income
Bahamas|North America|High income
Bahrain|Asia|High income
Bangladesh|Asia|Lower middle income
Barbados|North America|High income
Belarus|Europe|Upper middle income
Belgium|Europe|High income
Belize|North America|Upper middle income
Benin|Africa|Lower middle income
Bermuda|North America|High income
Bhutan|Asia|High income
Bolivia|South America|Lower middle income
Bonaire, Sint Eustatius and Saba|North America|High income
Bosnia and Herzegovina|Europe|Upper middle income
Botswana|Africa|Upper middle income
Brazil|South America|Upper middle income
British Virgin Islands|North America|High income
Brunei|Asia|High income
Bulgaria|Europe|High income
Burkina Faso|Africa|Low income
Burundi|Africa|Low income
Cambodia|Asia|Lower middle income
Cameroon|Africa|Lower middle income
Canada|North America|High income
Cape Verde|Africa|Upper middle income
Cayman Islands|North America|High income
Central African Republic|Africa|Low income
Chad|Africa|Low income
Chile|South America|High income
China|Asia|Upper middle income
Christmas Island|Oceania|High income
Cocos Islands|Oceania|High income
Colombia|South America|Upper middle income
Comoros|Africa|Lower middle income
Cook Islands|Oceania|High income
Costa Rica|North America|High income
Côte d'Ivoire|Africa|Lower middle income
Croatia|Europe|High income
Cuba|North America|Upper middle income
Curaçao|North America|High income
Cyprus|Europe|High income
Czech Republic|Europe|High income
Democratic Republic of the Congo|Africa|Low income
Denmark|Europe|High income
Djibouti|Africa|Lower middle income
Dominica|North America|High income
Dominican Republic|North America|Upper middle income
Ecuador|South America|Upper middle income
Egypt|Africa|Lower middle income
El Salvador|North America|Lower middle income
Equatorial Guinea|Africa|High income
Eritrea|Africa|Low income
Estonia|Europe|High income
Ethiopia|Africa|Low income
Falkland Islands|South America|High income
Faroe Islands|Europe|High income
Fiji|Oceania|Upper middle income
Finland|Europe|High income
France|Europe|High income
French Guiana|South America|High income
French Polynesia|Oceania|High income
Gabon|Africa|Upper middle income
Gambia|Africa|Lower middle income
Georgia|Asia|Upper middle income
Germany|Europe|High income
Ghana|Africa|Lower middle income
Gibraltar|Europe|High income
Greece|Europe|High income
Greenland|Europe|High income
Grenada|North America|Upper middle income
Guadeloupe|North America|High income
Guam|Oceania|High income
Guatemala|North America|Lower middle income
Guernsey|Europe|High income
Guinea|Africa|Low income
Guinea-Bissau|Africa|Low income
Guyana|South America|Upper middle income
Haiti|North America|Low income
Honduras|North America|Lower middle income
Hong Kong|Asia|High income
Hungary|Europe|High income
Iceland|Europe|High income
India|Asia|Lower middle income
Indonesia|Asia|Upper middle income
Iran|Asia|Upper middle income
Iraq|Asia|Upper middle income
Ireland|Europe|High income
Isle of Man|Europe|High income
Israel|Asia|High income
Italy|Europe|High income
Jamaica|North America|Upper middle income
Japan|Asia|High income
Jersey|Europe|High income
Jordan|Asia|Upper middle income
Kazakhstan|Asia|Upper middle income
Kenya|Africa|Lower middle income
Kiribati|Oceania|Lower middle income
Kosovo|Europe|Upper middle income
Kuwait|Asia|High income
Kyrgyzstan|Asia|Lower middle income
Laos|Asia|Lower middle income
Latvia|Europe|High income
Lebanon|Asia|Upper middle income
Lesotho|Africa|Lower middle income
Liberia|Africa|Low income
Libya|Africa|Upper middle income
Liechtenstein|Europe|High income
Lithuania|Europe|High income
Luxembourg|Europe|High income
Macao|Asia|High income
Macedonia|Europe|Upper middle income
Madagascar|Africa|Low income
Malawi|Africa|Low income
Malaysia|Asia|Upper middle income
Maldives|Asia|Upper middle income
Mali|Africa|Low income
Malta|Europe|High income
Marshall Islands|Oceania|Upper middle income
Martinique|North America|High income
Mauritania|Africa|Lower middle income
Mauritius|Africa|Upper middle income
Mayotte|Africa|High income
Mexico|North America|Upper middle income
Micronesia|Oceania|Lower middle income
Moldova|Europe|Lower middle income
Monaco|Europe|High income
Mongolia|Asia|Lower middle income
Montenegro|Europe|Upper middle income
Montserrat|North America|High income
Morocco|Africa|Lower middle income
Mozambique|Africa|Low income
Myanmar|Asia|Lower middle income
Namibia|Africa|Upper middle income
Nauru|Oceania|High income
Nepal|Asia|Lower middle income
Netherlands|Europe|High income
New Caledonia|Oceania|High income
New Zealand|Oceania|High income
Nicaragua|North America|Lower middle income
Niger|Africa|Low income
Nigeria|Africa|Lower middle income
Niue|Oceania|High income
Norfolk Island|Oceania|High income
North Korea|Asia|Upper middle income
Northern Cyprus|Europe|Upper middle income
Northern Mariana Islands|Oceania|High income
Norway|Europe|High income
Oman|Asia|High income
Pakistan|Asia|Lower middle income
Palau|Oceania|High income
Palestina|Asia|Upper middle income
Panama|North America|High income
Papua New Guinea|Oceania|Lower middle income
Paracel Islands|Asia|Low income
Paraguay|South America|Upper middle income
Peru|South America|Upper middle income
Philippines|Asia|Lower middle income
Pitcairn Islands|Oceania|High income
Poland|Europe|High income
Portugal|Europe|High income
Puerto Rico|North America|High income
Qatar|Asia|High income
Republic of Congo|Africa|Lower middle income
Reunion|Africa|High income
Romania|Europe|Upper middle income
Russia|Europe|High income
Rwanda|Africa|Lower middle income
Saint Helena|Africa|High income
Saint Kitts and Nevis|North America|High income
Saint Lucia|North America|Upper middle income
Saint Pierre and Miquelon|North America|High income
Saint Vincent and the Grenadines|North America|Upper middle income
Saint-Barthélemy|North America|High income
Saint-Martin|North America|High income
Samoa|Oceania|Upper middle income
San Marino|Europe|High income
São Tomé and Príncipe|Africa|Lower middle income
Saudi Arabia|Asia|High income
Senegal|Africa|Lower middle income
Serbia|Europe|Upper middle income
Seychelles|Africa|High income
Sierra Leone|Africa|Low income
Singapore|Asia|High income
Sint Maarten|North America|High income
Slovakia|Europe|High income
Slovenia|Europe|High income
Solomon Islands|Oceania|Lower middle income
Somalia|Africa|Low income
South Africa|Africa|Upper middle income
South Korea|Asia|High income
South Sudan|Africa|Low income
Spain|Europe|High income
Sri Lanka|Asia|Lower middle income
Sudan|Africa|Low income
Suriname|South America|Upper middle income
Svalbard and Jan Mayen|Europe|High income
Swaziland|Africa|Lower middle income
Sweden|Europe|High income
Switzerland|Europe|High income
Syria|Asia|Low income
Taiwan|Asia|High income
Tajikistan|Asia|Lower middle income
Tanzania|Africa|Lower middle income
Thailand|Asia|Upper middle income
Timor-Leste|Asia|Lower middle income
Togo|Africa|Low income
Tokelau|Oceania|High income
Tonga|Oceania|Upper middle income
Trinidad and Tobago|North America|High income
Tunisia|Africa|Lower middle income
Turkey|Asia|Upper middle income
Turkmenistan|Asia|Upper middle income
Turks and Caicos Islands|North America|High income
Tuvalu|Oceania|High income
Uganda|Africa|Low income
Ukraine|Europe|Lower middle income
United Arab Emirates|Asia|High income
United Kingdom|Europe|High income
United States|North America|High income
Uruguay|South America|High income
Uzbekistan|Asia|Lower middle income
Vanuatu|Oceania|Lower middle income
Vatican City|Europe|High income
Venezuela|South America|Upper middle income
Vietnam|Asia|Lower middle income
Virgin Islands, U.S.|North America|High income
Wallis and Futuna|Oceania|High income
Western Sahara|Africa|Low income
Yemen|Asia|Low income
Zambia|Africa|Lower middle income
Zimbabwe|Africa|Lower middle income
""".strip()


def build_country_lookup():
    """Map each country name to its continent and income level."""
    lookup = {}
    for line in _COUNTRY_TABLE.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) != 3:
            continue
        country, continent, income = parts
        lookup[country.strip()] = {'Continent': continent.strip(), 'Income': income.strip()}
    return lookup


COUNTRY_LOOKUP = build_country_lookup()

def resolve_continent_income(country_names):
    """Map country names to continent and income level."""
    continents, incomes = [], []
    unmatched = set()
    for name in country_names:
        info = COUNTRY_LOOKUP.get(name)
        if info is None:
            continents.append('Unknown')
            incomes.append('Unknown')
            unmatched.add(name)
        else:
            continents.append(info['Continent'])
            incomes.append(info['Income'])
    if unmatched:
        print(f"  ! {len(unmatched)} NAME_0 values have no continent/income mapping, "
              f"set to 'Unknown' (affects continent and income levels only):")
        print(f"    {sorted(unmatched)}")
    return np.array(continents), np.array(incomes)


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
    base = {'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1}
    bucket = bucket.lower()
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


def bootstrap_worker(args):
    """Fit one bootstrap replicate and return class probabilities."""
    b_idx, X_train_rec, y_enc, X_pred_rec, unique_cats, bucket_params, treatment_levels = args
    try:
        X_train = pd.DataFrame(X_train_rec)
        X_pred = pd.DataFrame(X_pred_rec)
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
            min_child_weight=bucket_params['min_child_weight'],
            reg_lambda=bucket_params['reg_lambda'],
            reg_alpha=bucket_params['reg_alpha'],
            n_jobs=N_JOBS_XGB, random_state=b_idx + 42,
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
                        proba[i] = proba[i] / s if s > 0 else (
                            np.where(allowed, 1.0 / np.sum(allowed), 0.0))
        return b_idx, proba
    except Exception as e:
        print(f"  bootstrap {b_idx} failed: {e}")
        return b_idx, np.zeros((len(X_pred_rec), len(unique_cats)))


def run_bootstrap(X_train, y_enc, X_pred, unique_cats, bucket_params, treatment_levels):
    """Run B bootstrap replicates and return (B, n_pred, n_classes)."""
    X_tr_rec = X_train.to_dict('records')
    X_pr_rec = X_pred.to_dict('records')
    args_list = [(b, X_tr_rec, y_enc, X_pr_rec, unique_cats, bucket_params, treatment_levels)
                 for b in range(B)]
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
                    if completed % 50 == 0:
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
        d[f'{cat}_mean'] = np.mean(cp, axis=0)
        d[f'{cat}_std'] = np.std(cp, axis=0)
        d[f'{cat}_ci_lower'] = np.percentile(cp, 2.5, axis=0)
        d[f'{cat}_ci_upper'] = np.percentile(cp, 97.5, axis=0)
        d[f'{cat}_median'] = np.median(cp, axis=0)

    means = np.column_stack([d[f'{c}_mean'] for c in unique_cats])
    pred_idx = np.argmax(means, axis=1)
    d['predicted_biotreat'] = [unique_cats[i] for i in pred_idx]
    d['confidence'] = np.max(means, axis=1)
    eps = 1e-10
    safe = np.clip(means, eps, 1 - eps)
    d['prediction_entropy'] = -np.sum(safe * np.log(safe), axis=1)
    return pd.DataFrame(d)


def predict_unknown_proba(base_df):
    """Predict treatment-type probabilities for unlabelled plants, bucket by bucket."""
    print("=" * 70)
    print("Part 0: bootstrap classification of unlabelled plants")
    print("=" * 70)

    available_features = [c for c in FEATURE_COLS if c in base_df.columns]
    missing = [c for c in FEATURE_COLS if c not in base_df.columns]
    print(f"\nfeatures available: {len(available_features)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"  ! missing features: {missing}")

    train_df = base_df.dropna(subset=[LABEL_COL]).copy()
    pred_df = base_df[base_df[LABEL_COL].isna()].copy()
    train_df['population_bucket'] = train_df[POP_COL].apply(get_population_bucket)
    pred_df['population_bucket'] = pred_df[POP_COL].apply(get_population_bucket)
    print(f"  labelled (train): {len(train_df)} | unlabelled (predict): {len(pred_df)}")

    all_unique_cats = sorted(train_df[LABEL_COL].unique())
    print(f"  training classes ({len(all_unique_cats)}): {all_unique_cats}")

    for stale in glob.glob(f"{BUCKET_PROBA_PREFIX}_*.csv"):
        os.remove(stale)

    out = []
    for bucket in sorted(train_df['population_bucket'].unique()):
        tr_b = train_df[train_df['population_bucket'] == bucket]
        pred_b = pred_df[pred_df['population_bucket'] == bucket]
        print(f"\n  {'-' * 60}")
        print(f"  [{bucket}]  train {len(tr_b)} | predict {len(pred_b)}  "
              f"start {datetime.now().strftime('%H:%M:%S')}")

        if len(tr_b) < 10:
            print("    too few training samples, skipped")
            continue
        if len(pred_b) == 0:
            print("    no prediction samples, skipped")
            continue

        t0 = time.time()
        X_train = tr_b[available_features].copy()
        X_pred = pred_b[available_features].copy()

        for col in X_train.select_dtypes(include=np.number).columns:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_pred[col] = X_pred[col].fillna(med)

        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            all_cats = pd.Categorical(
                pd.concat([X_train[col], X_pred[col]], ignore_index=True)).categories
            X_train[col] = pd.Categorical(X_train[col], categories=all_cats).codes
            X_pred[col] = pd.Categorical(X_pred[col], categories=all_cats).codes

        cat_to_idx = {c: i for i, c in enumerate(all_unique_cats)}
        y_enc = np.array([cat_to_idx[c] for c in tr_b[LABEL_COL].values])

        treatment_levels = (pred_b['Treatment Level'].tolist()
                            if 'Treatment Level' in pred_b.columns else None)

        print(f"    bootstrap ({B} replicates)...")
        boot_results = run_bootstrap(X_train, y_enc, X_pred, all_unique_cats,
                                     get_bucket_params(bucket), treatment_levels)

        stats_df = compute_proba_stats(boot_results, all_unique_cats)
        stats_df.insert(0, ID_COL, pred_b[ID_COL].values)
        stats_df.insert(1, 'population_bucket', bucket)
        stats_df.insert(2, 'served_population', pred_b[POP_COL].values)
        if 'Treatment Level' in pred_b.columns:
            stats_df['treatment_level'] = pred_b['Treatment Level'].values
        for col in ['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4', 'Region']:
            if col in pred_b.columns:
                stats_df[col] = pred_b[col].values

        bucket_file = f"{BUCKET_PROBA_PREFIX}_{bucket.replace(': ', '_').replace('-', '_')}.csv"
        stats_df.to_csv(bucket_file, index=False, encoding='utf-8')

        out.append(stats_df)
        print(f"    done, {len(stats_df)} rows -> {bucket_file}"
              f" ({(time.time() - t0) / 60:.1f} min)")
        gc.collect()

    if not out:
        raise RuntimeError("no bucket produced predicted probabilities")

    proba_df = pd.concat(out, ignore_index=True)
    proba_df.to_csv(PROBA_OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\npredicted probabilities saved: {PROBA_OUTPUT_FILE} ({len(proba_df)} rows, for inspection only)")
    return proba_df


def build_wwtp_proba_matrix(base_df, proba_df):
    """Assemble the full probability matrix for every plant."""
    print("=" * 70)
    print("Part 1: assemble the full plant probability matrix")
    print("=" * 70)

    n_total = len(base_df)
    print(f"  {n_total} rows (this row order is the reference for everything downstream)")

    missing_keep = [c for c in KEEP_COLS if c not in base_df.columns]
    if missing_keep:
        raise KeyError(f"combined_wwtp.xlsx is missing required columns: {missing_keep}")

    if base_df[ID_COL].duplicated().sum() > 0:
        raise RuntimeError(f"combined_wwtp.xlsx has duplicate '{ID_COL}' values")

    result = base_df[KEEP_COLS].copy()
    result.index = base_df.index

    is_known = result[LABEL_COL].notna().values
    print(f"\nknown type: {is_known.sum()} | unknown type: {(~is_known).sum()}")

    for cat in TREATMENT_METHODS:
        result[f'{cat}_mean'] = 0.0
        result[f'{cat}_std'] = 0.0

    cat_to_idx = {c: i for i, c in enumerate(TREATMENT_METHODS)}
    label_values = result[LABEL_COL].values
    known_cat_idx = np.zeros(n_total, dtype=np.int32)

    for cat in TREATMENT_METHODS:
        mask = is_known & (label_values == cat)
        result.loc[mask, f'{cat}_mean'] = 1.0
        known_cat_idx[mask.values if hasattr(mask, 'values') else mask] = cat_to_idx[cat]

    known_mapped = np.isin(label_values[is_known], TREATMENT_METHODS)
    unmapped_label = int((~known_mapped).sum())
    if unmapped_label:
        bad_mask = is_known & (~pd.Series(label_values).isin(TREATMENT_METHODS).values)
        is_known = is_known & (~bad_mask)
        print(f"  ! {unmapped_label} labels are outside the 16 treatment types, handled as unknown")

    print(f"\npredicted probabilities from Part 0: {len(proba_df)} rows")

    if proba_df[ID_COL].duplicated().sum() > 0:
        raise RuntimeError(f"predicted probability table has duplicate '{ID_COL}' values")

    proba_cols = [f'{c}{s}' for c in TREATMENT_METHODS for s in ['_mean', '_std']]
    missing_proba_cols = set(proba_cols) - set(proba_df.columns)
    if missing_proba_cols:
        raise KeyError(f"predicted probability table is missing columns: {missing_proba_cols}")

    proba_lookup = proba_df.set_index(ID_COL)[proba_cols]

    unknown_idx = np.where(~is_known)[0]
    unknown_serials = result.iloc[unknown_idx][ID_COL].values
    matched_mask = np.isin(unknown_serials, proba_lookup.index)
    n_matched = int(matched_mask.sum())

    aligned = proba_lookup.reindex(unknown_serials)
    result.iloc[unknown_idx, [result.columns.get_loc(c) for c in proba_cols]] = \
        aligned[proba_cols].fillna(0.0).values

    print(f"  {n_matched}/{len(unknown_idx)} unknown-type plants matched to a prediction")
    if n_matched < len(unknown_idx):
        print(f"  ! {len(unknown_idx) - n_matched} unmatched rows get mean=0 for every class "
              f"and contribute zero emissions; check Part 0")

    assert len(result) == n_total
    assert (result[ID_COL].values == base_df[ID_COL].values).all()

    result.to_excel(MATRIX_OUTPUT_FILE, index=False)
    print(f"\nprobability matrix saved: {MATRIX_OUTPUT_FILE} ({len(result)} rows, for inspection only)")

    cat_mean_matrix = result[[f'{c}_mean' for c in TREATMENT_METHODS]].values.astype(np.float64)
    cat_std_matrix = result[[f'{c}_std' for c in TREATMENT_METHODS]].values.astype(np.float64)

    return result, is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix


def build_admin_tree(city_data):
    """Build the administrative hierarchy from the city table."""
    print("\nbuilding the administrative tree...")
    admin_tree = {}
    for _, row in city_data.iterrows():
        name0 = row['NAME_0']
        if pd.isna(name0) or name0 == 'nan':
            continue
        if name0 not in admin_tree:
            admin_tree[name0] = {'info': {'level': 0, 'population': 0}, 'children': {}}
        if CITY_POPULATION_COL in row and pd.notna(row[CITY_POPULATION_COL]):
            try:
                admin_tree[name0]['info']['population'] += float(row[CITY_POPULATION_COL])
            except (ValueError, TypeError):
                pass
        current_node = admin_tree[name0]
        for level in range(1, 5):
            col_name = f'NAME_{level}'
            if col_name in row and pd.notna(row[col_name]) and row[col_name] != 'nan':
                region_name = row[col_name]
                if region_name not in current_node['children']:
                    current_node['children'][region_name] = {
                        'info': {'level': level, 'population': 0, 'parent': current_node},
                        'children': {}
                    }
                if CITY_POPULATION_COL in row and pd.notna(row[CITY_POPULATION_COL]):
                    try:
                        current_node['children'][region_name]['info']['population'] += float(row[CITY_POPULATION_COL])
                    except (ValueError, TypeError):
                        pass
                current_node = current_node['children'][region_name]
    print(f"done, {len(admin_tree)} countries")
    return admin_tree


def aggregate_wwtp_by_region(wwtp_data):
    """Group plants by administrative region."""
    print("\nstep 1: aggregating plants by region...")
    region_groups = wwtp_data.groupby(['NAME_0', 'Region'])
    region_aggregated = {}
    for (name0, region), group in region_groups:
        if pd.isna(region) or region == 'nan':
            continue
        wwtp_indices = group.index.tolist()
        served_populations = group[POP_COL].tolist()
        region_aggregated[(name0, region)] = {
            'NAME_0': name0, 'Region': region,
            'wwtp_indices': wwtp_indices,
            'served_populations': served_populations,
            'Total_Served_Population': sum(served_populations),
            'WWTP_Count': len(group)
        }
    print(f"done, {len(region_aggregated)} regions")
    return region_aggregated


def mark_tree_with_aggregated_data(admin_tree, region_aggregated_data):
    """Attach regional plant aggregates to the administrative tree."""
    print("\nstep 2: marking tree nodes with regional aggregates...")
    matched_count = 0
    unmatched_count = 0
    for (name0, region), data in region_aggregated_data.items():
        found = False
        if name0 not in admin_tree:
            unmatched_count += 1
            continue
        if region == name0:
            admin_tree[name0]['info']['wwtp_indices'] = data['wwtp_indices']
            admin_tree[name0]['info']['served_populations'] = data['served_populations']
            admin_tree[name0]['info']['has_ef'] = True
            matched_count += 1
            continue
        stack = [(admin_tree[name0], [])]
        while stack and not found:
            node, path = stack.pop()
            for child_name, child_node in node['children'].items():
                if child_name == region:
                    child_node['info']['wwtp_indices'] = data['wwtp_indices']
                    child_node['info']['served_populations'] = data['served_populations']
                    child_node['info']['has_ef'] = True
                    found = True
                    matched_count += 1
                    break
                stack.append((child_node, path + [child_name]))
        if not found:
            unmatched_count += 1

    total = matched_count + unmatched_count
    if total:
        print(f"done: matched {matched_count} ({matched_count/total*100:.2f}%), "
              f"unmatched {unmatched_count} ({unmatched_count/total*100:.2f}%)")

    for name0, country_node in admin_tree.items():
        if 'has_ef' not in country_node['info']:
            country_node['info']['has_ef'] = False
            country_node['info']['wwtp_indices'] = []
            country_node['info']['served_populations'] = []
        stack = [country_node]
        while stack:
            node = stack.pop()
            if 'has_ef' not in node['info']:
                node['info']['has_ef'] = False
                node['info']['wwtp_indices'] = []
                node['info']['served_populations'] = []
            for child in node['children'].values():
                stack.append(child)
    return admin_tree


def apply_propagation_rules(admin_tree):
    """Propagate aggregates down the tree to nodes without plants."""
    print("\napplying EF propagation rules...")
    for level in range(3, 0, -1):
        for name0, country_node in admin_tree.items():
            parents_to_process = []
            stack = [(country_node, [])]
            while stack:
                node, path = stack.pop()
                if node['info']['level'] == level - 1:
                    parents_to_process.append((node, path))
                for child_name, child_node in node['children'].items():
                    stack.append((child_node, path + [child_name]))

            for parent_node, parent_path in parents_to_process:
                parent_has_ef = parent_node['info']['has_ef']
                parent_children_with_ef = []
                parent_children_without_ef = []
                for child_name, child_node in parent_node['children'].items():
                    if child_node['info']['level'] == level:
                        if child_node['info']['has_ef']:
                            parent_children_with_ef.append((child_name, child_node))
                        else:
                            parent_children_without_ef.append((child_name, child_node))

                if parent_has_ef and not parent_children_with_ef:
                    for child_name, child_node in parent_children_without_ef:
                        child_node['info']['has_ef'] = True
                        child_node['info']['wwtp_indices'] = parent_node['info']['wwtp_indices'].copy()
                        child_node['info']['served_populations'] = parent_node['info']['served_populations'].copy()
                        child_node['info']['ef_source'] = f"inherited from NAME_{level-1}"
                elif not parent_has_ef and parent_children_with_ef:
                    if parent_children_without_ef:
                        all_indices, all_pops = [], []
                        for child_name, child_node in parent_children_with_ef:
                            all_indices.extend(child_node['info']['wwtp_indices'])
                            all_pops.extend(child_node['info']['served_populations'])
                        if sum(all_pops) > 0:
                            for child_name, child_node in parent_children_without_ef:
                                child_node['info']['has_ef'] = True
                                child_node['info']['wwtp_indices'] = all_indices.copy()
                                child_node['info']['served_populations'] = all_pops.copy()
                                child_node['info']['ef_source'] = f"weighted from {len(parent_children_with_ef)} sibling regions"
                elif parent_has_ef and parent_children_with_ef:
                    parent_served_pop = sum(parent_node['info']['served_populations'])
                    children_served_pop = sum(sum(c['info']['served_populations']) for _, c in parent_children_with_ef)
                    if parent_served_pop > children_served_pop:
                        for child_name, child_node in parent_children_without_ef:
                            child_node['info']['has_ef'] = True
                            child_node['info']['wwtp_indices'] = parent_node['info']['wwtp_indices'].copy()
                            child_node['info']['served_populations'] = parent_node['info']['served_populations'].copy()
                            child_node['info']['ef_source'] = "inherited from parent (parent has larger served population)"
                    else:
                        if parent_children_without_ef:
                            all_indices, all_pops = [], []
                            for child_name, child_node in parent_children_with_ef:
                                all_indices.extend(child_node['info']['wwtp_indices'])
                                all_pops.extend(child_node['info']['served_populations'])
                            if sum(all_pops) > 0:
                                for child_name, child_node in parent_children_without_ef:
                                    child_node['info']['has_ef'] = True
                                    child_node['info']['wwtp_indices'] = all_indices.copy()
                                    child_node['info']['served_populations'] = all_pops.copy()
                                    child_node['info']['ef_source'] = f"weighted from {len(parent_children_with_ef)} sibling regions (child has larger served population)"

    for name0, country_node in admin_tree.items():
        country_pop = country_node['info']['population']
        is_small_country = country_pop < 10000000
        if is_small_country and country_node['info']['has_ef']:
            for province_name, province_node in country_node['children'].items():
                if province_node['info']['level'] == 1 and not province_node['info']['has_ef']:
                    province_node['info']['has_ef'] = True
                    province_node['info']['wwtp_indices'] = country_node['info']['wwtp_indices'].copy()
                    province_node['info']['served_populations'] = country_node['info']['served_populations'].copy()
                    province_node['info']['ef_source'] = "inherited from NAME_0"

    for name0, country_node in admin_tree.items():
        if country_node['info']['has_ef']:
            any_child_has_ef = False
            stack = [(country_node, [])]
            while stack and not any_child_has_ef:
                node, path = stack.pop()
                if path and node['info']['has_ef']:
                    any_child_has_ef = True
                    break
                for child_name, child_node in node['children'].items():
                    stack.append((child_node, path + [child_name]))
            if not any_child_has_ef:
                stack = [(country_node, [])]
                while stack:
                    node, path = stack.pop()
                    if path:
                        node['info']['has_ef'] = True
                        node['info']['wwtp_indices'] = country_node['info']['wwtp_indices'].copy()
                        node['info']['served_populations'] = country_node['info']['served_populations'].copy()
                        node['info']['ef_source'] = "inherited from NAME_0"
                    for child_name, child_node in node['children'].items():
                        stack.append((child_node, path + [child_name]))

    print("propagation rules applied")
    return admin_tree


def build_city_mapping(city_data, admin_tree):
    """Map each city to its plants with population weights."""
    print("\nbuilding the city to plant mapping...")
    city_mapping = {}
    cities_with_ef = 0
    cities_without_ef = 0

    for city_idx, city_row in city_data.iterrows():
        name0 = city_row['NAME_0']
        if pd.isna(name0) or name0 == 'nan' or name0 not in admin_tree:
            city_mapping[city_idx] = {'wwtp_indices': [], 'weights': [], 'ef_source': ''}
            cities_without_ef += 1
            continue

        current_node = admin_tree[name0]
        matched_node = None
        max_matched_level = 0

        for level in range(1, 5):
            col_name = f'NAME_{level}'
            if col_name in city_row and pd.notna(city_row[col_name]) and city_row[col_name] != 'nan':
                region_name = city_row[col_name]
                if region_name in current_node['children']:
                    current_node = current_node['children'][region_name]
                    if current_node['info']['has_ef']:
                        max_matched_level = level
                        matched_node = current_node
                else:
                    break

        found_ef = False
        if matched_node is not None:
            found_ef = True
            total_pop = sum(matched_node['info']['served_populations'])
            if total_pop > 0:
                weights = [p / total_pop for p in matched_node['info']['served_populations']]
            else:
                weights = [1.0 / len(matched_node['info']['wwtp_indices'])] * len(matched_node['info']['wwtp_indices'])
            ef_source = matched_node['info'].get('ef_source', f"direct match at NAME_{max_matched_level}")
            city_mapping[city_idx] = {
                'wwtp_indices': matched_node['info']['wwtp_indices'].copy(),
                'weights': weights, 'ef_source': ef_source, 'matched_level': max_matched_level
            }
            cities_with_ef += 1
        elif admin_tree[name0]['info']['has_ef']:
            country_node = admin_tree[name0]
            country_pop = country_node['info']['population']
            is_small_country = country_pop < 10000000
            only_country_data = 'country level only' in str(country_node['info'].get('ef_source', ''))
            if is_small_country or only_country_data:
                found_ef = True
                total_pop = sum(country_node['info']['served_populations'])
                if total_pop > 0:
                    weights = [p / total_pop for p in country_node['info']['served_populations']]
                else:
                    weights = [1.0 / len(country_node['info']['wwtp_indices'])] * len(country_node['info']['wwtp_indices'])
                city_mapping[city_idx] = {
                    'wwtp_indices': country_node['info']['wwtp_indices'].copy(),
                    'weights': weights, 'ef_source': 'matched at NAME_0 only', 'matched_level': 0
                }
                cities_with_ef += 1

        if not found_ef:
            city_mapping[city_idx] = {'wwtp_indices': [], 'weights': [], 'ef_source': '', 'matched_level': max_matched_level}
            cities_without_ef += 1

    print(f"\nmapping done: {len(city_data)} cities, "
          f"with EF {cities_with_ef} ({cities_with_ef/len(city_data)*100:.1f}%), "
          f"without EF {cities_without_ef} ({cities_without_ef/len(city_data)*100:.1f}%)")
    return city_mapping


def build_wwtp_city_mapping(wwtp_data):
    """Build and cache the city to plant mapping."""
    print("\n" + "=" * 70)
    print("Part 2: build the city to plant mapping")
    print("=" * 70)

    print(f"\nloading city data: {CITY_DATA_FILE}")
    city_data = pd.read_excel(CITY_DATA_FILE)
    for i in range(5):
        col = f'NAME_{i}'
        if col in city_data.columns:
            city_data[col] = city_data[col].astype(str)
    if 'Region' in city_data.columns:
        city_data['Region'] = city_data['Region'].astype(str)
    print(f"  {len(city_data)} rows")

    wwtp_data = wwtp_data.copy()
    for i in range(5):
        col = f'NAME_{i}'
        if col in wwtp_data.columns:
            wwtp_data[col] = wwtp_data[col].astype(str)
    if 'Region' in wwtp_data.columns:
        wwtp_data['Region'] = wwtp_data['Region'].astype(str)
    wwtp_data[POP_COL] = wwtp_data[POP_COL].fillna(0)

    admin_tree = build_admin_tree(city_data)
    region_aggregated_data = aggregate_wwtp_by_region(wwtp_data)
    admin_tree = mark_tree_with_aggregated_data(admin_tree, region_aggregated_data)
    admin_tree = apply_propagation_rules(admin_tree)
    city_mapping = build_city_mapping(city_data, admin_tree)

    with open(CITY_MAPPING_OUTPUT, 'wb') as f:
        pickle.dump(city_mapping, f)
    print(f"\ncity mapping saved: {CITY_MAPPING_OUTPUT} (for inspection only)")

    return city_mapping, city_data


def _clip_floor(vals, floor=FB_NORM_FLOOR):
    """Drop NaN and clip to a lower bound."""
    a = np.asarray(list(vals), dtype=np.float64)
    a = a[~np.isnan(a)]
    if floor is not None:
        a = np.clip(a, floor, None)
    return a


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
        rec = {'Type': t, 'N_eligible': n, 'Mean_before': vals.mean()}

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
                        'Lower': np.nan, 'Upper': np.nan, 'N_removed': 0,
                        'Pct_removed': 0.0, 'N_kept': n, 'Mean_after': vals.mean()})
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
              f" {n_out / max(n_input, 1):.2%})")
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


def load_ef_raw_data():
    """Load and IQR-clean the N2O and CH4 emission factor tables."""
    print("\nloading and cleaning the raw EF tables...")

    fb_df = pd.read_excel(FB_FILE)
    fb_by_type = {}
    for _, row in fb_df.iterrows():
        fb_by_type.setdefault(row['Type'], []).append(float(row[FB_COL]))

    fb_mean_by_type = {}
    for t, v in fb_by_type.items():
        a = _clip_floor(v)
        fb_mean_by_type[t] = float(a.mean()) if len(a) else np.nan
    all_flat = [v for vv in fb_by_type.values() for v in vv]
    gmr = float(_clip_floor(all_flat).mean()) if all_flat else 1.0
    print(f"  FB pool: {len(fb_mean_by_type)} Types, global fallback mean = {gmr:.4g}"
          f" (floor {FB_NORM_FLOOR})")

    n_low = int((np.asarray(all_flat, dtype=np.float64) < (FB_NORM_FLOOR or 0)).sum())
    if n_low:
        print(f"  ! {n_low} FB values below {FB_NORM_FLOOR}. The normalising mean is clipped,")
        print("    but build_resolved_pools divides by a randomly drawn FB value, so a draw")
        print("    near zero inflates the EF. See FB_SAMPLING_FLOOR.")

    if FB_SAMPLING_FLOOR is not None:
        fb_by_type = {t: [max(float(v), FB_SAMPLING_FLOOR) for v in vv]
                      for t, vv in fb_by_type.items()}
        print(f"  sampling FB values clipped to {FB_SAMPLING_FLOOR}")

    print("\n  --- N2O ---")
    n2o_df = pd.read_excel(N2O_EF_FILE)
    if 'F/B' not in n2o_df.columns:
        print("  ! no 'F/B' column, every record treated as non-F")
        n2o_df['F/B'] = np.nan

    is_F = n2o_df['F/B'].astype(str).str.strip().eq('F')
    r = n2o_df['Type'].map(lambda t: fb_mean_by_type.get(t, gmr))
    r = r.replace(0, np.nan).fillna(gmr)
    n2o_df['_screen'] = np.where(is_F, n2o_df[N2O_EF_COL], n2o_df[N2O_EF_COL] / r)

    mixed = [t for t, g in n2o_df.groupby('Type')
             if is_F.loc[g.index].any() and (~is_F.loc[g.index]).any()]
    print(f"  F/non-F split: F={int(is_F.sum())}, non-F={int((~is_F).sum())} | "
          f"{len(mixed)} mixed Types {sorted(mixed)}")

    n2o_df, _ = iqr_clean_by_type(n2o_df, '_screen', label=f'{N2O_EF_COL} (normalised scale)')
    n2o_df = n2o_df.drop(columns=['_screen'])
    n2o_raw = n2o_df.to_dict('records')

    print("\n  --- CH4 ---")
    ch4_df = pd.read_excel(CH4_EF_FILE)
    ch4_df, _ = iqr_clean_by_type(ch4_df, CH4_EF_COL, label=CH4_EF_COL)
    ch4_raw = ch4_df.to_dict('records')

    print(f"\n  N2O EF: {len(n2o_raw)} records | CH4 EF: {len(ch4_raw)} records")
    return n2o_raw, ch4_raw, fb_by_type


def build_resolved_pools(n2o_raw, ch4_raw, fb_by_type, categories, pool_size=POOL_SIZE):
    """Pre-resolve one sampling pool of emission factors per treatment type."""
    print("\npre-resolving the EF sampling pool for each treatment type...")
    n2o_pools, ch4_pools = {}, {}
    all_ratios_flat = [v for vv in fb_by_type.values() for v in vv]

    for cat in categories:
        if cat in N2O_DEFAULT_VALUES:
            n2o_pools[cat] = np.full(pool_size, N2O_DEFAULT_VALUES[cat], dtype=np.float64)
        else:
            recs = [r for r in n2o_raw if r['Type'] == cat]
            if recs:
                vals = np.empty(pool_size, dtype=np.float64)
                idxs = np.random.randint(0, len(recs), size=pool_size)
                ratio_pool = fb_by_type.get(cat)
                for i, ridx in enumerate(idxs):
                    rec = recs[ridx]
                    ef = rec[N2O_EF_COL] / 100
                    if rec.get('F/B') != 'F':
                        if ratio_pool:
                            ef = ef / ratio_pool[np.random.randint(len(ratio_pool))]
                        else:
                            ef = ef / (np.mean(all_ratios_flat) if all_ratios_flat else 1.0)
                    vals[i] = ef
                n2o_pools[cat] = vals
            else:
                n2o_pools[cat] = None
                print(f"  ! N2O: no raw EF records for type '{cat}'")

        if cat in CH4_DEFAULT_VALUES:
            default = CH4_DEFAULT_VALUES[cat]
            std = default * CH4_DEFAULT_UNCERTAINTY
            ch4_pools[cat] = np.abs(np.random.normal(default, std, pool_size))
        elif cat in CH4_MEAN_UNCERTAINTY_TYPES:
            recs = [r for r in ch4_raw if r['Type'] == cat]
            if recs:
                base = float(np.mean([r[CH4_EF_COL] for r in recs]))
                std = base * CH4_DEFAULT_UNCERTAINTY
                ch4_pools[cat] = np.abs(np.random.normal(base, std, pool_size))
                print(f"  CH4: type '{cat}' uses base value {base:.4g} (mean of CH4_EF.xlsx records)"
                      f" with an added +/-{CH4_DEFAULT_UNCERTAINTY:.0%} normal uncertainty")
            else:
                ch4_pools[cat] = None
                print(f"  ! CH4: type '{cat}' has no record in CH4_EF.xlsx, no base value available")
        else:
            recs = [r for r in ch4_raw if r['Type'] == cat]
            if recs:
                idxs = np.random.randint(0, len(recs), size=pool_size)
                ch4_pools[cat] = np.array([recs[i][CH4_EF_COL] for i in idxs], dtype=np.float64)
            else:
                ch4_pools[cat] = None
                print(f"  ! CH4: no raw EF records for type '{cat}'")

    return n2o_pools, ch4_pools


def diagnose_ch4_zeroing(wwtp_df, ch4_pools, n2o_pools, ch4_raw, is_known, known_cat_idx, categories):
    """Report which CH4 pools are empty and the population share they affect."""
    print("\n" + "=" * 70)
    print("Diagnostics: pool status and the share of CH4 forced to zero")
    print("=" * 70)
    ch4_none, n2o_none = [], []
    print(f"{'Type':<10}{'CH4 pool':>12}{'N2O pool':>12}   CH4 source")
    for cat in categories:
        cp = ch4_pools.get(cat)
        npool = n2o_pools.get(cat)
        cpm = float(np.mean(cp)) if cp is not None else float('nan')
        npm = float(np.mean(npool)) if npool is not None else float('nan')
        if cat in CH4_DEFAULT_VALUES:
            src = 'fixed default'
        elif cp is None:
            src = 'None, CH4 forced to zero'
            ch4_none.append(cat)
        else:
            src = 'bootstrap records'
        if npool is None:
            n2o_none.append(cat)
        print(f"{cat:<10}{cpm:>12.4g}{npm:>12.4g}   {src}")

    if n2o_none:
        print(f"\n  types with a None N2O pool, N2O forced to zero: {n2o_none}")

    if not ch4_none:
        print("\n  no None CH4 pool, so low CH4 is not caused by forced zeros")
        return

    cat_idx = {c: i for i, c in enumerate(categories)}
    none_ids = [cat_idx[c] for c in ch4_none]
    mean_mat = wwtp_df[[f'{c}_mean' for c in categories]].values.astype(float)
    pop = wwtp_df[POP_COL].fillna(0).values.astype(float)

    exp_none = np.zeros(len(wwtp_df))
    unk = ~is_known
    rs = mean_mat[unk].sum(axis=1, keepdims=True)
    rs[rs <= 0] = 1.0
    exp_none[unk] = mean_mat[unk][:, none_ids].sum(axis=1) / rs[:, 0]
    exp_none[is_known] = np.isin(known_cat_idx[is_known], none_ids).astype(float)

    share = np.sum(exp_none * pop) / np.sum(pop) if pop.sum() > 0 else float('nan')
    print(f"\n  ! types with CH4 forced to zero: {ch4_none}")
    print(f"  weighted by served population, about {share:.1%} of the CH4 type mass falls on them")
    print("  and is always zero, which is the direct cause of systematically low CH4")
    print("  (if these types should have an EF, give them a fallback base value instead of None)")


def sample_beta_prob_batch(mean_arr, std_arr):
    """Draw class probabilities from Betas matched to mean and std."""
    mean_arr = np.asarray(mean_arr, dtype=np.float64)
    std_arr = np.asarray(std_arr, dtype=np.float64)
    out = np.zeros_like(mean_arr)

    valid = (~np.isnan(mean_arr)) & (mean_arr > 0)
    no_std = valid & ((np.isnan(std_arr)) | (std_arr <= 0))
    out[no_std] = np.clip(mean_arr[no_std], 0, 1)

    has_std = valid & (~no_std)
    if np.any(has_std):
        m = np.clip(mean_arr[has_std], 0.001, 0.999)
        s = std_arr[has_std]
        var = np.minimum(s ** 2, m * (1 - m) * 0.99)
        cf = np.where(var > 0, m * (1 - m) / np.maximum(var, 1e-12) - 1, -1)
        ok = cf > 0
        result = m.copy()
        if np.any(ok):
            a = m[ok] * cf[ok]
            b = (1 - m[ok]) * cf[ok]
            result[ok] = np.random.beta(a, b)
        out[has_std] = result

    return out


def draw_categories_for_iteration(is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix, n_cat):
    """Draw one treatment type per plant for this iteration."""
    n = len(is_known)
    chosen = np.empty(n, dtype=np.int32)
    chosen[is_known] = known_cat_idx[is_known]

    unk_mask = ~is_known
    if np.any(unk_mask):
        means = cat_mean_matrix[unk_mask]
        stds = cat_std_matrix[unk_mask]
        sampled_probs = np.zeros_like(means)
        for ci in range(n_cat):
            sampled_probs[:, ci] = sample_beta_prob_batch(means[:, ci], stds[:, ci])

        row_sum = sampled_probs.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        norm_probs = sampled_probs / row_sum

        cum_probs = np.cumsum(norm_probs, axis=1)
        u = np.random.uniform(size=(norm_probs.shape[0], 1))
        picked = (u < cum_probs).argmax(axis=1)
        picked[cum_probs[:, -1] <= 0] = -1
        chosen[unk_mask] = picked

    return chosen


def prepare_city_activity_arrays(city_data):
    """Read city activity columns and derive total BOD and TN loads."""
    c = CITY_COLS
    arrs = {}
    print("\nchecking city activity data for missing values...")
    for key, col in c.items():
        if col not in city_data.columns:
            raise KeyError(f"city data is missing column '{col}' (field {key})")
        raw = city_data[col].values.astype(np.float64)
        n_nan = np.isnan(raw).sum()
        if n_nan > 0:
            print(f"  ! column '{col}' has {n_nan} NaN, filled with 0"
                  f" (otherwise those cities produce NaN every iteration and contaminate all levels)")
            raw = np.nan_to_num(raw, nan=0.0)
        arrs[key] = raw

    K = arrs['pop']
    BOD_all = K * arrs['bod_value'] * 365 * 1e-9
    TN_all = (K * arrs['protein'] * 0.16 * arrs['f_noncon'] * arrs['n_hh']
              * 44 / 28 * 365 * 1e-9 * arrs['protein_consumed'])
    arrs['BOD_all'] = BOD_all
    arrs['TN_all'] = TN_all
    return arrs


def compute_secondary_loads(city_arrs):
    """Deterministic BOD and TN loads entering secondary treatment."""
    U = city_arrs['pct_sewer'] / 100
    V = city_arrs['pct_septic'] / 100
    AA = city_arrs['pct_sewer_treated'] / 100
    AB = city_arrs['pct_septic_treated'] / 100

    BOD_all = city_arrs['BOD_all']
    TN_all = city_arrs['TN_all']

    BOD_intoSecondary = BOD_all * (1.25 * U * AA * 0.811333 + V * 0.375 * AB)
    TN_intoSecondary = TN_all * (U * AA * 1.25 + V * 0.85 * AB)
    return BOD_intoSecondary, TN_intoSecondary


def compute_city_emissions(city_arrs, city_ch4_ef, city_n2o_ef, city_sludge_ratio, city_ad_ratio,
                           ch4_gen_rate=0.55, ch4_leak_rate=0.05):
    """Compute the five emission quantities for every city."""
    BOD_intoSecondary, TN_intoSecondary = compute_secondary_loads(city_arrs)

    CH4_Safetreated = BOD_intoSecondary * city_ch4_ef
    CH4_AD = BOD_intoSecondary * city_sludge_ratio * city_ad_ratio * ch4_gen_rate * ch4_leak_rate
    CH4_inPlant = CH4_Safetreated + CH4_AD
    N2O_Safetreated = TN_intoSecondary * city_n2o_ef
    CO2e_total = CH4_inPlant * GWP_CH4 + N2O_Safetreated * GWP_N2O

    return {
        'CH4_Safetreated': CH4_Safetreated,
        'CH4_AD': CH4_AD,
        'CH4_inPlant': CH4_inPlant,
        'N2O_Safetreated': N2O_Safetreated,
        'CO2e_total': CO2e_total,
    }


TARGET_KEYS = ['CH4_Safetreated', 'CH4_AD', 'CH4_inPlant', 'N2O_Safetreated', 'CO2e_total']

EF_DEFS = [
    ('CH4_EF_safetreated', 'CH4_Safetreated', 'BOD'),
    ('CH4_EF_inPlant',     'CH4_inPlant',     'BOD'),
    ('N2O_EF',             'N2O_Safetreated', 'TN'),
]


def _agg_det(vec, idx, n):
    """Sum a deterministic vector by group."""
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, idx, vec)
    return out


def run_unified_simulation(is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix,
                            city_mapping, city_data, n2o_pools, ch4_pools,
                            categories, n_simulations=N_SIMULATIONS):
    """Run the Monte Carlo and return distributions for every aggregation level."""
    print("\n" + "=" * 70)
    print(f"Part 3: single-layer Monte Carlo simulation (n={n_simulations})")
    print("targets: CH4_Safetreated / CH4_AD / CH4_inPlant / N2O_Safetreated / CO2e_total")
    print(f"CO2e: CH4 x {GWP_CH4} + N2O x {GWP_N2O} (AR6 GWP-100, biogenic CH4)")
    print("=" * 70)

    n_wwtps = len(is_known)
    n_cities = len(city_data)
    n_cat = len(categories)

    type_sludge_ratio = np.array([SLUDGE_RATIO[c] for c in categories])
    type_ch4_reduction = np.array([CH4_REDUCTION[c] for c in categories])
    type_n2o_reduction = np.array([N2O_REDUCTION[c] for c in categories])

    city_idx_flat, wwtp_idx_flat, weight_flat = [], [], []
    for city_idx, m in city_mapping.items():
        for w_idx, wt in zip(m['wwtp_indices'], m['weights']):
            if 0 <= w_idx < n_wwtps:
                city_idx_flat.append(city_idx)
                wwtp_idx_flat.append(w_idx)
                weight_flat.append(wt)
    city_idx_flat = np.array(city_idx_flat, dtype=np.int64)
    wwtp_idx_flat = np.array(wwtp_idx_flat, dtype=np.int64)
    weight_flat = np.array(weight_flat, dtype=np.float64)

    has_class_info = is_known | (cat_mean_matrix.sum(axis=1) > 0)
    n_no_info = int((~has_class_info).sum())
    if n_no_info:
        keep = has_class_info[wwtp_idx_flat]
        dropped_w = np.zeros(n_cities, dtype=np.float64)
        np.add.at(dropped_w, city_idx_flat, weight_flat * (~keep))
        city_idx_flat = city_idx_flat[keep]
        wwtp_idx_flat = wwtp_idx_flat[keep]
        weight_flat = weight_flat[keep]
        kept_w = np.zeros(n_cities, dtype=np.float64)
        np.add.at(kept_w, city_idx_flat, weight_flat)
        weight_flat = weight_flat / np.where(kept_w[city_idx_flat] > 0,
                                             kept_w[city_idx_flat], 1.0)
        n_cities_hit = int((dropped_w > 0).sum())
        n_cities_empty = int(((dropped_w > 0) & (kept_w <= 0)).sum())
        print(f"\n  ! {n_no_info} plants carry no class information and were excluded from the "
              f"city weights; {n_cities_hit} cities affected, weights renormalised over the "
              f"remaining plants, {n_cities_empty} cities left with none")

    city_arrs = prepare_city_activity_arrays(city_data)

    BOD_secondary, TN_secondary = compute_secondary_loads(city_arrs)

    country_names = city_data['NAME_0'].astype(str).values
    unique_countries = sorted(set(country_names))
    country_to_idx = {c: i for i, c in enumerate(unique_countries)}
    city_country_idx = np.array([country_to_idx[c] for c in country_names])
    n_countries = len(unique_countries)

    continent_names, income_names = resolve_continent_income(country_names)
    unique_continents = sorted(set(continent_names))
    continent_to_idx = {c: i for i, c in enumerate(unique_continents)}
    city_continent_idx = np.array([continent_to_idx[c] for c in continent_names])
    n_continents = len(unique_continents)

    unique_incomes = sorted(set(income_names))
    income_to_idx = {c: i for i, c in enumerate(unique_incomes)}
    city_income_idx = np.array([income_to_idx[c] for c in income_names])
    n_incomes = len(unique_incomes)

    if REGION_COL not in city_data.columns:
        raise KeyError(f"{CITY_DATA_FILE} is missing the '{REGION_COL}' column, "
                       f"cannot group by region")
    region_raw = city_data[REGION_COL].astype(str).values
    region_keys = np.array([f"{c0} | {rg}" for c0, rg in zip(country_names, region_raw)])
    unique_regions = sorted(set(region_keys))
    region_to_idx = {r: i for i, r in enumerate(unique_regions)}
    city_region_idx = np.array([region_to_idx[r] for r in region_keys])
    n_regions = len(unique_regions)

    est_gb = n_simulations * n_regions * 4 * len(TARGET_KEYS) / (1024 ** 3)
    print(f"\n  regions: {n_regions} (distribution storage about {est_gb:.2f} GB, float32)")

    keep_sims = np.arange(0, n_simulations, max(1, int(CITY_SUBSAMPLE)))
    keep_pos = {int(s): i for i, s in enumerate(keep_sims)}
    n_keep = len(keep_sims)
    city_gb = n_keep * n_cities * 4 * 3 / (1024 ** 3)
    print(f"  city-level storage: {n_keep}/{n_simulations} iterations x {n_cities} cities x 3 arrays"
          f" ≈ {city_gb:.2f} GB (float32)")
    if CITY_SUBSAMPLE > 1:
        print(f"  ! CITY_SUBSAMPLE={CITY_SUBSAMPLE}, city percentiles use {n_keep} sampled points; "
              f"means still use all {n_simulations} iterations")

    city_ch4_ef_store = np.zeros((n_keep, n_cities), dtype=np.float32)
    city_n2o_ef_store = np.zeros((n_keep, n_cities), dtype=np.float32)
    city_sludge_store = np.zeros((n_keep, n_cities), dtype=np.float32)
    gen_store = np.zeros(n_keep, dtype=np.float64)
    leak_store = np.zeros(n_keep, dtype=np.float64)

    country_results_all = {k: np.zeros((n_simulations, n_countries), dtype=np.float64) for k in TARGET_KEYS}
    continent_results_all = {k: np.zeros((n_simulations, n_continents), dtype=np.float64) for k in TARGET_KEYS}
    income_results_all = {k: np.zeros((n_simulations, n_incomes), dtype=np.float64) for k in TARGET_KEYS}
    region_results_all = {k: np.zeros((n_simulations, n_regions), dtype=np.float32) for k in TARGET_KEYS}
    global_results_all = {k: np.zeros(n_simulations, dtype=np.float64) for k in TARGET_KEYS}

    city_sum = {k: np.zeros(n_cities, dtype=np.float64) for k in TARGET_KEYS}
    city_ch4_ef_sum = np.zeros(n_cities, dtype=np.float64)
    city_n2o_ef_sum = np.zeros(n_cities, dtype=np.float64)
    city_sludge_sum = np.zeros(n_cities, dtype=np.float64)
    city_ch4_red_sum = np.zeros(n_cities, dtype=np.float64)
    city_n2o_red_sum = np.zeros(n_cities, dtype=np.float64)
    city_has_mapping = np.zeros(n_cities, dtype=bool)
    city_has_mapping[city_idx_flat] = True

    start_time = time.time()

    for sim in range(n_simulations):
        chosen_cat_idx = draw_categories_for_iteration(
            is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix, n_cat
        )

        wwtp_ch4_ef = np.zeros(n_wwtps, dtype=np.float64)
        wwtp_n2o_ef = np.zeros(n_wwtps, dtype=np.float64)

        valid_cat = chosen_cat_idx >= 0
        for ci, cat in enumerate(categories):
            mask = chosen_cat_idx == ci
            n_needed = mask.sum()
            if n_needed == 0:
                continue
            pool = ch4_pools.get(cat)
            if pool is not None:
                idxs = np.random.randint(0, len(pool), size=n_needed)
                wwtp_ch4_ef[mask] = pool[idxs]
            npool = n2o_pools.get(cat)
            if npool is not None:
                idxs = np.random.randint(0, len(npool), size=n_needed)
                wwtp_n2o_ef[mask] = npool[idxs]

        safe_idx = np.where(valid_cat, chosen_cat_idx, 0)
        wwtp_sludge_ratio = type_sludge_ratio[safe_idx] * valid_cat
        wwtp_ch4_reduction = type_ch4_reduction[safe_idx] * valid_cat
        wwtp_n2o_reduction = type_n2o_reduction[safe_idx] * valid_cat

        city_ch4_ef = np.zeros(n_cities)
        city_n2o_ef = np.zeros(n_cities)
        city_sludge_ratio = np.zeros(n_cities)
        city_ch4_reduction = np.zeros(n_cities)
        city_n2o_reduction = np.zeros(n_cities)

        np.add.at(city_ch4_ef, city_idx_flat, wwtp_ch4_ef[wwtp_idx_flat] * weight_flat)
        np.add.at(city_n2o_ef, city_idx_flat, wwtp_n2o_ef[wwtp_idx_flat] * weight_flat)
        np.add.at(city_sludge_ratio, city_idx_flat, wwtp_sludge_ratio[wwtp_idx_flat] * weight_flat)
        np.add.at(city_ch4_reduction, city_idx_flat, wwtp_ch4_reduction[wwtp_idx_flat] * weight_flat)
        np.add.at(city_n2o_reduction, city_idx_flat, wwtp_n2o_reduction[wwtp_idx_flat] * weight_flat)

        ch4_gen_rate = draw_ch4_gen_rate()
        ch4_leak_rate = draw_ch4_leak_rate()

        city_results = compute_city_emissions(
            city_arrs, city_ch4_ef, city_n2o_ef, city_sludge_ratio, city_arrs['ad_ratio'],
            ch4_gen_rate=ch4_gen_rate, ch4_leak_rate=ch4_leak_rate
        )

        city_ch4_ef_sum += city_ch4_ef
        city_n2o_ef_sum += city_n2o_ef
        city_sludge_sum += city_sludge_ratio
        city_ch4_red_sum += city_ch4_reduction
        city_n2o_red_sum += city_n2o_reduction

        pos = keep_pos.get(sim)
        if pos is not None:
            city_ch4_ef_store[pos] = city_ch4_ef.astype(np.float32)
            city_n2o_ef_store[pos] = city_n2o_ef.astype(np.float32)
            city_sludge_store[pos] = city_sludge_ratio.astype(np.float32)
            gen_store[pos] = ch4_gen_rate
            leak_store[pos] = ch4_leak_rate

        for key in TARGET_KEYS:
            city_val = city_results[key]
            np.add.at(country_results_all[key][sim], city_country_idx, city_val)
            np.add.at(continent_results_all[key][sim], city_continent_idx, city_val)
            np.add.at(income_results_all[key][sim], city_income_idx, city_val)
            np.add.at(region_results_all[key][sim], city_region_idx, city_val.astype(np.float32))
            global_results_all[key][sim] = city_val.sum()
            city_sum[key] += city_val

        if (sim + 1) % max(1, n_simulations // 20) == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (sim + 1) * (n_simulations - sim - 1)
            print(f"  progress: {sim + 1}/{n_simulations} ({(sim+1)/n_simulations*100:.1f}%) | "
                  f"elapsed {elapsed:.1f}s | remaining {remaining:.1f}s")

    print(f"\nsimulation done in {(time.time()-start_time)/60:.1f} min")

    det_loads = {
        'country': (_agg_det(BOD_secondary, city_country_idx, n_countries),
                    _agg_det(TN_secondary, city_country_idx, n_countries)),
        'continent': (_agg_det(BOD_secondary, city_continent_idx, n_continents),
                      _agg_det(TN_secondary, city_continent_idx, n_continents)),
        'income': (_agg_det(BOD_secondary, city_income_idx, n_incomes),
                   _agg_det(TN_secondary, city_income_idx, n_incomes)),
        'region': (_agg_det(BOD_secondary, city_region_idx, n_regions),
                   _agg_det(TN_secondary, city_region_idx, n_regions)),
        'global': (float(BOD_secondary.sum()), float(TN_secondary.sum())),
    }

    return {
        'target_keys': TARGET_KEYS,
        'country':   {'results': country_results_all,   'names': unique_countries},
        'continent': {'results': continent_results_all, 'names': unique_continents},
        'income':    {'results': income_results_all,    'names': unique_incomes},
        'region':    {'results': region_results_all,    'names': unique_regions},
        'global':    {'results': global_results_all},
        'det_loads': det_loads,
        'city': {
            'data': city_data,
            'sum': city_sum,
            'ch4_ef_sum': city_ch4_ef_sum,
            'n2o_ef_sum': city_n2o_ef_sum,
            'sludge_sum': city_sludge_sum,
            'ch4_red_sum': city_ch4_red_sum,
            'n2o_red_sum': city_n2o_red_sum,
            'has_mapping': city_has_mapping,
            'ch4_ef_store': city_ch4_ef_store,
            'n2o_ef_store': city_n2o_ef_store,
            'sludge_store': city_sludge_store,
            'gen_store': gen_store,
            'leak_store': leak_store,
            'n_keep': n_keep,
            'ad_ratio': city_arrs['ad_ratio'],
            'bod_secondary': BOD_secondary,
            'tn_secondary': TN_secondary,
            'bod_all': city_arrs['BOD_all'],
            'tn_all': city_arrs['TN_all'],
            'pop': city_arrs['pop'],
            'region_key': region_keys,
            'n_sim': n_simulations,
        },
    }


def _stat(arr):
    """Return mean, p2.5, median, p97.5 and NaN count."""
    arr = np.asarray(arr, dtype=np.float64)
    if np.isnan(arr).any():
        return (np.nanmean(arr), np.nanpercentile(arr, 2.5), np.nanmedian(arr),
                np.nanpercentile(arr, 97.5), int(np.isnan(arr).sum()))
    return (np.mean(arr), np.percentile(arr, 2.5), np.median(arr),
            np.percentile(arr, 97.5), 0)


def _put_stat(row, label, arr):
    """Write the four statistics of one quantity into the output rows."""
    mean, p25, med, p975, n_nan = _stat(arr)
    row[f'{label}_Mean'] = mean
    row[f'{label}_P2.5'] = p25
    row[f'{label}_Median'] = med
    row[f'{label}_P97.5'] = p975
    if n_nan:
        row[f'{label}_NaN_Iterations'] = n_nan
    return row


def _put_ef(row, group_results_all, i, bod_i, tn_i):
    """Load-weighted effective EF, computed per iteration before taking statistics."""
    for label, key, den_kind in EF_DEFS:
        den = bod_i if den_kind == 'BOD' else tn_i
        if den is not None and np.isfinite(den) and den > 0:
            _put_stat(row, label, group_results_all[key][:, i] / den)
        else:
            for suf in ('Mean', 'P2.5', 'Median', 'P97.5'):
                row[f'{label}_{suf}'] = 0.0
    row['CH4_EF_Defined'] = int(bod_i is not None and np.isfinite(bod_i) and bod_i > 0)
    row['N2O_EF_Defined'] = int(tn_i is not None and np.isfinite(tn_i) and tn_i > 0)
    return row


def _build_group_df(group_results_all, unique_groups, target_keys, group_col_name,
                    bod_totals=None, tn_totals=None):
    """Build the statistics table for one aggregation level."""
    rows = []
    n_undef_bod = n_undef_tn = 0
    for i, group_name in enumerate(unique_groups):
        row = {group_col_name: group_name}
        for key in target_keys:
            _put_stat(row, key, group_results_all[key][:, i])

        bod_i = float(bod_totals[i]) if bod_totals is not None else None
        tn_i = float(tn_totals[i]) if tn_totals is not None else None
        _put_ef(row, group_results_all, i, bod_i, tn_i)
        n_undef_bod += (row['CH4_EF_Defined'] == 0)
        n_undef_tn += (row['N2O_EF_Defined'] == 0)

        if bod_i is not None:
            row['BOD_intoSecondary_Total'] = bod_i
        if tn_i is not None:
            row['TN_intoSecondary_Total'] = tn_i
        rows.append(row)
    df = pd.DataFrame(rows)

    for key in target_keys:
        nan_col = f'{key}_NaN_Iterations'
        if nan_col in df.columns:
            n_affected = df[nan_col].notna().sum()
            print(f"  ! {key}: {n_affected} {group_col_name} groups had NaN iterations, skipped by "
                  f"nan-aware statistics; check the '{nan_col}' column")
    if n_undef_bod or n_undef_tn:
        print(f"  {group_col_name}: {n_undef_bod} groups with zero BOD load, {n_undef_tn} with zero TN load; "
              f"their EF is undefined and written as 0, see the *_EF_Defined columns")
    return df


def _split_region_columns(region_df):
    """Split the combined region key back into two columns."""
    parts = region_df['Region_Key'].str.split(' | ', n=1, regex=False)
    region_df.insert(0, 'NAME_0', parts.str[0])
    region_df.insert(1, 'Region', parts.str[1])
    return region_df


def _city_percentiles(city):
    """Rebuild the city-level distribution in chunks and take percentiles."""
    bod = city['bod_secondary'].astype(np.float32)
    tn = city['tn_secondary'].astype(np.float32)
    ad = city['ad_ratio'].astype(np.float32)
    gen = city['gen_store'].astype(np.float32)[:, None]
    leak = city['leak_store'].astype(np.float32)[:, None]
    n_cities = len(bod)

    labels = ['CH4_EF_safetreated', 'CH4_EF_inPlant', 'N2O_EF',
              'CH4_Safetreated', 'CH4_AD', 'CH4_inPlant', 'N2O_Safetreated', 'CO2e_total']
    out = {lb: {s: np.zeros(n_cities, dtype=np.float64)
                for s in ('P2.5', 'Median', 'P97.5')} for lb in labels}

    step = max(1, int(CITY_PCTL_CHUNK))
    for a in range(0, n_cities, step):
        b = min(a + step, n_cities)
        cef = city['ch4_ef_store'][:, a:b]
        nef = city['n2o_ef_store'][:, a:b]
        sl = city['sludge_store'][:, a:b]
        bd = bod[a:b][None, :]
        tnn = tn[a:b][None, :]
        adr = ad[a:b][None, :]

        ch4_safe = bd * cef
        ch4_ad = bd * sl * adr * gen * leak
        ch4_inp = ch4_safe + ch4_ad
        n2o_e = tnn * nef
        co2e = ch4_inp * np.float32(GWP_CH4) + n2o_e * np.float32(GWP_N2O)
        ef_inp = np.divide(ch4_inp, bd, out=np.zeros_like(ch4_inp),
                           where=np.broadcast_to(bd > 0, ch4_inp.shape))

        vals = {'CH4_EF_safetreated': cef, 'CH4_EF_inPlant': ef_inp, 'N2O_EF': nef,
                'CH4_Safetreated': ch4_safe, 'CH4_AD': ch4_ad, 'CH4_inPlant': ch4_inp,
                'N2O_Safetreated': n2o_e, 'CO2e_total': co2e}
        for lb, v in vals.items():
            q = np.percentile(v.astype(np.float64), [2.5, 50, 97.5], axis=0)
            out[lb]['P2.5'][a:b] = q[0]
            out[lb]['Median'][a:b] = q[1]
            out[lb]['P97.5'][a:b] = q[2]

    return out


def summarize_and_save(R):
    """Summarise every level and write all output files."""
    print("\n" + "=" * 70)
    print("Part 4: compute and save statistics for every aggregation level")
    print("=" * 70)

    target_keys = R['target_keys']
    det = R['det_loads']

    country_df = _build_group_df(R['country']['results'], R['country']['names'], target_keys,
                                 'Country', det['country'][0], det['country'][1])
    continent_df = _build_group_df(R['continent']['results'], R['continent']['names'], target_keys,
                                   'Continent', det['continent'][0], det['continent'][1])
    income_df = _build_group_df(R['income']['results'], R['income']['names'], target_keys,
                                'Income_Level', det['income'][0], det['income'][1])
    region_df = _build_group_df(R['region']['results'], R['region']['names'], target_keys,
                                'Region_Key', det['region'][0], det['region'][1])
    region_df = _split_region_columns(region_df)

    global_row = {}
    for key in target_keys:
        arr = R['global']['results'][key]
        _put_stat(global_row, key, arr)
        if np.isnan(arr).any():
            print(f"  ! global {key}: {int(np.isnan(arr).sum())}/{len(arr)} iterations are NaN, "
                  f"skipped by nan-aware statistics")
    g_bod, g_tn = det['global'][0], det['global'][1]
    for label, key, den_kind in EF_DEFS:
        den = g_bod if den_kind == 'BOD' else g_tn
        if den > 0:
            _put_stat(global_row, label, R['global']['results'][key] / den)
        else:
            for suf in ('Mean', 'P2.5', 'Median', 'P97.5'):
                global_row[f'{label}_{suf}'] = 0.0
    global_row['CH4_EF_Defined'] = int(g_bod > 0)
    global_row['N2O_EF_Defined'] = int(g_tn > 0)
    global_row['BOD_intoSecondary_Total'] = g_bod
    global_row['TN_intoSecondary_Total'] = g_tn
    global_df = pd.DataFrame([global_row])

    print("\n  rebuilding the city-level distribution and taking percentiles...")
    t_city = time.time()
    city = R['city']
    n_sim = city['n_sim']
    pct = _city_percentiles(city)
    print(f"  city percentiles done in {time.time() - t_city:.1f}s")

    cd = city['data']
    city_cols = {}
    for col in ['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4', REGION_COL]:
        if col in cd.columns:
            city_cols[col] = cd[col].values
    city_out = pd.DataFrame(city_cols)
    city_out['Population'] = city['pop']
    city_out['BOD_all'] = city['bod_all']
    city_out['BOD_intoSecondary'] = city['bod_secondary']
    city_out['TN_all'] = city['tn_all']
    city_out['TN_intoSecondary'] = city['tn_secondary']

    for key in target_keys:
        city_out[f'{key}_Mean'] = city['sum'][key] / n_sim
        for suf in ('P2.5', 'Median', 'P97.5'):
            city_out[f'{key}_{suf}'] = pct[key][suf]

    bod_c = city['bod_secondary']
    ch4_ef_mean = city['ch4_ef_sum'] / n_sim
    n2o_ef_mean = city['n2o_ef_sum'] / n_sim
    inplant_mean = city['sum']['CH4_inPlant'] / n_sim
    ef_inplant_mean = np.divide(inplant_mean, bod_c,
                                out=np.zeros_like(inplant_mean), where=bod_c > 0)

    ef_means = {'CH4_EF_safetreated': ch4_ef_mean,
                'CH4_EF_inPlant': ef_inplant_mean,
                'N2O_EF': n2o_ef_mean}
    for label, _, _ in EF_DEFS:
        city_out[f'{label}_Mean'] = ef_means[label]
        for suf in ('P2.5', 'Median', 'P97.5'):
            city_out[f'{label}_{suf}'] = pct[label][suf]

    city_out['CH4_EF_Defined'] = (bod_c > 0).astype(int)
    city_out['N2O_EF_Defined'] = (city['tn_secondary'] > 0).astype(int)

    n_undef = int((bod_c <= 0).sum()), int((city['tn_secondary'] <= 0).sum())
    if n_undef[0] or n_undef[1]:
        print(f"  cities: {n_undef[0]} with zero BOD load, {n_undef[1]} with zero TN load; "
              f"their EF is undefined and written as 0, see the *_EF_Defined columns")

    country_file = out('country_emission_95PI.xlsx')
    continent_file = out('continent_emission_95PI.xlsx')
    income_file = out('income_level_emission_95PI.xlsx')
    region_file = out('region_emission_95PI.xlsx')
    global_file = out('global_emission_95PI.xlsx')
    city_file = out('city_emission_and_loads.csv')
    reduction_file = out('city_reduction_and_sludge.xlsx')
    country_reduction_file = out('country_reduction_and_sludge.xlsx')

    country_df.to_excel(country_file, index=False)
    continent_df.to_excel(continent_file, index=False)
    income_df.to_excel(income_file, index=False)
    region_df.to_excel(region_file, index=False)
    global_df.to_excel(global_file, index=False)
    city_out.to_csv(city_file, index=False, encoding='utf-8-sig')

    mapped = city['has_mapping']
    tr = {
        'ch4_reduction': np.where(mapped, city['ch4_red_sum'] / n_sim, 0.0),
        'n2o_reduction': np.where(mapped, city['n2o_red_sum'] / n_sim, 0.0),
        'sludge_ratio': np.where(mapped, city['sludge_sum'] / n_sim, 0.0),
    }
    red_cols = {}
    for lvl in ['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4']:
        if lvl in cd.columns:
            red_cols[lvl] = cd[lvl].values
    red_cols['CH4_Reduction_rate'] = tr['ch4_reduction']
    red_cols['N2O_Reduction_rate'] = tr['n2o_reduction']
    red_cols['BODintoSludge_Ratio'] = tr['sludge_ratio']
    reduction_out = pd.DataFrame(red_cols)
    reduction_out.to_excel(reduction_file, index=False)
    n_unmapped = int((~mapped).sum())

    bod_w = np.asarray(city['bod_secondary'], dtype=np.float64)
    tn_w = np.asarray(city['tn_secondary'], dtype=np.float64)
    country_key = cd['NAME_0'].astype(str).values
    agg = pd.DataFrame({
        'NAME_0': country_key,
        'bod': bod_w, 'tn': tn_w,
        'ch4_num': tr['ch4_reduction'] * bod_w,
        'n2o_num': tr['n2o_reduction'] * tn_w,
        'sludge_num': tr['sludge_ratio'] * bod_w,
    }).groupby('NAME_0', as_index=False).sum()

    country_red = pd.DataFrame({
        'NAME_0': agg['NAME_0'],
        'CH4_Reduction_rate': np.where(agg['bod'] > 0, agg['ch4_num'] / agg['bod'].replace(0, 1), 0.0),
        'N2O_Reduction_rate': np.where(agg['tn'] > 0, agg['n2o_num'] / agg['tn'].replace(0, 1), 0.0),
        'BODintoSludge_Ratio': np.where(agg['bod'] > 0, agg['sludge_num'] / agg['bod'].replace(0, 1), 0.0),
        'BOD_intoSecondary_Total': agg['bod'],
        'TN_intoSecondary_Total': agg['tn'],
    })
    country_red.to_excel(country_reduction_file, index=False)


    print(f"country 95% PI saved: {country_file}")
    print(f"continent 95% PI saved: {continent_file}")
    print(f"income level 95% PI saved: {income_file}")
    print(f"region 95% PI saved: {region_file} ({len(region_df)} regions)")
    print(f"global 95% PI saved: {global_file}")
    print(f"city detail saved: {city_file} ({len(city_out)} rows x {len(city_out.columns)} columns)")
    print(f"city reduction and sludge ratio saved: {reduction_file} ({len(reduction_out)} rows)")
    print(f"country reduction and sludge ratio saved: {country_reduction_file} "
          f"({len(country_red)} rows)")
    if n_unmapped:
        print(f"  {n_unmapped} cities matched no plant, all three columns written as 0")

    print("\nglobal results:")
    for key in target_keys:
        print(f"  {key}: {global_row[f'{key}_Mean']:.4g}  "
              f"[{global_row[f'{key}_P2.5']:.4g}, {global_row[f'{key}_P97.5']:.4g}]")
    print("\nglobal load-weighted effective EF:")
    for label, _, _ in EF_DEFS:
        print(f"  {label}: {global_row[f'{label}_Mean']:.6g}  "
              f"[{global_row[f'{label}_P2.5']:.6g}, {global_row[f'{label}_P97.5']:.6g}]")
    print(f"  BOD_intoSecondary total: {g_bod:.4f}")
    print(f"  TN_intoSecondary  total: {g_tn:.4f}")

    chk = global_row['CH4_EF_safetreated_Mean'] * g_bod
    ref = global_row['CH4_Safetreated_Mean']
    if ref != 0 and abs(chk - ref) / abs(ref) > 1e-9:
        print(f"  ! self-check failed: EF mean x load = {chk:.6g} does not match emission mean {ref:.6g}")
    else:
        print("  self-check passed: global EF mean x BOD load = CH4_Safetreated mean")

    return country_df, continent_df, income_df, region_df, global_df, city_out


def main():
    """Run the full pipeline end to end."""
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    t0 = time.time()

    print(f"\nloading the master table: {COMBINED_WWTP_FILE}")
    base_df = pd.read_excel(COMBINED_WWTP_FILE)
    print(f"  {len(base_df)} rows (this row order is the reference for everything downstream)")

    proba_df = predict_unknown_proba(base_df)

    wwtp_df, is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix = \
        build_wwtp_proba_matrix(base_df, proba_df)

    city_mapping, city_data = build_wwtp_city_mapping(wwtp_df)

    n2o_raw, ch4_raw, fb_by_type = load_ef_raw_data()
    n2o_pools, ch4_pools = build_resolved_pools(n2o_raw, ch4_raw, fb_by_type, TREATMENT_METHODS)

    diagnose_ch4_zeroing(wwtp_df, ch4_pools, n2o_pools, ch4_raw, is_known, known_cat_idx, TREATMENT_METHODS)

    R = run_unified_simulation(
        is_known, known_cat_idx, cat_mean_matrix, cat_std_matrix,
        city_mapping, city_data, n2o_pools, ch4_pools, TREATMENT_METHODS,
        n_simulations=N_SIMULATIONS
    )

    summarize_and_save(R)

    print(f"\nall done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    _method = 'fork' if platform.system() != 'Windows' else 'spawn'
    try:
        multiprocessing.set_start_method(_method, force=True)
    except RuntimeError:
        pass
    main()