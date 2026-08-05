"""Anaerobic digestion classification, sewer CH4 estimation, and expansion of both to the city database."""

import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from paths import INPUT_DIR, inp, out

warnings.filterwarnings('ignore')

# --- input ---------------------------------------------------------------
OBSERVED_FILES = [inp('Japan_merged.xlsx'), inp('USA_merged.xlsx')]
WWTP_FILE = inp('combined_wwtp.xlsx')
CITY_FILE = inp('citytreat.xlsx')

# --- output --------------------------------------------------------------
PREDICTED_FILE = out('anaerobic_predicted.xlsx')
SUMMARY_FILE = out('anaerobic_summary_by_country.xlsx')
SEWER_FILE = out('sewer_ch4.xlsx')
CITY_OUTPUT_FILE = out('city_database_with_anaerobic_sewer.xlsx')

POP_COL = 'Current Served Population (estimate)'
LABEL_COL = 'Predicted_Anaerobic'
POSITIVE_VALUE = 'Anaerobic'
NEGATIVE_VALUE = 'Non-Anaerobic'
VALUE_COLS = ['Anaerobic_digestion_ratio', 'Sewer_CH4_per_capita_kg_d']

RANDOM_SEED = 42
GWP_CH4 = 27
LITRES_PER_CAPITA_DAY = 200
SMALL_COUNTRY_POP = 10_000_000

FEATURE_COLS = [
    'Current Served Population (estimate)', 'Population_Density', 'Urban_cluster_share',
    'GDP_PerCapita_PPP', 'Extreme Monthly Average Temperature (High)', 'Building_Area',
    'Water_Area', 'Aridity_Index', 'HDI', 'Region_Area', 'Rural_share',
    'Decadal Average Temperature', 'Extreme Monthly Average Precipitation (High)',
    'Extreme Monthly Average Precipitation (Low)', 'Average Atmospheric Pressure'
]

MODEL_PARAMS = dict(n_estimators=2000, max_depth=6, learning_rate=0.10,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    reg_lambda=3, reg_alpha=1, gamma=0.1)

TARGET_COUNTRIES = [
    'Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 'Greece',
    'Ireland', 'Italy', 'Luxembourg', 'Netherlands', 'Portugal', 'Spain', 'Sweden',
    'United Kingdom', 'Bulgaria', 'Cyprus', 'Czech Republic', 'Estonia', 'Hungary',
    'Latvia', 'Poland', 'Slovakia', 'Slovenia', 'South Korea', 'Brazil',
]

SHARMA = pd.DataFrame({
    'Network': list('ABCDEFGHIJKLMNOPQRSTU'),
    'Total_PE': [13000, 7200, 7300, 4800, 24600, 18900, 7400, 9600, 13900, 33500,
                 2200000, 49500, 60400, 58900, 46800, 100200, 99000, 18600, 26000,
                 186500, 6000],
    'Climate': ['Humid subtropical', 'Humid subtropical', 'Humid subtropical',
                'Mediterranean', 'Humid subtropical', 'Humid subtropical',
                'Humid subtropical', 'Humid subtropical', 'Hot semi-arid',
                'Humid subtropical', 'Humid continental', 'Humid subtropical',
                'Humid subtropical', 'Mediterranean', 'Mediterranean', 'Mediterranean',
                'Mediterranean', 'Mediterranean', 'Humid subtropical',
                'Humid subtropical', 'Temperate oceanic'],
    'E_ML': [9.2, 6.8, 7.0, 2.6, 5.5, 10.8, 7.4, 7.3, 4.9, 6.8, 11.4, 0.6, 3.5,
             5.2, 8.3, 3.5, 8.2, 14.9, 1.2, 0.8, 18.9],
})
SHARMA['log_PE'] = np.log10(SHARMA['Total_PE'])


def load_observed():
    """Read the plants with a reported digestion status and label them."""
    frames = []
    for f in OBSERVED_FILES:
        if not os.path.exists(f):
            print(f"  ! {os.path.basename(f)} not found, skipped")
            continue
        d = pd.read_excel(f)
        need = ['NAME_0', 'Region', POP_COL, 'Anaerobic Digestion']
        miss = [c for c in need if c not in d.columns]
        if miss:
            print(f"  ! {os.path.basename(f)} is missing {miss}, skipped")
            continue
        d = d.copy()
        is_an = pd.to_numeric(d['Anaerobic Digestion'], errors='coerce').fillna(0) == 1
        d[LABEL_COL] = np.where(is_an, POSITIVE_VALUE, NEGATIVE_VALUE)
        d['Digestion_Source'] = 'Observed'
        frames.append(d)
        print(f"  {os.path.basename(f)}: {len(d)} rows, {d[LABEL_COL].value_counts().to_dict()}")
    if not frames:
        raise FileNotFoundError(f"no observed digestion file found in {INPUT_DIR}")
    out = pd.concat(frames, ignore_index=True)
    print(f"  observed total: {len(out)} rows")
    return out


def balanced_weights(y):
    """Sample weights that equalise the two classes."""
    counts = Counter(y)
    total = sum(counts.values())
    w = {k: total / (len(counts) * v) for k, v in counts.items()}
    return np.array([w[yi] for yi in y])


def train_digestion_model(observed):
    """Train the anaerobic-digestion classifier and report test performance."""
    print("\n" + "=" * 70)
    print("Part 1: anaerobic digestion classifier")
    print("=" * 70)

    observed['label'] = (observed[LABEL_COL] == POSITIVE_VALUE).astype(int)
    print(f"  anaerobic {int((observed['label'] == 1).sum())} | "
          f"non-anaerobic {int((observed['label'] == 0).sum())}")

    medians = observed[FEATURE_COLS].median(numeric_only=True)
    X_all = observed[FEATURE_COLS].fillna(medians)
    y_all = observed['label'].values

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_tmp)

    model = XGBClassifier(**MODEL_PARAMS, early_stopping_rounds=50,
                          eval_metric='logloss', random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=balanced_weights(y_train),
              eval_set=[(X_val, y_val)], verbose=100)
    print(f"  best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n  test set")
    print(classification_report(y_test, y_pred,
                                target_names=[NEGATIVE_VALUE, POSITIVE_VALUE]))
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  feature importance")
    print(imp.to_string())
    return model, medians


def predict_digestion(model, medians, wwtp):
    """Predict anaerobic digestion for plants in the target countries."""
    target = wwtp[wwtp['NAME_0'].isin(TARGET_COUNTRIES)].copy().reset_index(drop=True)
    print(f"\n  target countries: {len(target)}/{len(wwtp)} plants")

    missing = target[FEATURE_COLS].isnull().sum()
    if missing.sum() > 0:
        print(f"  missing values filled with the training median:\n{missing[missing > 0].to_string()}")
    X_pred = target[FEATURE_COLS].fillna(medians)

    pred = model.predict(X_pred)
    target[LABEL_COL] = np.where(pred == 1, POSITIVE_VALUE, NEGATIVE_VALUE)
    target['Anaerobic_Prob'] = model.predict_proba(X_pred)[:, 1].round(4)
    target['Digestion_Source'] = 'Predicted'
    print(f"  anaerobic {int((pred == 1).sum())} ({(pred == 1).mean() * 100:.1f}%) | "
          f"non-anaerobic {int((pred == 0).sum())} ({(pred == 0).mean() * 100:.1f}%)")

    rows = []
    for country in sorted(target['NAME_0'].unique()):
        sub = target[target['NAME_0'] == country]
        pop_tot = sub[POP_COL].sum()
        an = sub[LABEL_COL] == POSITIVE_VALUE
        an_pop = sub.loc[an, POP_COL].sum()
        rows.append({
            'Country': country,
            'Total_Facilities': len(sub),
            'Anaerobic_Count': int(an.sum()),
            'AD_rate_Count_%': round(an.sum() / len(sub) * 100, 1) if len(sub) else 0,
            'Total_Pop': round(pop_tot, 0),
            'Anaerobic_Pop': round(an_pop, 0),
            'AD_rate_Pop_%': round(an_pop / pop_tot * 100, 1) if pop_tot > 0 else 0,
        })
    summary = pd.DataFrame(rows).sort_values('AD_rate_Pop_%', ascending=False)

    target.to_excel(PREDICTED_FILE, index=False)
    summary.to_excel(SUMMARY_FILE, index=False)
    print(f"  saved {PREDICTED_FILE} ({len(target)} rows)")
    print(f"  saved {SUMMARY_FILE} ({len(summary)} rows)")
    return target


def estimate_sewer_ch4(wwtp):
    """Estimate sewer CH4 by matching each plant to a Sharma et al. network."""
    print("\n" + "=" * 70)
    print("Part 2: sewer CH4 (Sharma et al. 2026, Nature Water)")
    print("=" * 70)

    required = [POP_COL, 'Decadal Average Temperature', 'Decadal Average Precipitation']
    missing = [c for c in required if c not in wwtp.columns]
    if missing:
        raise ValueError(f"{WWTP_FILE} is missing columns: {missing}")

    df = wwtp.copy()
    temp = df['Decadal Average Temperature']
    precip = df['Decadal Average Precipitation'] * 12

    climate = pd.Series('Humid subtropical', index=df.index)
    climate[(precip >= 500) & (precip <= 1200) & (temp >= 15)] = 'Mediterranean'
    climate[(temp < 15) & (precip >= 500) & (precip <= 1200)] = 'Temperate oceanic'
    climate[temp < 10] = 'Humid continental'
    climate[(precip < 500) & (temp > 18)] = 'Hot semi-arid'
    df['Sewer_Climate'] = climate

    log_pe = np.log10(df[POP_COL].clip(lower=1).to_numpy(dtype=float))
    wwtp_climate = df['Sewer_Climate'].to_numpy(dtype=object)
    ref_climate = SHARMA['Climate'].to_numpy(dtype=object)
    ref_network = SHARMA['Network'].to_numpy(dtype=object)
    ref_log_pe = SHARMA['log_PE'].to_numpy(dtype=float)
    ref_e_ml = SHARMA['E_ML'].to_numpy(dtype=float)

    penalty = (wwtp_climate[:, None] != ref_climate[None, :]) * 3.0
    distance = np.abs(log_pe[:, None] - ref_log_pe[None, :])
    best = np.argmin(penalty + distance, axis=1)

    df['Matched_Network'] = ref_network[best]
    df['Matched_Climate'] = ref_climate[best]
    df['E_ML_kg_per_ML'] = ref_e_ml[best]

    df['Q_ML_d'] = df[POP_COL] * LITRES_PER_CAPITA_DAY / 1e6
    df['Sewer_CH4_kg_d'] = df['E_ML_kg_per_ML'] * df['Q_ML_d']
    df['Sewer_CH4_t_yr'] = df['Sewer_CH4_kg_d'] * 365 / 1000
    df['Sewer_CO2e_t_yr'] = df['Sewer_CH4_t_yr'] * GWP_CH4
    df['Sewer_CO2e_kt_yr'] = df['Sewer_CO2e_t_yr'] / 1000

    total_ch4_tg = df['Sewer_CH4_t_yr'].sum() / 1e6
    total_co2e_mt = df['Sewer_CO2e_t_yr'].sum() / 1e6
    print(f"  plants matched: {len(df):,}")
    print(f"  global sewer CH4: {total_ch4_tg:.3f} Tg CH4/yr")
    print(f"  global sewer CO2e (GWP{GWP_CH4}): {total_co2e_mt:.3f} Mt CO2e/yr")

    by_climate = df.groupby('Matched_Climate').agg(
        n_plants=(POP_COL, 'count'),
        CH4_Tg_yr=('Sewer_CH4_t_yr', lambda x: x.sum() / 1e6),
        CO2e_Mt_yr=('Sewer_CO2e_t_yr', lambda x: x.sum() / 1e6)).round(4)
    print("\n  by climate")
    print(by_climate.to_string())

    with pd.ExcelWriter(SEWER_FILE, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Main', index=False)
        by_climate.to_excel(writer, sheet_name='Summary_by_Climate')
    print(f"\n  saved {SEWER_FILE}")
    return df


def aggregate_digestion(dig_df):
    """Population-weighted anaerobic digestion share by country and region."""
    valid = dig_df[dig_df['Region'].notna() & (dig_df['Region'].astype(str) != 'nan')].copy()
    valid[POP_COL] = pd.to_numeric(valid[POP_COL], errors='coerce').fillna(0)
    rows = []
    for (name0, region), g in valid.groupby(['NAME_0', 'Region']):
        pop = g[POP_COL].sum()
        an_pop = g.loc[g[LABEL_COL] == POSITIVE_VALUE, POP_COL].sum()
        rows.append({'NAME_0': name0, 'Region': region,
                     'Anaerobic_digestion_ratio': an_pop / pop if pop > 0 else 0.0,
                     'Total_Served_Population': pop})
    out = pd.DataFrame(rows)
    print(f"  digestion regions: {len(out)}")
    return out


def aggregate_sewer(sewer_df):
    """Per-capita sewer CH4 intensity by country and region."""
    valid = sewer_df[sewer_df['Region'].notna() & (sewer_df['Region'].astype(str) != 'nan')].copy()
    valid[POP_COL] = pd.to_numeric(valid[POP_COL], errors='coerce').fillna(0)
    valid['Sewer_CH4_kg_d'] = pd.to_numeric(valid['Sewer_CH4_kg_d'], errors='coerce').fillna(0)
    rows = []
    for (name0, region), g in valid.groupby(['NAME_0', 'Region']):
        pop = g[POP_COL].sum()
        ch4 = g['Sewer_CH4_kg_d'].sum()
        rows.append({'NAME_0': name0, 'Region': region,
                     'Sewer_CH4_per_capita_kg_d': ch4 / pop if pop > 0 else 0.0,
                     'Total_Served_Population': pop})
    out = pd.DataFrame(rows)
    print(f"  sewer regions: {len(out)}")
    return out


def merge_region_tables(dig_region, sewer_region):
    """Outer-join the two regional tables on country and region."""
    merged = pd.merge(
        dig_region[['NAME_0', 'Region', 'Anaerobic_digestion_ratio', 'Total_Served_Population']],
        sewer_region[['NAME_0', 'Region', 'Sewer_CH4_per_capita_kg_d', 'Total_Served_Population']],
        on=['NAME_0', 'Region'], how='outer', suffixes=('_dig', '_sewer'))
    merged['Total_Served_Population'] = merged[
        ['Total_Served_Population_dig', 'Total_Served_Population_sewer']].max(axis=1)
    merged = merged.drop(columns=['Total_Served_Population_dig', 'Total_Served_Population_sewer'])
    print(f"  merged regions: {len(merged)}")
    return merged


def build_admin_tree(city_data):
    """Build the administrative hierarchy from the city table."""
    tree = {}
    for _, row in city_data.iterrows():
        name0 = row['NAME_0']
        if pd.isna(name0) or str(name0) == 'nan':
            continue
        if name0 not in tree:
            tree[name0] = {'info': {'level': 0, 'population': 0, 'has_data': False,
                                    'data': None, 'data_source': ''}, 'children': {}}
        try:
            tree[name0]['info']['population'] += float(row.get('Population', 0) or 0)
        except (ValueError, TypeError):
            pass
        current = tree[name0]
        for level in range(1, 5):
            cn = f'NAME_{level}'
            if cn not in row or pd.isna(row[cn]) or str(row[cn]) == 'nan':
                break
            rname = str(row[cn])
            if rname not in current['children']:
                current['children'][rname] = {
                    'info': {'level': level, 'population': 0, 'parent': current,
                             'has_data': False, 'data': None, 'data_source': ''},
                    'children': {}}
            try:
                current['children'][rname]['info']['population'] += float(row.get('Population', 0) or 0)
            except (ValueError, TypeError):
                pass
            current = current['children'][rname]
    print(f"  countries in the tree: {len(tree)}")
    return tree


def mark_data_nodes(tree, region_df):
    """Attach the regional values to the matching tree nodes."""
    for _, row in region_df.iterrows():
        name0, region = str(row['NAME_0']), str(row['Region'])
        if name0 == 'nan' or region == 'nan' or name0 not in tree:
            continue
        country_node = tree[name0]
        if region == name0:
            country_node['info']['has_data'] = True
            country_node['info']['data'] = row
            country_node['info']['data_source'] = 'Original'
            continue
        stack = [country_node]
        found = False
        while stack and not found:
            node = stack.pop()
            for child_name, child_node in node['children'].items():
                if child_name == region:
                    child_node['info']['has_data'] = True
                    child_node['info']['data'] = row
                    child_node['info']['data_source'] = 'Original'
                    found = True
                    break
                stack.append(child_node)


def _safe_pop(data):
    """Served population of one node, zero when unavailable."""
    try:
        v = data.get('Total_Served_Population', 0)
        return float(v) if pd.notna(v) else 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _weighted_avg(children_with_data, cols):
    """Served-population weighted mean over sibling nodes."""
    total_pop = 0.0
    sums = defaultdict(float)
    for _, node in children_with_data:
        d = node['info']['data']
        sp = _safe_pop(d)
        if sp <= 0:
            continue
        total_pop += sp
        for col in cols:
            val = d.get(col, np.nan)
            if val is not None and pd.notna(val):
                try:
                    sums[col] += float(val) * sp
                except (TypeError, ValueError):
                    pass
    if total_pop <= 0:
        return None
    return {col: sums[col] / total_pop for col in cols}


def propagate_data(tree, value_cols):
    """Fill nodes without data from their parent or their siblings."""
    for level in range(3, 0, -1):
        for name0, country_node in tree.items():
            parents, stack = [], [country_node]
            while stack:
                node = stack.pop()
                if node['info']['level'] == level - 1:
                    parents.append(node)
                stack.extend(node['children'].values())

            for parent in parents:
                at_level = [(cn, n) for cn, n in parent['children'].items()
                            if n['info']['level'] == level]
                with_data = [(cn, n) for cn, n in at_level if n['info']['has_data']]
                without = [(cn, n) for cn, n in at_level if not n['info']['has_data']]
                parent_has = parent['info']['has_data']

                if parent_has and not with_data:
                    for _, node in without:
                        node['info'].update(has_data=True, data=parent['info']['data'],
                                            data_source=f'Propagated(from NAME_{level - 1})')
                elif not parent_has and with_data and without:
                    avg = _weighted_avg(with_data, value_cols)
                    if avg is not None:
                        for cn, node in without:
                            node['info'].update(
                                has_data=True,
                                data={**avg, 'NAME_0': name0, 'Region': cn,
                                      'Total_Served_Population': 0},
                                data_source=f'Propagated(sibling avg, n={len(with_data)})')
                elif parent_has and with_data and without:
                    parent_sp = _safe_pop(parent['info']['data'])
                    children_sp = sum(_safe_pop(n['info']['data']) for _, n in with_data)
                    if parent_sp >= children_sp:
                        for _, node in without:
                            node['info'].update(has_data=True, data=parent['info']['data'],
                                                data_source='Propagated(parent larger pop)')
                    else:
                        avg = _weighted_avg(with_data, value_cols)
                        if avg is not None:
                            for cn, node in without:
                                node['info'].update(
                                    has_data=True,
                                    data={**avg, 'NAME_0': name0, 'Region': cn,
                                          'Total_Served_Population': 0},
                                    data_source='Propagated(sibling avg, child larger pop)')

    for name0, country_node in tree.items():
        try:
            is_small = float(country_node['info']['population']) < SMALL_COUNTRY_POP
        except (TypeError, ValueError):
            is_small = False
        if is_small and country_node['info']['has_data']:
            for node in country_node['children'].values():
                if node['info']['level'] == 1 and not node['info']['has_data']:
                    node['info'].update(has_data=True, data=country_node['info']['data'],
                                        data_source='Propagated(small country)')

    for name0, country_node in tree.items():
        if not country_node['info']['has_data']:
            continue
        any_child, stack = False, list(country_node['children'].values())
        while stack and not any_child:
            node = stack.pop()
            if node['info']['has_data']:
                any_child = True
            stack.extend(node['children'].values())
        if not any_child:
            stack = list(country_node['children'].values())
            while stack:
                node = stack.pop()
                node['info'].update(has_data=True, data=country_node['info']['data'],
                                    data_source='Propagated(country-only)')
                stack.extend(node['children'].values())


def extend_city_database(city_data, tree, value_cols):
    """Write the regional values back onto every city row."""
    out = city_data.copy()
    for col in value_cols:
        out[col] = pd.array([pd.NA] * len(out), dtype='Float64')
    out['Admin_Match_Level'] = ''
    out['Data_Source'] = 'None'

    n_orig = n_prop = n_none = 0
    for i, row in out.iterrows():
        name0 = str(row.get('NAME_0', ''))
        matched, max_level = None, 0
        if name0 and name0 != 'nan' and name0 in tree:
            current = tree[name0]
            for level in range(1, 5):
                cn = f'NAME_{level}'
                if cn not in row or pd.isna(row[cn]) or str(row[cn]) == 'nan':
                    break
                rname = str(row[cn])
                if rname in current['children']:
                    current = current['children'][rname]
                    if current['info']['has_data']:
                        matched, max_level = current, level
                else:
                    break
            if matched is None and tree[name0]['info']['has_data']:
                node = tree[name0]
                try:
                    is_small = float(node['info']['population']) < SMALL_COUNTRY_POP
                except (TypeError, ValueError):
                    is_small = False
                if is_small or 'country-only' in str(node['info'].get('data_source', '')):
                    matched, max_level = node, 0

        if matched is None:
            n_none += 1
            continue

        d = matched['info']['data']
        src = matched['info'].get('data_source', 'Original')
        label = 'Original' if src in ('Original', '') else 'Propagated'
        for col in value_cols:
            try:
                v = d.get(col, np.nan)
            except AttributeError:
                v = np.nan
            if v is not None and pd.notna(v):
                try:
                    out.at[i, col] = float(v)
                except (TypeError, ValueError):
                    pass
        out.at[i, 'Admin_Match_Level'] = f'NAME_{max_level}'
        out.at[i, 'Data_Source'] = label
        n_orig += (label == 'Original')
        n_prop += (label == 'Propagated')

    print(f"  original {n_orig} | propagated {n_prop} | unmatched {n_none}")
    out.to_excel(CITY_OUTPUT_FILE, index=False)
    print(f"  saved {CITY_OUTPUT_FILE} ({len(out)} rows)")
    return out


def main():
    """Run the classifier, the sewer estimate, and the expansion to the city database."""
    print(f"loading observed digestion labels from {INPUT_DIR}/")
    observed = load_observed()

    print(f"\nloading {WWTP_FILE}")
    wwtp = pd.read_excel(WWTP_FILE)
    print(f"  {len(wwtp)} plants")

    model, medians = train_digestion_model(observed)
    predicted = predict_digestion(model, medians, wwtp)

    sewer = estimate_sewer_ch4(wwtp)

    print("\n" + "=" * 70)
    print("Part 3: aggregate to region and expand to the city database")
    print("=" * 70)

    keep = ['NAME_0', 'Region', POP_COL, LABEL_COL, 'Digestion_Source']
    dig_df = pd.concat([predicted[keep], observed[keep]], ignore_index=True)
    print(f"  digestion records (predicted + observed): {len(dig_df)}")

    for df in (dig_df, sewer):
        for col in ['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3', 'NAME_4', 'Region']:
            if col in df.columns:
                df[col] = df[col].astype(str)

    region_df = merge_region_tables(aggregate_digestion(dig_df), aggregate_sewer(sewer))

    print(f"\nloading {CITY_FILE}")
    city_data = pd.read_excel(CITY_FILE)
    print(f"  {len(city_data)} rows")
    for col in [c for c in city_data.columns if c.startswith('NAME_')] + ['Region']:
        if col in city_data.columns:
            city_data[col] = city_data[col].astype(str)
    if 'Population' in city_data.columns:
        city_data['Population'] = pd.to_numeric(city_data['Population'], errors='coerce').fillna(0)

    tree = build_admin_tree(city_data)
    mark_data_nodes(tree, region_df)
    propagate_data(tree, VALUE_COLS)
    extend_city_database(city_data, tree, VALUE_COLS)

    print("\ndone")


if __name__ == '__main__':
    main()