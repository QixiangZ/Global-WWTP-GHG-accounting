"""City-level WWTP emissions aggregated to city, province, country and continent,
compared with external inventories, and ranked against the reported wastewater totals.

The country ranking of reported wastewater emissions, previously a separate script
(source_wastewater.py), is produced here by rank_wastewater_by_country().
"""

import os
from glob import glob

import numpy as np
import pandas as pd

from paths import OUTPUT_DIR, inp

# --- input ---------------------------------------------------------------
INPUT_FILE = inp("Region_resolved pollutant flows and GHG emissions.xlsx")
INPUT_SHEET = "GHG Calculation"
SOURCE_DATA_DIR = inp("CH4_N2O_DATA_From_EPA")

# --- output --------------------------------------------------------------
RESULT_DIR = OUTPUT_DIR
ANALYZED_RESULT_FILE = os.path.join(RESULT_DIR, "country_analyzed.xlsx")
WASTEWATER_RANK_FILE = os.path.join(RESULT_DIR, "epa_wastewater_by_country.xlsx")

GDP_PC_COL = "GDP_PerCapita_PPP"

POP_COL = 'Population'
INCOME_COL = 'Income_Level'

KT_TO_MT = 1000.0

CH4_COLUMNS = ['CH4_WWTP', 'CH4_Sewer', 'CH4_unsafelytreated', 'CH4_eff', 'CH4_Total']
N2O_COLUMNS = ['N2O_WWTP', 'N2O_unsafelytreated', 'N2O_eff', 'N2O_Total']
SUM_COLUMNS = CH4_COLUMNS + N2O_COLUMNS

WEIGHTED_COLUMNS = ['CH4_Intensity', 'N2O_Intensity']
ATTR_COLUMNS = ['Continent', INCOME_COL]

HEADLINE = ['GHG_WWTP', 'CH4_WWTP', 'N2O_WWTP']

INVENTORY_GASES = ['CH4', 'N2O']
INVENTORY_COMBINED = 'GHG'

no_province_countries = {
    'Aruba', 'Anguilla', 'Bahrain', 'Saint-Barthélemy', 'Cocos Islands',
    'Cook Islands', 'Curaçao', 'Christmas Island', 'Falkland Islands',
    'Guernsey', 'Gibraltar', 'Hong Kong', 'Kiribati', 'Kuwait', 'Macao',
    'Saint-Martin', 'Monaco', 'Maldives', 'Marshall Islands', 'Montserrat',
    'Norfolk Island', 'Niue', 'Pitcairn Islands', 'Singapore',
    'Svalbard and Jan Mayen', 'Saint Pierre and Miquelon', 'Sint Maarten',
    'Seychelles', 'Vatican City', 'Wallis and Futuna', 'Paracel Islands'
}

COUNTRY_NAME_MAP = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'Cabo Verde': 'Cape Verde',
    'Congo': 'Republic of Congo',
    'Congo_the Democratic Republic of the': 'Democratic Republic of the Congo',
    "Cote d'Ivoire": "Côte d'Ivoire",
    'Czechia': 'Czech Republic',
    "Democratic People's Republic of Korea": 'North Korea',
    'Eswatini': 'Swaziland',
    'Iran (Islamic Republic of)': 'Iran',
    'Lao PDR': 'Laos',
    "Lao People's Democratic Republic": 'Laos',
    'Micronesia (Federated States of)': 'Micronesia',
    'North Macedonia': 'Macedonia',
    'Republic of Korea': 'South Korea',
    'Republic of Moldova': 'Moldova',
    'Russian Federation': 'Russia',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'State of Palestine': 'Palestina',
    'Syrian Arab Republic': 'Syria',
    'Timor Leste': 'Timor-Leste',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America': 'United States',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Viet Nam': 'Vietnam',
}


def load_and_prepare(file_path, sheet_name=INPUT_SHEET):
    """Read the city table, merge GDP and fill missing administrative levels."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError:
        available = pd.ExcelFile(file_path).sheet_names
        raise ValueError(f"{file_path} has no sheet '{sheet_name}'. Sheets present: {available}")
    print(f"read {df.shape[0]} rows, {df.shape[1]} columns from sheet '{sheet_name}'")
    df = df.copy()

    required = (['NAME_0', 'NAME_1', 'NAME_2', 'Continent', INCOME_COL, POP_COL, GDP_PC_COL]
                + SUM_COLUMNS + WEIGHTED_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input file is missing required columns: {missing}")

    df['GHG_WWTP'] = df['CH4_WWTP'] + df['N2O_WWTP']

    df['GDP_Total'] = df[GDP_PC_COL] * df[POP_COL]
    n_gdp = df[GDP_PC_COL].notna().sum()
    print(f"'{GDP_PC_COL}': {n_gdp}/{len(df)} rows populated ({n_gdp / len(df) * 100:.1f}%)")

    df['City_Level'] = None
    for idx, row in df.iterrows():
        country = row['NAME_0']
        province = row.get('NAME_1', None)
        city = row.get('NAME_2', None)
        if country in no_province_countries:
            df.at[idx, 'NAME_1'] = country
            df.at[idx, 'NAME_2'] = country
            df.at[idx, 'City_Level'] = 'Country_as_City'
        elif pd.isna(province) or province == '':
            df.at[idx, 'NAME_1'] = country
            df.at[idx, 'NAME_2'] = country
            df.at[idx, 'City_Level'] = 'Country_as_Province_as_City'
        elif pd.isna(city) or city == '':
            df.at[idx, 'NAME_2'] = province
            df.at[idx, 'City_Level'] = 'Province_as_City'
        else:
            df.at[idx, 'City_Level'] = 'City'

    for col, label in [('Continent', 'Continent'), (INCOME_COL, INCOME_COL)]:
        n_uniq = df.groupby('NAME_0')[col].nunique()
        bad = n_uniq[n_uniq > 1]
        if len(bad):
            print(f"! these countries have more than one {label} value, the first is used: {list(bad.index)}")

    for col in SUM_COLUMNS + WEIGHTED_COLUMNS + [POP_COL]:
        n_na = df[col].isna().sum()
        if n_na > 0:
            print(f"  column '{col}' has {n_na} missing values")

    return df


def weighted_avg_by_group(data, group_cols, value_cols, weight_col):
    """Weighted mean of the given columns within each group."""
    rows = []
    for keys, g in data.groupby(group_cols):
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        w = g[weight_col].fillna(0).values.astype(float)
        for col in value_cols:
            v = g[col].values.astype(float)
            ok = (~np.isnan(v)) & (w > 0)
            rec[col] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(data, group_cols):
    """Aggregate one administrative level: totals, weighted intensity and first-valued attributes."""
    sum_cols = SUM_COLUMNS + ['GHG_WWTP', POP_COL, 'GDP_Total']
    sums = data.groupby(group_cols)[sum_cols].sum().reset_index()
    weighted = weighted_avg_by_group(data, group_cols, WEIGHTED_COLUMNS, POP_COL)
    result = sums.merge(weighted, on=group_cols, how='left')

    attrs_cols = [c for c in ATTR_COLUMNS if c not in group_cols]
    if attrs_cols:
        attrs = data.groupby(group_cols)[attrs_cols].first().reset_index()
        result = result.merge(attrs, on=group_cols, how='left')
    return result


def to_mt(data):
    """Convert every emission column from kt CO2e to Mt CO2e."""
    data = data.copy()
    for col in SUM_COLUMNS + ['GHG_WWTP']:
        if col in data.columns:
            data[col] = data[col] / KT_TO_MT
    return data


def add_global_share(data, cols):
    """Share of the global total for the given columns."""
    data = data.copy()
    for col in cols:
        total = data[col].sum()
        data[f'{col}_GlobalShare_%'] = data[col] / total * 100 if total > 0 else np.nan
    return data


def order_country_columns(df):
    """Put the country table into the requested column order."""
    lead = ['NAME_0', INCOME_COL]
    lead += HEADLINE + [f'{c}_GlobalShare_%' for c in HEADLINE]
    lead += [POP_COL, f'{POP_COL}_GlobalShare_%']
    rest = [c for c in df.columns if c not in lead]
    return df[[c for c in lead if c in df.columns] + rest]


def order_city_rows(df):
    """Sort cities by total GDP, with zero-emission cities placed last."""
    df = df.copy()
    df['_zero'] = (df['GHG_WWTP'].fillna(0) <= 0).astype(int)
    df = df.sort_values(['_zero', 'GDP_Total'], ascending=[True, False])
    return df.drop(columns=['_zero']).reset_index(drop=True)


def load_formatted_data(source_dir=SOURCE_DATA_DIR):
    """Read the formatted external inventory files from the input directory."""
    print("reading the formatted inventory files...")
    excel_files = glob(os.path.join(source_dir, "*_formatted*.xlsx"))
    if not excel_files:
        print(f"no formatted file found in {source_dir}")
        return None

    print(f"{len(excel_files)} formatted files found:")
    frames = []
    for file_path in excel_files:
        try:
            d = pd.read_excel(file_path)
            print(f"read {os.path.basename(file_path)}: {d.shape}")
            frames.append(d)
        except Exception as e:
            print(f"failed to read {file_path}: {str(e)}")

    if not frames:
        print("no formatted file could be read")
        return None

    combined = pd.concat(frames, ignore_index=True)
    print(f"combined shape: {combined.shape}")

    if 'Country' in combined.columns:
        n = combined['Country'].isin(COUNTRY_NAME_MAP).sum()
        combined['Country'] = combined['Country'].replace(COUNTRY_NAME_MAP)
        print(f"{n} country names mapped to the NAME_0 naming")
    return combined


def _rank_desc(series):
    """Dense descending rank, largest value gets rank 1."""
    return series.rank(ascending=False, method='min')


def analyze_emissions_with_sources(country_df, source_dir=SOURCE_DATA_DIR,
                                   result_file=ANALYZED_RESULT_FILE):
    """Compare the country totals with each external inventory, sector by sector."""
    print("\n=== Step 2.1: country totals from step 1 used as the reference ===")
    ours = country_df.rename(columns={'NAME_0': 'Country'}).copy()
    print(f"country table shape: {ours.shape}")

    print("\n=== Step 2.2: read the external inventories ===")
    source_data = load_formatted_data(source_dir)
    if source_data is None:
        print("no external inventory available, saving the country table as is")
        ours.to_excel(result_file, index=False)
        print(f"saved: {result_file}")
        return ours

    print("\n=== Step 2.3: pick the reporting year ===")
    year_col = next((c for c in source_data.columns if 'Y_2022' in str(c)), None)
    if year_col is None:
        years = [c for c in source_data.columns if str(c).startswith('Y_')]
        if not years:
            print("no year column found")
            return ours
        year_col = max(years)
        print(f"no 2022 column, using the most recent year: {year_col}")
    else:
        print(f"using {year_col}")

    print("\n=== Step 2.4: split the inventory into wastewater and other sectors ===")
    is_ww = source_data['source'].str.contains('wastewater', case=False, na=False)
    ww = source_data[is_ww]
    other = source_data[~is_ww]
    print(f"wastewater rows: {len(ww)} | other sector rows: {len(other)}")

    ww_sum = ww.groupby(['Country', 'Substance'])[year_col].sum()
    other_sum = other.groupby(['Country', 'Substance'])[year_col].sum()

    print("\n=== Step 2.5: per-country shares and sector ranks ===")
    result = ours.copy()

    for gas, our_ww_col, our_wwtp_col in [('CH4', 'CH4_Total', 'CH4_WWTP'),
                                          ('N2O', 'N2O_Total', 'N2O_WWTP')]:
        orig_ww, other_tot = [], []
        rank_orig, rank_ours, rank_wwtp = [], [], []

        for _, row in result.iterrows():
            c = row['Country']
            o_ww = float(ww_sum.get((c, gas), np.nan))
            o_other = float(other_sum.get((c, gas), np.nan))
            orig_ww.append(o_ww)
            other_tot.append(o_other)

            sectors = other[(other['Country'] == c) & (other['Substance'] == gas)][year_col]
            if len(sectors) == 0:
                rank_orig.append(np.nan)
                rank_ours.append(np.nan)
                rank_wwtp.append(np.nan)
                continue
            rank_orig.append(int((sectors > o_ww).sum() + 1) if np.isfinite(o_ww) else np.nan)
            rank_ours.append(int((sectors > row[our_ww_col]).sum() + 1))
            rank_wwtp.append(int((sectors > row[our_wwtp_col]).sum() + 1))

        orig_ww = np.array(orig_ww, dtype=float)
        other_tot = np.array(other_tot, dtype=float)
        total_orig = other_tot + orig_ww
        total_corr = other_tot + result[our_ww_col].values

        result[f'{gas}_OtherSectors'] = other_tot
        result[f'{gas}_Inventory_WW'] = orig_ww
        result[f'{gas}_Inventory_CountryTotal'] = total_orig
        result[f'{gas}_Inventory_WW_Share_%'] = np.where(
            total_orig > 0, orig_ww / total_orig * 100, np.nan)
        result[f'{gas}_Inventory_WW_SectorRank'] = rank_orig

        result[f'{gas}_Corrected_CountryTotal'] = total_corr
        result[f'{gas}_Our_WW_Share_%'] = np.where(
            total_corr > 0, result[our_ww_col].values / total_corr * 100, np.nan)
        result[f'{gas}_Our_WW_SectorRank'] = rank_ours

        result[f'{gas}_Our_WWTP_Share_%'] = np.where(
            total_corr > 0, result[our_wwtp_col].values / total_corr * 100, np.nan)
        result[f'{gas}_Our_WWTP_SectorRank'] = rank_wwtp

        matched = int(np.isfinite(other_tot).sum())
        print(f"  {gas}: {matched}/{len(result)} countries matched to the inventory")

    result.to_excel(result_file, index=False)
    print(f"\nsaved: {result_file}")
    return result


def _pick_year_column(data):
    """Use the 2022 column of the inventory, or the most recent year available."""
    col = next((c for c in data.columns if 'Y_2022' in str(c)), None)
    if col is not None:
        return col
    years = sorted(c for c in data.columns if str(c).startswith('Y_'))
    if not years:
        raise KeyError('the inventory files contain no year column')
    print(f"no 2022 column, using the most recent year: {years[-1]}")
    return years[-1]


def rank_wastewater_by_country(country_df, source_dir=SOURCE_DATA_DIR,
                               result_file=WASTEWATER_RANK_FILE):
    """Rank the countries by their reported wastewater emissions and their global share."""
    print("\n=== Step 3: reported wastewater emissions ranked across countries ===")
    source_data = load_formatted_data(source_dir)
    if source_data is None:
        print("no external inventory available, ranking skipped")
        return None

    year_col = _pick_year_column(source_data)
    print(f"using {year_col}")

    is_ww = source_data['source'].str.contains('wastewater', case=False, na=False)
    ww = source_data[is_ww]
    print(f"wastewater rows: {len(ww)} | sources: {sorted(ww['source'].unique())}")

    table = ww.pivot_table(index='Country', columns='Substance', values=year_col,
                           aggfunc='sum').reindex(columns=INVENTORY_GASES)
    table[INVENTORY_COMBINED] = table[INVENTORY_GASES].sum(axis=1, min_count=1)
    table = table.reset_index()

    countries = set(country_df['NAME_0'].dropna().astype(str))
    absent = sorted(countries - set(table['Country']))
    if absent:
        print(f"! {len(absent)} of our countries have no wastewater row: {absent[:15]}")
    table = table[table['Country'].isin(countries)].copy()

    labels = INVENTORY_GASES + [INVENTORY_COMBINED]
    for label in labels:
        total = table[label].sum()
        table[f'{label}_GlobalShare_%'] = table[label] / total * 100 if total > 0 else np.nan
        table[f'{label}_Rank'] = _rank_desc(table[label])
        print(f"  {label}: total across the retained countries = {total:,.2f}")

    ordered = ['Country'] + [f'{label}{suffix}' for label in labels
                             for suffix in ('', '_GlobalShare_%', '_Rank')]
    table = table[ordered].sort_values(f'{INVENTORY_COMBINED}',
                                       ascending=False, na_position='last').reset_index(drop=True)
    table.to_excel(result_file, index=False)
    print(f"saved: {result_file} ({len(table)} countries)")

    print(f"\ntop 10 by {' + '.join(INVENTORY_GASES)}")
    for _, r in table.head(10).iterrows():
        rank = r[f'{INVENTORY_COMBINED}_Rank']
        share = r[f'{INVENTORY_COMBINED}_GlobalShare_%']
        rank_txt = int(rank) if pd.notna(rank) else '-'
        share_txt = f'{share:.2f}%' if pd.notna(share) else 'n/a'
        print(f"  {rank_txt:>3}. {r['Country']:<25} "
              f"{r[INVENTORY_COMBINED]:>12,.2f}  ({share_txt})")
    return table


def main():
    """Run the aggregation and the cross-source comparison."""
    df = load_and_prepare(INPUT_FILE)
    df.to_excel(os.path.join(RESULT_DIR, "city_raw_processed.xlsx"), index=False)
    print(f"saved {RESULT_DIR}/city_raw_processed.xlsx (row level, derived columns added)")

    city_result = to_mt(aggregate(df, ['NAME_0', 'NAME_1', 'NAME_2']))
    city_result = add_global_share(city_result, ['GHG_WWTP', POP_COL, 'GDP_Total'])
    city_result = order_city_rows(city_result)
    city_result.to_excel(os.path.join(RESULT_DIR, "city_result.xlsx"), index=False)
    print(f"saved {RESULT_DIR}/city_result.xlsx ({len(city_result)} rows)")

    province_result = to_mt(aggregate(df, ['NAME_0', 'NAME_1']))
    province_result = add_global_share(province_result, HEADLINE + [POP_COL])
    province_result.to_excel(os.path.join(RESULT_DIR, "province_result.xlsx"), index=False)
    print(f"saved {RESULT_DIR}/province_result.xlsx ({len(province_result)} rows)")

    country_result = to_mt(aggregate(df, ['NAME_0']))
    country_result = add_global_share(country_result, HEADLINE + [POP_COL])
    country_result = order_country_columns(country_result)
    country_result.to_excel(os.path.join(RESULT_DIR, "country_result.xlsx"), index=False)
    print(f"saved {RESULT_DIR}/country_result.xlsx ({len(country_result)} rows)")

    continent_result = to_mt(aggregate(df, ['Continent']))
    continent_result = add_global_share(continent_result, HEADLINE + [POP_COL])
    continent_result.to_excel(os.path.join(RESULT_DIR, "continent_result.xlsx"), index=False)
    print(f"saved {RESULT_DIR}/continent_result.xlsx ({len(continent_result)} rows)")

    print("\nadministrative level fallbacks:")
    print(df['City_Level'].value_counts().to_string())

    analyze_emissions_with_sources(country_result)
    rank_wastewater_by_country(country_result)


if __name__ == "__main__":
    main()