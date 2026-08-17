# Global-WWTP-GHG-accounting

## System requirements

Python 3.12. Package versions are [XGBoost 2.1.3, scikit-learn 1.6.1, pandas 2.2.3,
GeoPandas 0.14.4, NumPy 1.26.4, SciPy 1.15.1]. Tested on Windows 11 with Python 3.12
and the package versions listed above. No non-standard hardware is required.

## Installation

```
git clone https://github.com/QixiangZ/Global-WWTP-GHG-accounting.git
pip install -r requirements.txt
```

Typical install time: about 5 minutes.

## Instructions for use

Put the [input] folder and the code in the same folder, then run the scripts in
this order:

1. `AD_Sewer_code.py` — anaerobic digestion classification and sewer CH4
2. `Main_code.py` — classification and Monte Carlo emission estimates
3. `Analysis_code.py` — aggregation and comparison with national inventories

Steps 1 and 2 do not write directly into the file that step 3 reads. Their outputs
have to be entered into the corresponding columns of
`Region_resolved pollutant flows and GHG emissions.xlsx` and the file placed back in
[input] before step 3 is run; that file documents what every column holds. The copy
already in [input] is the version behind the published results, so step 3 can be run
on its own to reproduce them.


The scripts run on the complete database as provided, with the settings used for
the published results already in place. No separate demo dataset is needed.

## Expected output

`Main_code.py` writes emissions with 95% prediction intervals at global,
continental, income-group, national, regional and city level to `country_results/`,
together with the BOD and TN loads entering secondary treatment and the
city- and country-level reduction rates and BOD-to-sludge ratios.

`Analysis_code.py` writes the aggregated tables and the comparison with the
reported national inventories to `Average_result/`.

## Expected run time

About 10 hours on a normal desktop computer for the complete database.
