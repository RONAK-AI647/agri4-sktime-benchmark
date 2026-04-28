## Roadblock 1 — CSV Header Issue
CSV first row was being read as column names instead of data.
Fix: Used `names=` and `skiprows=1` parameters in pd.read_csv()

## Roadblock 2 — Timestamp Format
ts column was in Unix float format (scientific notation).
Fix: Used pd.to_datetime with unit='s' to convert properly