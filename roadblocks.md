## Roadblock 1 — CSV Header Issue
CSV first row was being read as column names instead of data.
Fix: Used `names=` and `skiprows=1` parameters in pd.read_csv()

## Roadblock 2 — Timestamp Format
ts column was in Unix float format (scientific notation).
Fix: Used pd.to_datetime with unit='s' to convert properly

## Roadblock 3 — ClaSPSegmentation requires numba
ClaSPSegmentation from sktime.detection.clasp throws 
ModuleNotFoundError because numba is not listed as a 
required dependency during standard pip install of sktime.
Fix attempted: pip install numba
Potential PR: Add numba to optional dependencies documentation
or add a clearer error message guiding users to install it.

## Roadblock 4 — numba JIT compilation delay
First run of ClaSPSegmentation takes several minutes due to 
numba JIT compilation. This is expected behaviour but could 
be a problem for real-time agricultural machinery monitoring.
Potential PR: Add a warning in sktime docs that first run 
of ClaSP will be slow due to numba compilation overhead.

## Roadblock 5 — ClaSP too slow for large series
ClaSPSegmentation with period_length=50 on 187k points 
ran for 48+ minutes without completing.
Fix: Subsample to 5000 points for demonstration purposes.
Real project will need either subsampling strategy or 
a more scalable detection algorithm.

## Roadblock 6 — STRAY has no predict() method
STRAY detector in sktime throws AttributeError when calling 
.predict() — the standard sktime detector interface method.
This breaks the standard detector API contract where all 
detectors should implement fit/predict.
Potential PR: Implement predict() method in STRAY or fix 
the API inconsistency and add to documentation.