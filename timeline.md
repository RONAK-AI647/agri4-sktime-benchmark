## 📋 Project Phases & Timeline

---

### Phase 1 — Data Ingestion & Preprocessing

Before any model, we understand the data's shape. Key questions to answer immediately:
- How many sensors are there?
- What are their sampling rates?
- Are streams synchronous or interleaved?
- How long are the events on average?

**Loading into sktime panel format**
Raw CSV is converted exactly as done with the IoT surrogate — device/sensor as the
instance index, timestamp as the time index, making a proper `pd-multiindex` panel:

```python
check_raise(df, mtype="pd-multiindex", scitype="Panel")
```

**Timestamp alignment**
With multiple sensors firing at different rates (vibration at 1kHz, acoustic at 2kHz,
mechanical at 500Hz), we resample everything to a common clock:

```python
# Forward fill for mechanical state columns
# Linear interpolation for continuous sensor readings
df.resample("1ms").interpolate(method="linear")
```

**Window construction**
We slide a fixed-length window over the continuous stream and label each window.
A window is labelled `1` if any part overlaps a ground-truth event, else `0`.

| Parameter | Example Value |
|---|---|
| `window_size` | 200ms at 1kHz = 200 samples |
| `stride` | 50% overlap = 100 samples |

**Class imbalance handling**
Do not just oversample. Use stratified splitting and always report **event-level metrics**
— a classifier that predicts "normal" 100% of the time will still achieve 99.9% window
accuracy.

---

### Phase 2 — Feature Extraction Pipeline

Built as a sktime-compatible `FunctionTransformer` or `BaseTransformer` subclass so it
plugs into any pipeline.

| Feature Family | Features |
|---|---|
| **Time-domain** *(per window, per channel)* | Mean, std, RMS energy, peak-to-peak amplitude, slope, kurtosis *(sensitive to impulsive events — matters for foreign object impacts)* |
| **Frequency-domain** *(via FFT)* | Dominant frequency, power in frequency bands (low/mid/high), spectral entropy, spectral centroid *(foreign object impacts show broadband energy spikes — spectral entropy drops sharply)* |
| **Cross-sensor** | Pearson correlation (vibration vs acoustic), cross-correlation lag, ratio of vibration to acoustic energy *(distinguishes foreign object events from normal mechanical vibration)* |

```python
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Option A — automatic feature extraction via ROCKET
pipe = Pipeline([
    ("rocket", MiniRocket()),
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier())
])

# Option B — explicit feature extraction (more interpretable)
from sktime.transformations.series.summarize import SummaryTransformer
pipe = Pipeline([
    ("features", SummaryTransformer()),   # mean, std, etc per channel
    ("clf",      RandomForestClassifier())
])
```

---

### Phase 3 — Detection Algorithms

All algorithms run in parallel on the same windowed dataset and compared on the same
evaluation suite.

| Tier | Algorithms | Notes |
|---|---|---|
| **Tier 1 — Baselines** | Statistical control limits (mean + 3-sigma) | Floor benchmark — anything that doesn't beat this isn't worth deploying |
| **Tier 2 — sktime Classifiers** | `RocketClassifier`, `TimeSeriesForestClassifier` (`use_multivariate="yes"`), `ShapeletTransformClassifier` | Supervised, labeled data required |
| **Tier 3 — Semi-supervised** | Autoencoder / density estimator on normal data, `STRAY`, `ClaSPSegmentation` | Train purely on normal data — no labeled event data needed |

> **Advance detection time** is the most important metric for this project — it determines
> whether the warning is actually actionable. Handled via PR [#9939](https://github.com/sktime/sktime/pull/9939)
> `MultiChannelDetectionDelay` once merged.

---

### 🗓️ Timeline

> *May change after seeing the real dataset quality. Every structural surprise becomes
> either a sktime issue or a PR — that is the pattern I have already established.*

| Period | Milestone | Deliverables |
|---|---|---|
| **Weeks 1–2** | Data integration & preprocessing | Real dataset plugged into `SensorDataLoader`, timestamp alignment, sliding window construction, all structural differences documented in `roadblocks.md` |
| **Weeks 3–4** | Detection algorithm benchmarking | Full benchmark suite: Wavelet + XGBoost, `RocketClassifier`, `DrCIF`, `HIVE-COTE 2`, ROCKET + One-Class SVM — same `DetectionBenchmark` harness, same split, same metrics |
| **🏁 Midterm** | Midterm evaluation | Working pipeline on real dataset · ≥3 algorithms benchmarked · PR [#9939](https://github.com/sktime/sktime/pull/9939) merged or under active review · Roadblocks documented · Mentor benchmark review |
| **Weeks 5–6** | sktime extension PRs | `STRAY predict()` fix ([#10100](https://github.com/sktime/sktime/issues/10100)), `WindowedTimeSeriesEvaluator` ([#9894](https://github.com/sktime/sktime/issues/9894)), multivariate detection support for gaps exposed by real dataset |
| **Weeks 7–8** | Explainability & final deliverables | SHAP analysis on best pipeline, final benchmark report, clean reproducible notebooks, ONNX export *(if time permits)* |

---

### 🏆 Final Evaluation Deliverables

| Deliverable | Detail |
|---|---|
| Complete detection pipeline | Documented, tested, reproducible on sponsor dataset |
| Benchmark report | ≥5 algorithms · TPR / FPR / `advance_ms` across all sensor channels |
| sktime PRs | ≥4 merged PRs with tests — detection module meaningfully extended |
| Explainability report | SHAP analysis delivered to sponsor |
| Deployment recommendation | Which algorithm, why, and what trade-offs |