## 🌱 Why This Project?

### *"Embedded AI for Predictive Sensor Systems in Agriculture 4.0"*

The description of this project resonated with me immediately — because it is asking for
exactly the kind of work I have already been doing independently as preparation, not as a
checkbox exercise, but because I found it **genuinely interesting**.

---

The core challenge — detecting foreign object events in a continuous multi-sensor stream,
distinguishing them from thousands of hours of normal operating conditions — is a
**hard real-world problem**. It is not a toy benchmark.

- The sensors are fast (**millisecond granularity**)
- The events are **rare**
- The cost asymmetry is technically meaningful:

| Event | Cost |
|---|---|
| False Negative (missed detection) | Machine damage |
| False Positive (unnecessary stop) | Operational loss |

> That asymmetry makes it both **technically interesting** and **practically meaningful.**

---

### 🎯 This Project is About Two Things

1. Use **sktime's mature parts** — classifiers, transformations, pipelines — for the
   agriculture sensor pipeline

2. **Build the missing detection-specific infrastructure** — metrics, benchmarking,
   multivariate detectors — as extended capabilities of sktime