# agri4-sktime-benchmark

Surrogate benchmarking project for Agriculture 4.0 
applied project using sktime.

![agriculture](images.jpg)

## Dataset
IoT Environmental Sensor Telemetry Data (405k rows, 3 devices)

## What This Demonstrates
- Loading multivariate IoT sensor data into sktime Panel format
- Change point detection using ClaSPSegmentation
- Motion event correlation with temperature anomalies

## Roadblocks Found in sktime
See roadblocks.md for real issues encountered during development.

## Results
![Change Points](data/changepoint_detection.png)
![Motion vs Change Points](data/motion_vs_changepoints.png)

