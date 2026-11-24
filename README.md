Engine Anomaly Detection Across a Vessel Fleet
Unsupervised anomaly detection on engine sensor data, comparing a statistical baseline (IQR) with two machine-learning methods (One-Class SVM and Isolation Forest), tuned to a realistic 1–5% anomaly rate.

University of Cambridge, Institute of Continuing Education — Data Science, Machine Learning and AI programme, April 2025.

Problem
A shipping fleet has experienced engine failures. Using sensor readings alone (no fault labels), identify the engines whose operating conditions are anomalous, and recommend a detection method that could be used routinely.

Data
Records	19,535 engines
Features	Engine rpm, lube-oil pressure, fuel pressure, coolant pressure, lube-oil temperature, coolant temperature
Quality	No missing values, no duplicate rows
Labels	None — unsupervised task
Structure	Condition snapshots (one row per engine), not a time series
Method
Exploratory analysis — descriptive statistics, mean vs median per feature, correlation heatmap, histograms with fitted Gaussian curves. Shapiro–Wilk tests showed no feature is normally distributed, which motivated distribution-free and ML methods.
Statistical baseline (IQR) — per-feature outlier bounds (Q1 − 1.5·IQR, Q3 + 1.5·IQR) and a binary flag per feature. Most outliers were in lube-oil temperature (2,617) and fuel pressure (1,135). Requiring ≥ 3 simultaneous flags gives 2.16% of engines; ≥ 2 flags gives 23.7%.
Scaling and PCA — standardised features; 2-component PCA for visualisation, with a loadings heatmap (engine rpm dominates the first components).
One-Class SVM (RBF kernel) — tuned nu and gamma; final setting (nu = 0.03) flags 586 engines (3.0%). Decision-function contours are plotted in PCA space.
Isolation Forest — contamination tested at 5%, 3% and 2.5%; final model flags 489 engines (2.5%), within the target range.
Results
Method	Engines flagged	Share
IQR, ≥ 3 features in outlier range	422	2.16%
One-Class SVM (nu = 0.03)	586	3.0%
Isolation Forest (contamination 2.5%)	489	2.5%
Recommendation: Isolation Forest as the primary detector — stable, scalable and interpretable via anomaly scores — with IQR retained as a transparent baseline. Anomalous engines were associated with consistently low coolant or lubrication pressure.

Limitations and next steps
No ground truth. Without fault labels the anomalies cannot be scored for precision or recall. Next step: validate flagged engines against maintenance logs or engineer review, and measure agreement between the three methods.
Feature selection. The RFE step in the notebook was fitted against a placeholder target, so its feature ranking is not informative; selection should instead use PCA loadings or domain knowledge.
Explained-variance figure. The final PCA cell includes the added outlier-flag columns, which inflates PC1's explained variance; the reported 82.8% should not be relied on.
No time dimension. With timestamped readings this would extend naturally to condition monitoring: rolling baselines per engine, drift detection and predictive maintenance.
Repository contents
Path	Contents
notebooks/	Analysis notebook
reports/	Written report and appendix (distributions, violin plots)
figures/	Exported notebook figures and report pages
Tools
Python · pandas · NumPy · SciPy (Shapiro–Wilk, IQR) · scikit-learn (StandardScaler, PCA, OneClassSVM, IsolationForest, RFE) · seaborn · matplotlib
