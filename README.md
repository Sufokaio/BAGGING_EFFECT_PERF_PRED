# Understanding the Effect of Bagging on Performance Prediction in Configurable Software Systems

This repository contains the source code, the data used, and the raw
results for each experiment from the corresponding paper. It extends the
code and results of the conference version:
https://github.com/Sufokaio/BAGGING_PERF_PRED_CONF

## Documents

### /data

This folder contains the data used for all experiments. There is one CSV
file for each subject system (eight in total). Each column represents a
configuration option, and the last column is the performance value to be
predicted.

### /results_original
This folder contains the results of `main.py` (conference version) for every run of each experiment across all evaluation scenarios. Future executions will save results in a separate `/results` directory.

The format of the results is as follows:

One folder per dataset, and within each dataset folder, one folder per
sample size (five per dataset).

In each sample size subfolder, there are 4 files named
`{Model-Name}_single_metrics.json`. These files represent the results
of the single variants of the corresponding model type (4 model
types).

Each file has the following format:

A JSON file containing one list with one dict entry. The dict
represents the best variant of the corresponding model with its
corresponding hyperparameters (visible in the `"Params"` entry). The dict also contains the mean
values across 30 runs for each evaluation metric (`Mean_MAE`,
`Mean_MRE`, `Mean_MBRE`, `Mean_MIBRE`). Additionally, `"Metrics"` and
`"Runs"` contain the evaluation metric values of each run and the raw
predictions of each run for each element in the corresponding test sets.

The ensemble files are named:

-   `{Model-Name}_bagging_N{k}_Fall_{combination-rule}.json`

with `k` representing the number of single learners and `combination-rule` representing the combination rule.

These files contain a single dict with the mean values across 30 runs
for each evaluation metric (`Mean_MAE`, `Mean_MRE`, `Mean_MBRE`,
`Mean_MIBRE`). `"Metrics"` contains the evaluation metric
values of each run.

The single results of SVR, RT, KRR and HINNPerf are used in this paper
(copied to `/results_ext`); the other files are not.

### /results_ext

The results of the paper, with the same layout as `/results_original`.
In each sample size subfolder:

-   `{Model-Name}_single_metrics.json` and `baseline.txt`: single
    variants of the 6 learners (copied from `/results_original` for
    SVR, RT, KRR and HINNPerf, computed by `main_ext.py` for LR and KNN).
-   `{Model-Name}_bagging_N{k}_{mean|median|irwm}.json`, k = 1..10:
    bagging ensembles of the first k members, same format as above.
-   `{Model-Name}_bagging_members.npz`: test and out-of-bag predictions
    of the 10 members of every run.
-   `{Model-Name}_bagging_*.parquet`: out-of-bag metrics, diversity
    measures and member metrics per run.

### /analysis_ext/output

Outputs used in the paper (`.pdf`/`.png` figures, `.tex` tables and the
CSV files with all values):

-   RQ1: `T1v1_best_cluster_*`, `T1v1_per_system_*` (CSV:
    `T1_best_cluster_*`, `T1v1_per_system_*`)
-   RQ2: `F3_k_best_cluster_*` (CSV: `T_k_best_cluster_*`)
-   RQ3: `T_rules_*` (CSV: `T_rules_*`, `T_rules_k_*`)
-   RQ4: `F6_bias_variance_*` (CSV: `T4d_variance_vs_k_*`)
-   RQ5: `T3_diversity_*`

### /DEEPPERF and /HINNPERF

These directories contain the implementations of both state-of-the-art
approaches, using their publicly available resources:
https://github.com/DeepPerf/DeepPerf and https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh

### ML Techniques

Each ML technique has its own class based on the `Base` class in
`base.py`. It contains the general logic described in the paper for
constructing the single variants.

### main.py

This file contains the general logic for evaluating the single variants
and the ensemble variants of the conference version.

### main_ext.py, ext_members.py and postprocess_ext.py

`main_ext.py` computes the single variants of LR and KNN and trains 10
bagging members per run for the 6 learners (`ext_members.py`), with the
seeds, splits and hyperparameters of the single variants.
`postprocess_ext.py` combines the first k members (k = 1..10) with MEAN,
MEDIAN and IRWM and computes the out-of-bag metrics and the diversity
measures.

### analysis_ext/run_all.py

Computes the Scott-Knott tests, Borda counts, prediction variance and
diversity for k = 2..10 and writes `/analysis_ext/output`.

## Prerequisites and Installation

Use `requirements.txt` and follow any runtime messages. The experiments were executed with Python 3.9.x.

## How to Run

Comment out the datasets and models you want to exclude. Otherwise, it
will run all experiments by default, as described in the paper.

To reproduce the experiments of the conference version:

    python main.py --model baseline
    python main.py

Note: Results from the next execution will be saved in `/results` by default.

To reproduce the experiments and the analysis of this paper:

    python main_ext.py --stage copy
    python main_ext.py --stage members
    python postprocess_ext.py
    python analysis_ext/run_all.py

`main_ext.py --stage copy` copies the single results of SVR, RT, KRR and
HINNPerf and `baseline.txt` from `/results_original`.
