# Training of the bagging members.

import time

import numpy as np

import base as base_module
from ensemble import HomogeneousEnsemble
from ext_common import INIT_SEED, MAX_N, run_split


def _oob_mask(n_train, idx):
    mask = np.ones(n_train, dtype=bool)
    mask[np.unique(idx)] = False
    return mask


def build_ml_estimators(params):
    params_copy = params.copy()
    model = eval(params_copy.pop("model"), vars(base_module))()
    scaler = eval(params_copy.pop("scaler", "StandardScaler()"), vars(base_module))()
    for k, v in params_copy.items():
        model.set_params(**{k.split("__")[1]: None if v == "None" else v})
    return model, scaler


def ml_members_run(X, y, num_train, params, run, max_n=MAX_N):
    train_idx, test_idx = run_split(len(y), num_train, run)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model, scaler = build_ml_estimators(params)
    ensemble = HomogeneousEnsemble(
        base_model=model,
        base_scaler=scaler,
        strategy="bagging",
        feature_fraction=None,
        random_state=INIT_SEED + run,
    )
    t0 = time.time()
    ensemble.fit_incremental(X_train, y_train, max_n)
    fit_time = time.time() - t0

    rng = np.random.RandomState(INIT_SEED + run)
    boot_idx = np.empty((max_n, num_train), dtype=np.int32)
    for m in range(max_n):
        idx = rng.choice(num_train, num_train, replace=True)
        X_m, y_m = ensemble.sampled_data[m]
        if not (np.array_equal(X_m, X_train[idx]) and np.array_equal(y_m, y_train[idx])):
            raise RuntimeError(f"Bootstrap replay mismatch at member {m}, run {run}")
        boot_idx[m] = idx

    test_pred = ensemble.predict_subset(X_test, max_n, return_all_preds=True).astype(np.float64)

    oob_pred = np.full((max_n, num_train), np.nan, dtype=np.float32)
    for m in range(max_n):
        mask = _oob_mask(num_train, boot_idx[m])
        if mask.any():
            oob_pred[m, mask] = ensemble.models[m].predict(X_train[mask])

    return {
        "train_idx": train_idx.astype(np.int32),
        "y_train": y_train.astype(np.float64),
        "y_test": y_test.astype(np.float64),
        "test_pred": test_pred,
        "boot_idx": boot_idx,
        "oob_pred": oob_pred,
        "member_time": np.full(max_n, fit_time / max_n),
    }


def hinnperf_members_run(evaluator, params, run, max_n=MAX_N, single_pass=True):
    from HINNPERF.runHINNPerf import get_HINNPerf_MRE, get_HINNPerf_preds_fixed_config

    y = evaluator.y
    num_train = evaluator.split
    np.random.seed(INIT_SEED + run)
    rng = np.random.RandomState(INIT_SEED + run)
    indices = np.random.permutation(len(y))
    training_index = indices[:num_train]
    testing_index = indices[num_train:]

    train_index_samples = []
    for _ in range(max_n):
        train_index_samples.append(rng.choice(training_index, size=len(training_index), replace=True))

    feat_idx = np.arange(evaluator.X.shape[1])
    config_with_lists = {k: [v] for k, v in params.items()}
    config_with_lists["input_dim"] = [len(feat_idx)]
    data = evaluator.whole_data[:, list(feat_idx) + [-1]]

    pos_of = {int(r): p for p, r in enumerate(training_index)}
    n_test = len(testing_index)
    test_pred = np.empty((max_n, n_test), dtype=np.float64)
    oob_pred = np.full((max_n, num_train), np.nan, dtype=np.float32)
    boot_idx = np.empty((max_n, num_train), dtype=np.int32)
    member_time = np.empty(max_n)

    for m in range(max_n):
        sampled = train_index_samples[m]
        boot_idx[m] = [pos_of[int(r)] for r in sampled]
        oob_pos = np.where(_oob_mask(num_train, boot_idx[m]))[0]
        t0 = time.time()
        if single_pass:
            eval_index = np.concatenate([testing_index, training_index[oob_pos]])
            pred, _ = get_HINNPerf_preds_fixed_config(data, sampled, eval_index, config_with_lists)
            test_pred[m] = pred[:n_test]
            oob_pred[m, oob_pos] = pred[n_test:]
        else:
            _, pred, _, _, _ = get_HINNPerf_MRE([data, sampled, testing_index, False, config_with_lists])
            test_pred[m] = np.asarray(pred).ravel()
        member_time[m] = time.time() - t0

    return {
        "train_idx": training_index.astype(np.int32),
        "y_train": y[training_index].astype(np.float64),
        "y_test": y[testing_index].astype(np.float64),
        "test_pred": test_pred,
        "boot_idx": boot_idx,
        "oob_pred": oob_pred,
        "member_time": member_time,
    }
