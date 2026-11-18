# Runs the extended experiments: single learners and bagging members.

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback

import numpy as np

from ext_common import (COPIED_SINGLES, DATA_DIR, DATASET_REGISTRY, INIT_SEED, LEARNERS, MAX_N, NUM_RUNS,
                        NEW_SINGLES, RESULTS_EXT, RESULTS_ORIG, load_top1_params, log_line, log_time,
                        scen_dir, scenarios, size_level)

cpu_count = os.cpu_count()


def learner_class(model_name):
    if model_name == "SVR":
        from SVRWrapper import SVRWrapper
        return SVRWrapper
    if model_name == "RT":
        from RT import RT
        return RT
    if model_name == "KRR":
        from KRR import KRR
        return KRR
    if model_name == "LR":
        from LR import LR
        return LR
    if model_name == "KNN":
        from KNN import KNN
        return KNN
    if model_name == "HINNPerf":
        from rHINNPERF import HINNPerf
        return HINNPerf
    raise ValueError(model_name)


def make_evaluator(model_name, dataset, s):
    cls = learner_class(model_name)
    results_root = RESULTS_EXT

    class ExtEvaluator(cls):
        def load_baseline(self):
            metrics = {}
            with open(os.path.join(results_root, self.dataset_name, str(self.split), "baseline.txt")) as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":")
                        metrics[key.strip()] = float(value.strip())
            return metrics["MAEp0"], metrics["Sp0"], metrics["SA_5"]

    return ExtEvaluator(dataset, data_path=DATA_DIR, split_mode=False, split=int(s), num_runs=NUM_RUNS)


def copy_originals():
    n = 0
    for ds, s in scenarios():
        src, dst = scen_dir(ds, s, RESULTS_ORIG), scen_dir(ds, s)
        os.makedirs(dst, exist_ok=True)
        for name in ["baseline.txt"] + [f"{m}_single_metrics.json" for m in COPIED_SINGLES]:
            if not os.path.exists(os.path.join(dst, name)):
                shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
                n += 1
    print(f"Copied {n} files from {RESULTS_ORIG}")


def ensure_single(model_name, dataset, s):
    path = os.path.join(scen_dir(dataset, s), f"{model_name}_single_metrics.json")
    if os.path.exists(path):
        return False
    if model_name not in NEW_SINGLES:
        raise FileNotFoundError(f"{path} missing; run --stage copy first")
    t0 = time.time()
    evaluator = make_evaluator(model_name, dataset, s)
    metrics_results = evaluator.run_experiment()
    for entry in metrics_results:
        for key, value in entry.get("Metrics", {}).items():
            if isinstance(value, np.ndarray):
                entry["Metrics"][key] = value.tolist()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics_results, f, indent=2)
    os.replace(tmp, path)
    log_time(f"Completed SINGLE {model_name} on {dataset} sample {s} in {time.time() - t0:.2f} seconds")
    return True


def members_path(dataset, s, model_name):
    return os.path.join(scen_dir(dataset, s), f"{model_name}_bagging_members.npz")


def ckpt_dir(dataset, s, model_name):
    return os.path.join(scen_dir(dataset, s), "_ckpt", model_name)


def ckpt_path(dataset, s, model_name, run):
    return os.path.join(ckpt_dir(dataset, s, model_name), f"run_{run:02d}.npz")


def merge_runs(dataset, s, model_name, params, extra_meta=None):
    runs = [np.load(ckpt_path(dataset, s, model_name, r)) for r in range(NUM_RUNS)]
    keys = ["train_idx", "y_train", "y_test", "test_pred", "boot_idx", "oob_pred", "member_time"]
    stacked = {k: np.stack([r[k] for r in runs]) for k in keys}
    meta = {
        "learner": model_name, "dataset": dataset, "sample_size": int(s),
        "size_level": size_level(dataset, s), "num_runs": NUM_RUNS, "max_n": MAX_N,
        "seed": f"{INIT_SEED}+run", "strategy": "bagging", "params": params,
        "hinnperf_single_pass": model_name == "HINNPerf",
    }
    meta.update(extra_meta or {})
    out = members_path(dataset, s, model_name)
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp,
        y_test=stacked["y_test"],
        member_test_pred=stacked["test_pred"],
        y_train=stacked["y_train"],
        train_idx=stacked["train_idx"],
        boot_idx=stacked["boot_idx"],
        oob_pred=stacked["oob_pred"].astype(np.float32),
        member_time=stacked["member_time"],
        meta=np.array(json.dumps(meta)),
    )
    for r in runs:
        r.close()
    os.replace(tmp, out)
    shutil.rmtree(ckpt_dir(dataset, s, model_name), ignore_errors=True)
    return out


def run_members_task(model_name, dataset, s):
    s = int(s)
    if os.path.exists(members_path(dataset, s, model_name)):
        return f"Skipped {model_name} {dataset} {s} (exists)"
    t0 = time.time()
    print(f"PID {os.getpid()} starts {model_name} on {dataset} sample {s}", flush=True)
    ensure_single(model_name, dataset, s)
    params = load_top1_params(dataset, s, model_name)
    evaluator = make_evaluator(model_name, dataset, s)
    os.makedirs(ckpt_dir(dataset, s, model_name), exist_ok=True)

    if model_name == "HINNPerf":
        from ext_members import hinnperf_members_run
        fn = lambda run: hinnperf_members_run(evaluator, params, run)
    else:
        from ext_members import ml_members_run
        fn = lambda run: ml_members_run(evaluator.X, evaluator.y, s, params, run)

    n_done = 0
    for run in range(NUM_RUNS):
        path = ckpt_path(dataset, s, model_name, run)
        if os.path.exists(path):
            continue
        t_run = time.time()
        res = fn(run)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, run=run, **res)
        os.replace(tmp, path)
        n_done += 1
        if model_name == "HINNPerf":
            print(f"{model_name} {dataset} {s} run {run + 1}/{NUM_RUNS} in {time.time() - t_run:.1f}s "
                  f"({np.mean(res['member_time']):.2f}s/member)", flush=True)

    merge_runs(dataset, s, model_name, params)
    elapsed = time.time() - t0
    log_time(f"Completed MEMBERS {model_name} on {dataset} sample {s} ({n_done} runs) in {elapsed:.2f} seconds")
    log_line("progress.log", f"{model_name}\t{dataset}\t{s}\tS{size_level(dataset, s)}\t{elapsed:.1f}s\t{n_done} runs")
    return f"Completed {model_name} {dataset} {s} in {elapsed:.1f}s"


def run_task(task):
    try:
        return run_members_task(*task)
    except Exception:
        traceback.print_exc()
        log_line("progress.log", f"FAILED\t{task}")
        return f"FAILED {task}"


def worker_init(tf_threads):
    if tf_threads:
        limit_threads(tf_threads)


def limit_threads(n):
    from threadpoolctl import threadpool_limits
    threadpool_limits(n)
    import tensorflow.compat.v1 as tf
    import HINNPERF.HINNPerf_models as hm

    def init_session(self):
        cfg = tf.ConfigProto(intra_op_parallelism_threads=n, inter_op_parallelism_threads=n)
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())

    hm.MLPHierarchicalModel.init_session = init_session


def build_tasks(models, datasets, size, levels):
    tasks = []
    for ds, s in scenarios(datasets, size, levels):
        for m in sorted(models, key=lambda m: (m != "HINNPerf", LEARNERS.index(m))):
            tasks.append((m, ds, s))
    return sorted(tasks, key=lambda t: (size_level(t[1], t[2]), t[0] != "HINNPerf", t[2], t[1]))


def main():
    parser = argparse.ArgumentParser(description="Extended bagging experiments")
    parser.add_argument("--stage", type=str, default="members", choices=["copy", "members", "all"])
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--levels", type=str, default=None, help="Size levels, e.g. 1 or 1,2,3")
    parser.add_argument("--processes", type=int, default=max(1, cpu_count - 3))
    parser.add_argument("--threads", type=int, default=0, help="Cap BLAS/TF threads per worker (0 = default)")
    args = parser.parse_args()

    if args.stage in ("copy", "all"):
        copy_originals()
        if args.stage == "copy":
            return

    datasets = None if args.dataset == "all" else args.dataset.split(",")
    models = LEARNERS if args.model == "all" else args.model.split(",")
    levels = None if args.levels is None else [int(x) for x in args.levels.split(",")]
    tasks = build_tasks(models, datasets, args.size, levels)
    print(f"{len(tasks)} tasks, {args.processes} processes", flush=True)
    log_line("progress.log", f"START {len(tasks)} tasks, processes={args.processes}, threads={args.threads or 'default'}")

    t0 = time.time()
    if args.processes == 1 or len(tasks) == 1:
        worker_init(args.threads)
        results = [run_task(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=min(args.processes, len(tasks)), initializer=worker_init,
                                  initargs=(args.threads,)) as pool:
            results = []
            for res in pool.imap_unordered(run_task, tasks, chunksize=1):
                print(res, flush=True)
                results.append(res)
    log_line("progress.log", f"END {len(tasks)} tasks in {time.time() - t0:.1f}s")
    print(f"All tasks finished in {time.time() - t0:.2f} seconds")
    if any(r.startswith("FAILED") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
