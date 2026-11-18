# Runs the analysis: python analysis_ext/run_all.py [--rebuild]

import argparse
import glob
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config as cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="Rebuild the caches from results_ext")
    a = p.parse_args()
    if a.rebuild:
        for f in glob.glob(os.path.join(cfg.CACHE_DIR, "*.parquet")):
            os.remove(f)

    from common import setup_style
    import analyses as an
    from loaders import load

    setup_style()
    data = load()
    for d in [cfg.FIG_DIR, cfg.TAB_DIR, cfg.CSV_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    sc = data.scopes["all"]

    print("RQ1")
    an.rq1_t1(data, sc)
    m = an.improvement_rows(data, sc)
    print("RQ2")
    an.rq2(data, sc)
    print("RQ3")
    an.rq3(data, sc)
    print("RQ4")
    an.rq4(data, sc)
    print("RQ5")
    an.rq5(data, sc, m)


if __name__ == "__main__":
    main()
