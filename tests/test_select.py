import numpy as np
import pandas as pd

from src.select import boot_diff, boot_r2, conformal, cv_select, r2


def _toy(n_folds=4, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_folds):
        y = rng.normal(size=n_per)
        dates = np.repeat([f"2018-{f+1:02d}-01", f"2017-{f+1:02d}-01"], n_per // 2)
        for cand, noise in [("good", 0.3), ("bad", 1.0)]:
            rows.append(pd.DataFrame({"seed": 0, "fold": f, "row": np.arange(n_per) + f * n_per, "cand": cand,
                                      "y": y, "y_pred": y + rng.normal(scale=noise, size=n_per),
                                      "Date": dates, "Site": [f"S{i % 5}" for i in range(n_per)], "group": "g"}))
    return pd.concat(rows, ignore_index=True)


def test_cv_select_never_uses_the_held_out_fold():
    d = _toy()
    # make 'bad' perfect on fold 0 only: fold 0's choice must still come from folds 1-3 ('good')
    m = (d.cand == "bad") & (d.fold == 0); d.loc[m, "y_pred"] = d.loc[m, "y"]
    kept, chosen = cv_select(d)
    assert set(kept[kept.fold == 0].cand) == {"good"}


def test_bootstrap_interval_brackets_point_and_diff_sign():
    d = _toy()
    good = d[d.cand == "good"]; bad = d[d.cand == "bad"]
    pt, lo, hi = boot_r2(good, "month", n_boot=300)
    assert lo <= pt <= hi and np.isclose(pt, r2(good.y, good.y_pred))
    delta, lo, hi, n = boot_diff(good, bad, "site", n_boot=300)
    assert delta > 0 and lo > 0 and n == len(good)


def test_conformal_coverage_close_to_nominal():
    d = _toy(n_folds=6, n_per=200)
    c = conformal(d[d.cand == "good"], level=0.9)
    assert abs(c.set_index("group").loc["all", "coverage"] - 0.9) < 0.05
