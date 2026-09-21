"""HW1 starter: OLS + Gaussian residual 가정으로 quantile 제출 파일 만들기 (numpy/pandas만 사용)

이 baseline은 "MSE 최소화 = Gaussian noise의 MLE"라는 가정을 그대로 쓴다:
    y | x ~ N(w^T x, sigma^2)   ->   q-quantile = w^T x + sigma * z_q
여러분의 과제는 task별 residual을 진단하고, 더 맞는 noise 모형/likelihood로 이 baseline을 이기는 것이다.

폴더 구조:  data/train_{A,B,C,D}.csv, data/test_{A,B,C,D}.csv, data/sample_submission.csv
실행:       python starter.py   ->  submission.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
TASKS = ["A", "B", "C", "D"]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# 표준정규분포의 quantile z_q (scipy.stats.norm.ppf(QUANTILES))
Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


def design_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """범주형은 one-hot, 수치형은 표준화, 마지막에 bias 항 추가."""
    both = pd.get_dummies(pd.concat([train, test], keys=["tr", "te"]), drop_first=True, dtype=float)
    Xtr, Xte = both.loc["tr"].to_numpy(), both.loc["te"].to_numpy()
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    add_bias = lambda X: np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return add_bias(Xtr), add_bias(Xte)


def main():
    parts = []
    for task in TASKS:
        train = pd.read_csv(DATA / f"train_{task}.csv")
        test = pd.read_csv(DATA / f"test_{task}.csv")
        ytr = train["y"].to_numpy(dtype=float)
        Xtr, Xte = design_matrix(train.drop(columns="y"), test.drop(columns="row_id"))

        w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)      # OLS = Gaussian MLE
        sigma = (ytr - Xtr @ w).std(ddof=1)
        pred = (Xte @ w)[:, None] + sigma * Z[None, :]      # (n_test, 9)
        pred = np.sort(pred, axis=1)                        # quantile은 단조 증가해야 한다

        for j, q in enumerate(QUANTILES):
            parts.append(pd.DataFrame({"id": test["row_id"] + f"_q{int(round(q * 100)):02d}",
                                       "value": pred[:, j]}))

    sub = pd.concat(parts, ignore_index=True)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert set(sub["id"]) == set(sample["id"]), "id가 sample_submission과 일치해야 합니다."
    sub.to_csv("submission.csv", index=False)
    print("saved submission.csv", sub.shape)


if __name__ == "__main__":
    main()
