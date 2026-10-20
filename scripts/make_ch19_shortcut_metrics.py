r"""같은 모형, 같은 코드, 평가 자료만 바꾸면 — 지표가 전부 무너진다.

병의 신호는 약하고(효과크기 $0.3$), 자료를 모은 **출처**가 라벨과 붙어 있으면
모형은 병이 아니라 출처를 배운다. 그 모형의 ROC·정확도·$F_1$ 은 모두 훌륭해 보인다.

① 같은 모형의 ROC 두 개 — 평가 자료만 다르다.
② 다섯 지표가 한꺼번에 무너진다.
③ 출처와 라벨이 얼마나 붙어 있느냐에 따른 두 AUC.

만드는 파일: docs/ch19/evaluation/img/shortcut_metrics.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

RNG = np.random.default_rng(20261020)
N = 6000
D = 0.3  # 병이 실제로 만드는 차이 — 약하다
GAP = 2.4  # 촬영 장비·연령대가 만드는 차이 — 크다


def make(n, p_source, rng):
    """p_source = P(출처 B | 환자군) = P(출처 A | 대조군). 1 이면 완전히 붙어 있다."""
    y = rng.integers(0, 2, n)
    sig = rng.normal(D * y, 1.0)  # 병이 만드는 신호
    s = np.where(
        rng.random(n) < p_source, y, 1 - y
    )  # 출처 표지 (0/1)
    style = rng.normal(GAP * s, 1.0)  # 출처가 만드는 겉모습
    return np.column_stack([sig, style]), y


def metrics(model, X, y):
    pr = model.predict_proba(X)[:, 1]
    yh = (pr >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, pr),
        "정확도": accuracy_score(y, yh),
        "정밀도": precision_score(y, yh, zero_division=0),
        "재현율": recall_score(y, yh),
        "$F_1$": f1_score(y, yh),
    }, pr


Xtr, ytr = make(N, 1.0, RNG)
clf = LogisticRegression().fit(Xtr, ytr)

Xa, ya = make(N, 1.0, RNG)  # 같은 방식으로 더 모은 자료
Xb, yb = make(N, 0.5, RNG)  # 출처를 섞은 자료
ma, pra = metrics(clf, Xa, ya)
mb, prb = metrics(clf, Xb, yb)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

for (X, y, pr, lab, c) in [
    (Xa, ya, pra, "같은 방식으로 모은 자료", "#1f77b4"),
    (Xb, yb, prb, "출처를 섞은 자료", "#d62728"),
]:
    fpr, tpr, _ = roc_curve(y, pr)
    ax1.plot(fpr, tpr, color=c, lw=2.6, label=f"{lab}  AUC $= {roc_auc_score(y, pr):.3f}$")
ax1.plot([0, 1], [0, 1], color="black", lw=1.1, ls="--", label="찍기  AUC $= 0.5$")
ax1.set_xlabel("거짓양성률")
ax1.set_ylabel("참양성률")
ax1.set_title("① 같은 모형의 ROC 두 개", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(alpha=0.25)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.set_aspect("equal")

keys = list(ma.keys())
xs = np.arange(len(keys))
w = 0.36
ax2.bar(xs - w / 2, [ma[k] for k in keys], w, color="#1f77b4", label="같은 방식으로 모은 자료")
ax2.bar(xs + w / 2, [mb[k] for k in keys], w, color="#d62728", label="출처를 섞은 자료")
for x, k in zip(xs, keys):
    ax2.text(x - w / 2, ma[k] + 0.015, f"{ma[k]:.2f}", ha="center", fontsize=9.5)
    ax2.text(x + w / 2, mb[k] + 0.015, f"{mb[k]:.2f}", ha="center", fontsize=9.5)
ax2.axhline(0.5, color="#555555", lw=1.1, ls="--")
ax2.text(-0.42, 0.515, "찍기", fontsize=9.5, color="#555555")
ax2.set_xticks(xs)
ax2.set_xticklabels(keys, fontsize=11)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("값")
ax2.set_title("② 다섯 지표가 한꺼번에", fontsize=12)
ax2.legend(loc="lower left", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

ps = np.linspace(0.5, 1.0, 21)
aa, bb = [], []
for p in ps:
    Xt, yt = make(N, p, RNG)
    m = LogisticRegression().fit(Xt, yt)
    Xa2, ya2 = make(N, p, RNG)
    Xb2, yb2 = make(N, 0.5, RNG)
    aa.append(roc_auc_score(ya2, m.predict_proba(Xa2)[:, 1]))
    bb.append(roc_auc_score(yb2, m.predict_proba(Xb2)[:, 1]))
ax3.plot(ps * 100, aa, "o-", color="#1f77b4", lw=2.3, ms=5, label="같은 방식으로 모은 자료")
ax3.plot(ps * 100, bb, "s-", color="#d62728", lw=2.3, ms=5, label="출처를 섞은 자료")
ax3.axhline(0.5, color="#555555", lw=1.0, ls="--")
ax3.set_xlabel("환자군이 한 출처에서 온 비율 ($\\%$)")
ax3.set_ylabel("AUC")
ax3.set_title("③ 붙어 있을수록 좋아 보인다", fontsize=12)
ax3.set_ylim(0.45, 1.02)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch19/evaluation/img/shortcut_metrics.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n병이 만드는 차이 {D}, 출처가 만드는 차이 {GAP}")
print(f"학습된 계수: 신호 {clf.coef_[0][0]:+.3f},  출처의 겉모습 {clf.coef_[0][1]:+.3f}")
print(f"  → 출처 쪽 계수가 {abs(clf.coef_[0][1] / clf.coef_[0][0]):.1f} 배")
print(f"\n{'지표':>8}{'같은 방식':>12}{'출처를 섞으면':>14}")
for k in keys:
    print(f"{k.replace('$F_1$', 'F1'):>8}{ma[k]:>12.4f}{mb[k]:>14.4f}")
from scipy import stats  # noqa: E402

print(f"\n신호만으로 낼 수 있는 AUC 의 이론값: Φ({D}/√2) = {stats.norm.cdf(D / np.sqrt(2)):.4f}")
print("\n③ 붙어 있는 정도별")
for p, a, b in zip(ps[::4], aa[::4], bb[::4]):
    print(f"  {p * 100:>5.0f}%:  같은 방식 {a:.4f}   섞으면 {b:.4f}")
