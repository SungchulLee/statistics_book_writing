r"""실제 자료는 얼마나 「정규」인가 — 세어 본다.

사이킷런에 들어 있는 자료 다섯 벌의 수치 열을 모아 샤피로–윌크 검정을 건다.
그리고 피어슨이 $1920$ 년에 적은 「뒤틀어 정규곡선에 밀어넣기」를 실제로 해 본다.

① 열 $62$ 개 가운데 통과하는 것이 몇 개인가.
② 같은 열을 표본 크기만 줄여 가며 재면 통과율이 어떻게 변하는가.
③ 거듭제곱변환으로 뒤틀면 통과율이 얼마나 오르는가 — 그리고 끝까지 가는가.

만드는 파일: docs/ch14/descriptive_statistics/img/how_normal.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from scipy import stats                                  # noqa: E402
from sklearn.datasets import (load_breast_cancer, load_diabetes,  # noqa: E402
                              load_iris, load_linnerud, load_wine)
from sklearn.preprocessing import PowerTransformer       # noqa: E402

RNG = np.random.default_rng(20261022)
ALPHA = 0.05
NS = np.array([10, 20, 50, 100, 200, 400])
REP = 40

SETS = [("붓꽃", load_iris), ("포도주", load_wine), ("유방암", load_breast_cancer),
        ("당뇨", load_diabetes), ("체력", load_linnerud)]

cols, where = [], []
for name, load in SETS:
    d = load()
    for j in range(d.data.shape[1]):
        if len(np.unique(d.data[:, j])) > 10:    # 사실상 범주인 열은 뺀다
            cols.append(d.data[:, j]); where.append(name)
    if np.ndim(d.target) > 1:                    # 체력 자료는 표적도 수치다
        for j in range(d.target.shape[1]):
            cols.append(d.target[:, j]); where.append(name)
print(f"자료 {len(SETS)} 벌에서 수치 열 {len(cols)} 개 "
      f"(표본 {min(map(len, cols))}–{max(map(len, cols))})")

# === ① 있는 그대로 ===================================================
p_raw = np.array([stats.shapiro(v).pvalue for v in cols])
print(f"\n① 샤피로–윌크 (알파 {ALPHA})")
print(f"   통과 {100 * (p_raw > ALPHA).mean():.1f}% "
      f"({(p_raw > ALPHA).sum()}/{len(p_raw)})")
print(f"   p < 0.001 인 열 {100 * (p_raw < 0.001).mean():.1f}%")
for name, _ in SETS:
    m = np.array(where) == name
    print(f"     {name:>4}: {int((p_raw[m] > ALPHA).sum())}/{int(m.sum())}")

# === ② 표본 크기만 줄이면 ============================================
print(f"\n② 같은 열을 부분표본으로 재면 (열마다 {REP} 번)")
rate = []
for n in NS:
    ok = []
    for v in cols:
        if len(v) < n:
            continue
        ok += [stats.shapiro(RNG.choice(v, int(n), replace=False)).pvalue > ALPHA
               for _ in range(REP)]
    rate.append(float(np.mean(ok)))
    usable = sum(1 for v in cols if len(v) >= n)
    print(f"   n = {n:>4} : 통과 {100 * rate[-1]:>5.1f}%  (쓸 수 있는 열 {usable})")
rate = np.array(rate)

# === ③ 뒤틀어 보면 ===================================================
p_tr, k_raw, k_tr = [], [], []
for v in cols:
    w = PowerTransformer(method="yeo-johnson").fit_transform(
        v.reshape(-1, 1)).ravel()
    p_tr.append(stats.shapiro(w).pvalue)
    k_raw.append(stats.kurtosis(v)); k_tr.append(stats.kurtosis(w))
p_tr = np.array(p_tr); k_raw = np.array(k_raw); k_tr = np.array(k_tr)
print(f"\n③ 여-존슨 거듭제곱변환을 건 뒤")
print(f"   통과 {100 * (p_raw > ALPHA).mean():.1f}% -> "
      f"{100 * (p_tr > ALPHA).mean():.1f}%")
print(f"   그래도 p < 0.001 인 열 {100 * (p_tr < 0.001).mean():.1f}%")
print(f"   초과첨도 중앙값 {np.median(k_raw):+.3f} -> {np.median(k_tr):+.3f}")
print(f"   |초과첨도| > 1 인 열 {(np.abs(k_raw) > 1).sum()} -> "
      f"{(np.abs(k_tr) > 1).sum()}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
FLOOR = 1e-16
order = np.argsort(p_raw)
xs = np.arange(len(cols))

ax1.semilogy(xs, np.maximum(p_raw[order], FLOOR), "o", ms=6, color="#d62728")
ax1.axhline(ALPHA, color="black", lw=1.5, ls="--")
ax1.text(1, ALPHA * 1.6, f"$\\alpha = {ALPHA}$", fontsize=10.5)
n_pass = int((p_raw > ALPHA).sum())
ax1.axvspan(len(cols) - n_pass - 0.5, len(cols) - 0.5, color="#2ca02c",
            alpha=0.13)
ax1.text(len(cols) - n_pass - 1.2, 1e-6,
         f"통과 {n_pass}/{len(cols)}\n(${100 * n_pass / len(cols):.1f}\\%$)",
         ha="right", fontsize=11.5, color="#2ca02c")
ax1.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
ax1.set_yticklabels(["$10^{-16}$", "$10^{-12}$", "$10^{-8}$", "$10^{-4}$", "$1$"])
ax1.minorticks_off()
ax1.set_xlabel(f"수치 열 {len(cols)} 개 ($p$ 값 순)")
ax1.set_ylabel("샤피로–윌크 $p$ 값")
ax1.set_title("① 실제 자료는 거의 정규가 아니다", fontsize=12)
ax1.grid(alpha=0.25, which="major")

ax2.semilogx(NS, 100 * rate, "o-", color="#1f77b4", lw=2.8, ms=9)
ax2.axhline(100 * (p_raw > ALPHA).mean(), color="#d62728", lw=1.8, ls="--",
            label=f"전부 쓸 때 ${100 * (p_raw > ALPHA).mean():.1f}\\%$")
for n in (10, 400):
    i = int(np.where(NS == n)[0][0])
    ax2.annotate(f"$n = {n}$ 이면\n${100 * rate[i]:.0f}\\%$", xy=(n, 100 * rate[i]),
                 xytext=(n * (1.6 if n == 10 else 0.25),
                         100 * rate[i] + (-16 if n == 10 else 16)),
                 fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v}$" for v in NS])
ax2.minorticks_off()
ax2.set_xlabel("부분표본의 크기 $n$")
ax2.set_ylabel("「정규다」로 판정되는 비율 ($\\%$)")
ax2.set_ylim(0, 92)
ax2.set_title("② 같은 자료, 다른 답", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.scatter(np.maximum(p_raw, FLOOR), np.maximum(p_tr, FLOOR), s=34,
            alpha=0.75, color="#1f77b4", edgecolors="none")
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.plot([FLOOR, 1], [FLOOR, 1], color="#999999", lw=1.2, ls=":")
ax3.axhline(ALPHA, color="black", lw=1.4, ls="--")
ax3.axvline(ALPHA, color="black", lw=1.4, ls="--")
ax3.set_xticks([1e-16, 1e-11, 1e-6, 1e-2, 1])
ax3.set_xticklabels(["$10^{-16}$", "$10^{-11}$", "$10^{-6}$", "$0.01$", "$1$"])
ax3.set_yticks([1e-16, 1e-11, 1e-6, 1e-2, 1])
ax3.set_yticklabels(["$10^{-16}$", "$10^{-11}$", "$10^{-6}$", "$0.01$", "$1$"])
ax3.minorticks_off()
ax3.text(0.97, 0.06,
         f"변환 뒤 통과 ${100 * (p_tr > ALPHA).mean():.1f}\\%$\n"
         f"그래도 $p < 0.001$ 인 열 ${100 * (p_tr < 0.001).mean():.0f}\\%$",
         transform=ax3.transAxes, ha="right", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax3.set_xlabel("변환 전 $p$ 값")
ax3.set_ylabel("여-존슨 변환 뒤 $p$ 값")
ax3.set_title("③ 뒤틀어도 끝까지 가지 않는다", fontsize=12)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch14/descriptive_statistics/img/how_normal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
