r"""검정 모음을 다 돌려 가장 작은 $p$ 를 고르면 어떻게 되는가.

이 쪽은 같은 물음에 쓸 수 있는 검정을 여럿 늘어놓는다. 그러면 넷을 다
돌려 보고 싶어진다. 귀무가설이 참일 때 실제 크기를 잰다.

① 네 검정의 $p$ 값은 서로 얼마나 닮았는가 — 순열과 $t$ 는 상관 $0.999$ 다.
② 가장 작은 것을 고르면 크기가 얼마가 되는가. 독립이라 가정한 값보다
   훨씬 작다 — 검정들이 서로 닮았기 때문이다.
③ 그래서 본페로니는 **지나치게** 자른다. 올바른 문턱은 따로 있다.

만드는 파일: docs/ch16/foundations/img/test_shopping.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
N, R, B, ALPHA = 20, 20000, 999, 0.05
NAMES = ["부호", "윌콕슨 부호순위", "대응 순열", "일표본 $t$"]
SHORT = ["부호", "윌콕슨", "순열", "$t$"]


def pvals(x, r):
    """같은 자료에 네 검정을 건다 (모두 양측, H0: 중심이 0)."""
    p_sign = stats.binomtest(int(np.sum(x > 0)), len(x), 0.5).pvalue
    p_wil = stats.wilcoxon(x).pvalue
    sg = np.abs(x) * np.where(r.random((B, len(x))) < 0.5, -1, 1)
    p_per = (np.sum(np.abs(sg.mean(axis=1)) >= abs(x.mean())) + 1) / (B + 1)
    p_t = stats.ttest_1samp(x, 0).pvalue
    return np.array([p_sign, p_wil, p_per, p_t])


POP = {"정규": lambda: RNG.normal(0, 1, N),
       "$t(3)$ (두꺼운 꼬리)": lambda: RNG.standard_t(3, N)}
P = {}
for nm, gen in POP.items():
    P[nm] = np.array([pvals(gen(), RNG) for _ in range(R)])

print(f"① $p$ 값 사이의 상관 (귀무가설 아래, 정규자료, n = {N}, {R:,} 번)")
C = np.corrcoef(P["정규"].T)
print("              " + "".join(f"{s:>10}" for s in SHORT))
for k, s in enumerate(SHORT):
    print(f"{s:>14}" + "".join(f"{C[k, j]:>10.3f}" for j in range(4)))
print(f"   순열과 $t$ 는 {C[2, 3]:.3f} 로 사실상 같은 검정이고, "
      f"부호검정만 {C[0, 2]:.3f} 로 떨어져 있다.")

print(f"\n② 가장 작은 p 를 고를 때의 실제 크기 (명목 {ALPHA})")
print(f"{'돌린 검정 수':>12}" + "".join(f"{nm:>22}" for nm in POP)
      + f"{'서로 독립이면':>14}")
SIZE = {nm: [] for nm in POP}
IND = []
for k in range(1, 5):
    IND.append(1 - (1 - ALPHA) ** k)
    row = ""
    for nm in POP:
        SIZE[nm].append(float(np.mean(P[nm][:, :k].min(axis=1) < ALPHA)))
        row += f"{SIZE[nm][-1]:>22.4f}"
    print(f"{k:>12}" + row + f"{IND[-1]:>14.4f}")

print(f"\n③ 올바른 문턱과 본페로니")
print(f"{'모집단':>22}{'올바른 문턱':>13}{'본페로니':>10}{'본페로니의 실제 크기':>20}")
QT, BSZ = {}, {}
for nm in POP:
    mn = P[nm].min(axis=1)
    QT[nm] = float(np.quantile(mn, ALPHA))
    BSZ[nm] = float(np.mean(mn < ALPHA / 4))
    print(f"{nm:>22}{QT[nm]:>13.4f}{ALPHA / 4:>10.4f}{BSZ[nm]:>20.4f}")
print(f"   본페로니는 명목의 절반에도 못 미치는 크기를 낸다 — "
      f"검정들이 서로 닮아서 보정이 과하다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))

im = ax1.imshow(C, cmap="viridis", vmin=0.4, vmax=1.0)
for i in range(4):
    for j in range(4):
        ax1.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                 fontsize=11, color="white" if C[i, j] < 0.85 else "black")
ax1.set_xticks(range(4)); ax1.set_xticklabels(SHORT, fontsize=10)
ax1.set_yticks(range(4)); ax1.set_yticklabels(SHORT, fontsize=10)
ax1.set_title("① 네 검정은 서로 닮았다", fontsize=12)
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

ks = np.arange(1, 5)
for nm, c, m in zip(POP, ("#2ca02c", "#1f77b4"), ("o", "s")):
    ax2.plot(ks, SIZE[nm], m + "-", color=c, lw=2.8, ms=8, label=nm)
ax2.plot(ks, IND, "^--", color="#d62728", lw=2.4, ms=8,
         label="서로 독립이라면")
ax2.axhline(ALPHA, color="black", lw=1.4, ls="--")
ax2.text(4.0, ALPHA * 0.72, f"약속한 ${ALPHA}$", fontsize=10.5, ha="right")
ax2.set_xticks(ks)
ax2.set_xticklabels([f"${v}$" for v in ks], fontsize=10.5)
ax2.set_xlabel("돌려 보고 가장 작은 $p$ 를 고른 검정의 수")
ax2.set_ylabel("귀무가설 아래의 실제 기각률")
ax2.set_ylim(0, 0.21)
ax2.set_title("② 네 개를 돌리면", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

xs = np.arange(len(POP))
w = 0.3
for j, (lab, vals, c) in enumerate([
        ("올바른 문턱으로 자르면", [ALPHA] * len(POP), "#2ca02c"),
        ("본페로니로 자르면", [BSZ[nm] for nm in POP], "#9467bd")]):
    ax3.bar(xs + (j - 0.5) * w, vals, w, color=c, label=lab)
    for x, v in zip(xs, vals):
        ax3.text(x + (j - 0.5) * w, v + 0.0015, f"${v:.4f}$", ha="center",
                 fontsize=10)
ax3.axhline(ALPHA, color="black", lw=1.4, ls="--")
for k, nm in enumerate(POP):
    ax3.text(k, 0.064, f"올바른 문턱 ${QT[nm]:.4f}$\n본페로니 "
                       f"${ALPHA / 4:.4f}$", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
ax3.set_xticks(xs)
ax3.set_xticklabels(list(POP), fontsize=10.5)
ax3.set_ylabel("귀무가설 아래의 실제 기각률")
ax3.set_ylim(0, 0.085)
ax3.set_title("③ 본페로니는 거꾸로 과하다", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch16/foundations/img/test_shopping.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
