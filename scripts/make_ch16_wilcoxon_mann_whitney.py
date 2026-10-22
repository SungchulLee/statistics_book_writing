r"""같은 검정을 두 사람이 따로, 그리고 전혀 다른 얼굴로 적었다.

윌콕슨$(1945)$ 은 **순위를 더했고**, 만과 휘트니$(1947)$ 는 **쌍을 세었다.**
두 수는 $U = W - n_1(n_1+1)/2$ 로 정확히 묶인다.

① 윌콕슨의 그림 — 합쳐서 매긴 순위.
② 만–휘트니의 그림 — $n_1 \times n_2$ 짝의 승패표.
③ 그런데 계산은 다르다 — 쌍 세기는 $O(n^2)$, 정렬은 $O(n\log n)$.

만드는 파일: docs/ch16/two_sample_nonparametric/img/wilcoxon_mann_whitney.png
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
NS = [500, 1000, 2000, 4000, 8000]
CHECK, CHECK_P = 5000, 3000

X = np.array([2.1, 3.4, 4.8, 5.2, 6.9, 8.1])
Y = np.array([1.3, 2.7, 3.9, 4.1, 5.8, 7.2, 9.4])
N1, N2 = len(X), len(Y)

rank_all = stats.rankdata(np.r_[X, Y])
W = float(rank_all[:N1].sum())
GRID = (X[:, None] > Y[None, :]).astype(float)
U = float(GRID.sum())
print(f"보기 자료 — X {N1} 개, Y {N2} 개")
print(f"   윌콕슨의 순위합 W = {W:g}")
print(f"   만–휘트니의 쌍 세기 U = {U:g}")
print(f"   W - n1(n1+1)/2 = {W - N1 * (N1 + 1) / 2:g}  -> 같다")

# === ② 언제나 같은가 =================================================
bad = 0
for _ in range(CHECK):
    n1, n2 = int(RNG.integers(3, 25)), int(RNG.integers(3, 25))
    a, b = RNG.normal(0, 1, n1), RNG.normal(0.3, 1, n2)
    if RNG.random() < 0.4:                       # 동점을 일부러 만든다
        a, b = np.round(a, 1), np.round(b, 1)
    r = stats.rankdata(np.r_[a, b])
    w = r[:n1].sum()
    u = np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])
    bad += abs(u - (w - n1 * (n1 + 1) / 2)) > 1e-9
print(f"\n② 무작위 자료 {CHECK:,} 벌에서 항등식이 깨진 경우 : {bad} 벌")

print(f"\n   p 값은 어떤가 ({CHECK_P:,} 벌)")
PV = {}
for lab, tie in (("동점 없음", False), ("동점 있음", True)):
    d_nc = d_df = 0
    mx = 0.0
    PV[lab] = []
    for _ in range(CHECK_P):
        n1, n2 = int(RNG.integers(5, 30)), int(RNG.integers(5, 30))
        a, b = RNG.normal(0, 1, n1), RNG.normal(0.3, 1, n2)
        if tie:
            a, b = np.round(a, 1), np.round(b, 1)
        p_rs = stats.ranksums(a, b).pvalue
        p_nc = stats.mannwhitneyu(a, b, alternative="two-sided",
                                  method="asymptotic",
                                  use_continuity=False).pvalue
        p_df = stats.mannwhitneyu(a, b, alternative="two-sided",
                                  method="asymptotic").pvalue
        d_nc += abs(p_rs - p_nc) > 1e-9
        d_df += abs(p_rs - p_df) > 1e-9
        mx = max(mx, abs(p_rs - p_nc))
        PV[lab].append((p_rs, p_df))
    v = np.array(PV[lab])
    print(f"      {lab} : ranksums 대 mannwhitneyu(연속성보정 끔) "
          f"{d_nc}/{CHECK_P} 다름 (최대차 {mx:.1e}),  "
          f"기본값과는 {d_df}/{CHECK_P} 다름 "
          f"(최대차 {np.abs(v[:, 1] - v[:, 0]).max():.4f})")

# === ③ 계산 시간 =====================================================
def best(f, rep=5):
    f()
    out = []
    for _ in range(rep):
        s = time.perf_counter()
        f()
        out.append(time.perf_counter() - s)
    return min(out)


print(f"\n③ 계산 시간 ($n_1 = n_2 = n$)")
print(f"{'n':>7}{'쌍 세기 ms':>13}{'정렬+순위합 ms':>17}{'배수':>8}")
TP, TR = [], []
for n in NS:
    a, b = RNG.normal(0, 1, n), RNG.normal(0, 1, n)
    TP.append(best(lambda: int(np.sum(a[:, None] > b[None, :]))) * 1e3)
    TR.append(best(lambda: stats.rankdata(np.r_[a, b])[:n].sum()) * 1e3)
    print(f"{n:>7}{TP[-1]:>13.2f}{TR[-1]:>17.2f}{TP[-1] / TR[-1]:>8.1f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C_X, C_Y = "#d62728", "#1f77b4"

order = np.argsort(np.r_[X, Y])
vals = np.r_[X, Y][order]
is_x = (np.arange(N1 + N2) < N1)[order]
for k, (v, ix) in enumerate(zip(vals, is_x)):
    c = C_X if ix else C_Y
    ax1.plot([k + 1], [v], "o", color=c, ms=11)
    ax1.text(k + 1, v + 0.45, f"${k + 1}$", ha="center", fontsize=10,
             color=c, fontweight="bold")
ax1.plot([], [], "o", color=C_X, ms=9, label=f"$X$ ({N1} 개)")
ax1.plot([], [], "o", color=C_Y, ms=9, label=f"$Y$ ({N2} 개)")
ax1.set_xticks(range(1, N1 + N2 + 1))
ax1.set_xlabel("합쳐서 매긴 순위")
ax1.set_ylabel("값")
ax1.set_ylim(0, 11)
ax1.set_title(f"① 윌콕슨 — 순위를 더한다  $W = {W:g}$", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

# 쪽의 2 절에 이미 승패표 그림이 있으므로, 여기서는 구현 차이를 보인다.
# 두 p 값이 대각선에 붙어 보이지 않으므로 차이 자체를 그린다.
for lab, c, m in (("동점 없음", "#2ca02c", "o"), ("동점 있음", "#d62728", "x")):
    v = np.array(PV[lab])
    ax2.plot(v[:, 0], v[:, 1] - v[:, 0], m, color=c, ms=3.5, alpha=0.45,
             label=lab)
ax2.axhline(0, color="#555555", lw=1.4)
ax2.set_xlabel("`ranksums` 의 $p$ 값")
ax2.set_ylabel("`mannwhitneyu` 기본값 $-$ `ranksums`")
ax2.set_xlim(0, 1)
ax2.set_title("② 같은 검정, 다른 기본값", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5, markerscale=2.5)
ax2.grid(alpha=0.25)

ax3.loglog(NS, TP, "o-", color=C_X, lw=2.8, ms=7, label="쌍 세기 $O(n^2)$")
ax3.loglog(NS, TR, "s-", color=C_Y, lw=2.8, ms=7,
           label="정렬 + 순위합 $O(n\\log n)$")
ax3.annotate(f"$n = {NS[-1]}$ 에서\n${TP[-1] / TR[-1]:.0f}$ 배 차이",
             xy=(NS[-1] * 0.98, np.sqrt(TP[-1] * TR[-1])),
             xytext=(900, 12.0), textcoords="data", fontsize=10.5,
             ha="center", arrowprops=dict(arrowstyle="->", lw=1.1))
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v}$" for v in NS], fontsize=9.5)
ax3.set_yticks([0.03, 0.1, 1, 10, 100])
ax3.set_yticklabels(["$0.03$", "$0.1$", "$1$", "$10$", "$100$"])
ax3.minorticks_off()
ax3.set_xlabel("$n_1 = n_2 = n$")
ax3.set_ylabel("계산 시간 (ms)")
ax3.set_title("③ 같은 수인데 셈이 다르다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch16/two_sample_nonparametric/img/wilcoxon_mann_whitney.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
