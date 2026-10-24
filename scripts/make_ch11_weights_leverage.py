r"""11.3절 「그림으로 보는 통계학」 — 열아홉 살 한 사람이 전국 평균을 움직였다.

가중치는 **지렛대**다. 한 관측의 가중치를 키우면 가중평균도 가중최소제곱의 적합선도
그 관측 쪽으로 끌려가고, 유효표본크기는 줄어든다. $2016$ 년 미국 대선 추적조사에서
실제로 그 일이 있었다.

① 한 사람이 쥔 몫 — 가중치가 평균의 $k$ 배이면 전체의 몇 %인가.
② 회귀에서는 지렛대가 된다 — 가중치를 키울수록 $h_{ii}$ 가 $1$ 로 간다.
③ 유효표본크기 — 키시의 $n_\text{eff} = (\sum w)^2 / \sum w^2$.

만드는 파일: docs/ch11/assumptions/img/weights_leverage.png
실행:  python3 scripts/make_ch13_weights_leverage.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/assumptions/img/weights_leverage.png"
RNG = np.random.default_rng(20261022)
NPANEL = 3_000          # 조사 패널의 크기
KS = np.logspace(0, np.log10(300), 200)

# --- ① 한 사람이 쥔 몫 --------------------------------------------------
share = KS / (NPANEL - 1 + KS)          # 그 한 사람의 가중치 ÷ 전체 가중치
print("① 패널 3,000 명에서 한 사람의 가중치가 평균의 k 배일 때")
for k in (1, 10, 30, 100, 300):
    s = k / (NPANEL - 1 + k)
    print(f"   k = {k:>3}:  전체 가중치의 {100*s:5.2f}%  "
          f"→ 그가 답을 바꾸면 가중평균이 {100*s:5.2f} 백분율점 움직인다")

# --- ② 회귀의 지렛대 ----------------------------------------------------
N2 = 40
x = np.sort(RNG.uniform(0, 10, N2))
y = 1.0 + 0.5 * x + 0.6 * RNG.standard_normal(N2)
x[-1], y[-1] = 9.2, 1.2                  # 선에서 멀리 떨어진 한 점
X = np.column_stack([np.ones(N2), x])


def wls(w):
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    H = X @ np.linalg.solve(XtWX, X.T * w)     # 가중 영향행렬
    return beta, np.diag(H)


WS = (1, 5, 30, 100)
fits = {}
print("\n② n = 40, 마지막 한 점의 가중치만 키운다")
for wk in WS:
    w = np.ones(N2)
    w[-1] = wk
    b, h = wls(w)
    fits[wk] = (b, h[-1])
    print(f"   w = {wk:>3}:  기울기 {b[1]:.4f}   그 점의 지렛대 h = {h[-1]:.4f}")

wgrid = np.logspace(0, 3, 120)
hgrid = np.array([wls(np.r_[np.ones(N2 - 1), wk])[1][-1] for wk in wgrid])

# --- ③ 유효표본크기 ----------------------------------------------------
neff = (NPANEL - 1 + KS) ** 2 / (NPANEL - 1 + KS ** 2)
print("\n③ 키시의 유효표본크기 (나머지 2,999 명은 가중치 1)")
for k in (1, 10, 30, 100, 300):
    ne = (NPANEL - 1 + k) ** 2 / (NPANEL - 1 + k ** 2)
    print(f"   k = {k:>3}:  n_eff = {ne:7.0f}  (잃은 몫 {100*(1 - ne/NPANEL):5.1f}%)")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
ax.plot(KS, 100 * share, color=C1, lw=2.2)
for k, col in ((30, C2),):
    s = 100 * k / (NPANEL - 1 + k)
    ax.plot([k], [s], "o", color=col, ms=8)
    ax.annotate(f"k = {k} 이면 {s:.2f}%\n— 그 한 사람이 가중평균을\n그만큼 움직인다",
                xy=(k, s), xytext=(1.6, 6.0), color=col, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=col))
ax.set_xscale("log")
ax.set_xlabel("그 한 사람의 가중치 ÷ 평균 가중치  (k)")
ax.set_ylabel("전체 가중치에서 차지하는 몫 (%)")
ax.set_title(f"① 패널 {NPANEL:,} 명 가운데 한 사람", fontsize=11)

ax = axes[1]
ax.scatter(x[:-1], y[:-1], s=28, color="0.55", zorder=3)
ax.scatter([x[-1]], [y[-1]], s=90, color=C2, zorder=4)
xs = np.linspace(0, 10, 50)
for wk, sty in zip(WS, ("-", "--", "-.", ":")):
    b = fits[wk][0]
    ax.plot(xs, b[0] + b[1] * xs, sty, color=C1, lw=1.8,
            label=f"w = {wk:>3}  기울기 {b[1]:.3f}  h = {fits[wk][1]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("② 가중치를 키우면 선이 끌려온다", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[2]
ax.plot(KS, neff, color=C1, lw=2.2)
ax.axhline(NPANEL, color="0.45", ls=":", lw=1.2)
ax.text(1.1, NPANEL * 0.96, f"가중치가 고르면 {NPANEL:,}", color="0.35", fontsize=9)
for k, col in ((30, C2), (300, C2)):
    ne = (NPANEL - 1 + k) ** 2 / (NPANEL - 1 + k ** 2)
    ax.plot([k], [ne], "o", color=col, ms=8)
    ax.annotate(f"k = {k} → {ne:,.0f}", xy=(k, ne),
                xytext=(k * 0.33, ne + 320), color=col, fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("그 한 사람의 가중치 ÷ 평균 가중치  (k)")
ax.set_ylabel("유효표본크기  $n_\\mathrm{eff}$")
ax.set_title("③ 한 사람이 깎아 내는 표본", fontsize=11)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
