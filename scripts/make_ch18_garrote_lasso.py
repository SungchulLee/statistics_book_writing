r"""라쏘가 개럿에서 한 단계를 덜어 낸 자리.

브라이먼의 **비음 개럿**은 두 단계다. 먼저 최소제곱 $\hat\beta_j$ 를 구하고,
그다음 그것에 곱할 비음 계수 $c_j$ 를 $\sum c_j \le t$ 아래에서 고른다.
티브시라니는 $\hat\beta$ 가 $p > N$ 에서 존재하지 않는다는 점을 들어
두 단계를 하나로 합쳤다. 그 한 걸음이 무엇을 바꾸는지 잰다.

① $p/n = 0.4$ 에서 두 해의 계수 — 거의 같은 일을 한다.
② $p/n$ 을 키우며 시험 RMSE — 개럿은 $p/n \to 1$ 에서 무너지고
   $p \ge n$ 에서는 아예 정의되지 않는다.
③ 변수 선택 — 개럿이 라쏘보다 훨씬 깨끗한 집합을 고른다.

만드는 파일: docs/ch18/lasso/img/garrote_lasso.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso, LassoCV   # noqa: E402
from sklearn.model_selection import KFold         # noqa: E402

RNG = np.random.default_rng(17)
N, S, SIG, NTE = 100, 5, 1.0, 4000
PS = [10, 20, 40, 60, 75, 85, 92, 96, 99, 110, 130, 160]
REPS = 10
ALPHAS = np.logspace(-3.0, 1.2, 40)


def make(p):
    b = np.zeros(p)
    b[:S] = RNG.choice([-1.0, 1.0], S) * RNG.uniform(1.5, 2.5, S)
    X = RNG.normal(0, 1, (N, p))
    y = X @ b + RNG.normal(0, SIG, N)
    Xt = RNG.normal(0, 1, (NTE, p))
    yt = Xt @ b + RNG.normal(0, SIG, NTE)
    return X, y, Xt, yt, b


def garrote_fit(X, y, alpha):
    """비음 개럿 — 최소제곱을 먼저 구하고 그것에 곱할 c 를 고른다."""
    bh = np.linalg.lstsq(X, y, rcond=None)[0]
    c = Lasso(alpha=alpha, positive=True, fit_intercept=False,
              max_iter=50000).fit(X * bh, y).coef_
    return c * bh, bh


def garrote_cv(X, y, alphas=ALPHAS, k=5):
    """alpha 를 5겹 교차검증으로 고른다. 접마다 최소제곱을 다시 구한다."""
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    sse = np.zeros(len(alphas))
    for tr, va in kf.split(X):
        for i, a in enumerate(alphas):
            b, _ = garrote_fit(X[tr], y[tr], a)
            sse[i] += float(np.sum((X[va] @ b - y[va]) ** 2))
    return garrote_fit(X, y, alphas[int(sse.argmin())])


def f1(b_hat, b):
    sel, tru = b_hat != 0, b != 0
    tp = int((sel & tru).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (int(sel.sum()) + int(tru.sum()))


rmse = lambda Xt, b, yt: float(np.sqrt(np.mean((Xt @ b - yt) ** 2)))  # noqa

print(f"n = {N}, 참 계수 {S} 개, 잡음 sd {SIG}, 되풀이 {REPS} 회")
print(f"{'p':>5}{'p/n':>6}{'개럿':>9}{'라쏘':>9}{'영모형':>9}"
      f"{'|b_OLS|':>10}{'개럿 F1':>9}{'라쏘 F1':>9}")
res = {k: [] for k in ["g", "l", "0", "nb", "gf", "lf"]}
coef_demo = None
for p in PS:
    acc = {k: [] for k in res}
    for r in range(REPS):
        X, y, Xt, yt, b = make(p)
        la = LassoCV(cv=5, n_alphas=60, max_iter=50000,
                     random_state=0).fit(X, y)
        acc["l"].append(rmse(Xt, la.coef_, yt))
        acc["lf"].append(f1(la.coef_, b))
        acc["0"].append(float(np.sqrt(np.mean(yt ** 2))))
        if p < N:
            bg, bh = garrote_cv(X, y)
            acc["g"].append(rmse(Xt, bg, yt))
            acc["gf"].append(f1(bg, b))
            acc["nb"].append(float(np.linalg.norm(bh)))
            if p == 40 and r == 0:
                coef_demo = (b.copy(), bg.copy(), la.coef_.copy())
    for k in res:
        res[k].append(float(np.mean(acc[k])) if acc[k] else np.nan)
    i = len(res["g"]) - 1
    if p < N:
        print(f"{p:>5}{p / N:>6.2f}{res['g'][i]:>9.3f}{res['l'][i]:>9.3f}"
              f"{res['0'][i]:>9.3f}{res['nb'][i]:>10.2f}"
              f"{res['gf'][i]:>9.3f}{res['lf'][i]:>9.3f}")
    else:
        print(f"{p:>5}{p / N:>6.2f}{'정의안됨':>9}{res['l'][i]:>9.3f}"
              f"{res['0'][i]:>9.3f}{'—':>10}{'—':>9}{res['lf'][i]:>9.3f}")

g = np.array(res["g"])
ok = ~np.isnan(g)
i_lo = int(np.nanargmin(g))
i_hi = int(np.nanargmax(g))
print(f"\n개럿 — p/n = {PS[i_lo] / N:.2f} 에서 {g[i_lo]:.3f} 로 가장 좋고, "
      f"p/n = {PS[i_hi] / N:.2f} 에서 {g[i_hi]:.3f} 까지 간다 "
      f"(영모형 {res['0'][i_hi]:.3f})")
print(f"라쏘 — p/n 이 {PS[0] / N:.2f} 에서 {PS[-1] / N:.2f} 로 "
      f"{PS[-1] / PS[0]:.0f} 배 커지는 동안 "
      f"{min(res['l']):.3f}–{max(res['l']):.3f} 에 머문다")
print(f"최소제곱 노름 — p/n = {PS[0] / N:.2f} 에서 {res['nb'][0]:.2f}, "
      f"p/n = {PS[i_hi] / N:.2f} 에서 {res['nb'][i_hi]:.2f} "
      f"({res['nb'][i_hi] / res['nb'][0]:.1f} 배)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_G, C_L, C_T = "#ff7f0e", "#1f77b4", "#2ca02c"
pn = np.array(PS) / N

b_true, b_g, b_l = coef_demo
idx = np.arange(len(b_true))
m = (b_true != 0) | (b_g != 0) | (b_l != 0)
xs = np.arange(int(m.sum()))
w = 0.27
ax1.bar(xs - w, b_true[m], w, color=C_T, label="참 계수")
ax1.bar(xs, b_g[m], w, color=C_G, label="비음 개럿")
ax1.bar(xs + w, b_l[m], w, color=C_L, label="라쏘")
ax1.axhline(0, color="black", lw=0.9)
ax1.set_xticks(xs)
ax1.set_xticklabels([f"${v + 1}$" for v in idx[m]], fontsize=9)
ax1.set_xlabel(f"$0$ 이 아닌 계수를 받은 변수 ($p = 40$, $p/n = 0.4$)")
ax1.set_ylabel("계수")
ax1.set_title("① 여유가 있을 때는 거의 같은 일을 한다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(pn[ok], g[ok], "o-", color=C_G, lw=2.6, ms=7, label="비음 개럿")
ax2.plot(pn, res["l"], "s-", color=C_L, lw=2.6, ms=7, label="라쏘")
ax2.axvline(1.0, color="#d62728", lw=1.6, ls="--")
ax2.axvspan(1.0, pn[-1] * 1.05, color="#d62728", alpha=0.09)
# mathtext 에 \ge 가 없다 — 유니코드로 적는다
ax2.text(1.04, 1.21, "p ≥ n 에서\n개럿은 아예\n정의되지 않는다",
         fontsize=10.5, color="#d62728", va="top")
ax2.text(0.03, 0.985, f"아무것도 안 하면 ${np.mean(res['0']):.2f}$",
         fontsize=10, color="#888888")
ax2.set_xlabel("$p/n$")
ax2.set_ylabel("시험 RMSE")
ax2.set_ylim(0.97, 1.27)
ax2.set_xlim(0, pn[-1] * 1.05)
ax2.set_title("② 예측으로는 거의 비긴다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.88))
ax2.grid(alpha=0.25)

ax3.plot(pn[ok], np.array(res["gf"])[ok], "o-", color=C_G, lw=2.6, ms=7,
         label="비음 개럿")
ax3.plot(pn, res["lf"], "s-", color=C_L, lw=2.6, ms=7, label="라쏘")
ax3.axvline(1.0, color="#d62728", lw=1.6, ls="--")
ax3.axvspan(1.0, pn[-1] * 1.05, color="#d62728", alpha=0.09)
ax3.annotate(f"${res['gf'][2]:.2f}$ 대 ${res['lf'][2]:.2f}$",
             xy=(pn[2], res["gf"][2]), xytext=(0.52, 0.93),
             textcoords="data", fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax3.set_xlabel("$p/n$")
ax3.set_ylabel("고른 변수집합의 $F_1$")
ax3.set_ylim(0.25, 1.02)
ax3.set_xlim(0, pn[-1] * 1.05)
ax3.set_title("③ 고르는 일은 개럿이 더 잘했다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/lasso/img/garrote_lasso.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
