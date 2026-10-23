r"""반복 교차검증은 답을 좋게 만들지 않는다. 답을 같게 만든다.

이 쪽 절 5 는 다섯 겹이 저마다 고르는 lambda 가 5.6 배까지 벌어진다는 것을 보였고,
절 6 은 「반복 교차검증을 쓰면 특정 분할에 대한 민감도가 줄어든다」고 적는다.
얼마나 줄어드는지, 그리고 줄어든 것이 무엇인지는 적혀 있지 않다.

자료를 하나로 고정하면 남는 우연은 **겹을 어떻게 나누느냐**뿐이다. 그 우연만
바꿔 가며 R 번 반복한 교차검증이 고르는 lambda 의 분포를 본다.

① 반복 횟수에 따른 고른 lambda 의 분포.
② 흩어짐이 1/sqrt(R) 로 주는가.
③ 그래서 시험오차가 좋아지는가.

만드는 파일: docs/ch18/tuning/img/repeated_cv.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.linear_model import lasso_path             # noqa: E402
from sklearn.model_selection import KFold               # noqa: E402

RNG = np.random.default_rng(0)
N, NTE, P, K_TRUE = 200, 5000, 30, 5
POOL, NDRAW = 400, 2000          # 분할 묶음 크기, 재표집 횟수


def make(rng):
    X = rng.normal(size=(N + NTE, P))
    X -= X.mean(0)
    X /= X.std(0)
    b = np.zeros(P)
    b[:K_TRUE] = np.array([2.0, -1.5, 1.0, 0.8, -0.6])
    y = X @ b + rng.normal(size=N + NTE)
    return X[:N], y[:N] - y[:N].mean(), X[N:], y[N:] - y[:N].mean()


X, y, XTE, YTE = make(RNG)
LAMS = np.logspace(np.log10(0.6), np.log10(0.004), 120)      # 내림차순

# === 분할 묶음 — 한 번의 5 겹 교차검증 곡선을 POOL 개 모은다 =========
CURVES = np.empty((POOL, len(LAMS)))
for b in range(POOL):
    kf = KFold(5, shuffle=True, random_state=b)
    err = np.zeros(len(LAMS))
    for i_tr, i_va in kf.split(X):
        _, co, _ = lasso_path(X[i_tr], y[i_tr], alphas=LAMS)
        pred = X[i_va] @ co
        err += ((y[i_va][:, None] - pred) ** 2).mean(0)
    CURVES[b] = err / 5

# 참 시험오차 — 전체 훈련자료로 적합해 따로 둔 시험집합에서 잰다
_, CO_FULL, _ = lasso_path(X, y, alphas=LAMS)
TEST = ((YTE[:, None] - XTE @ CO_FULL) ** 2).mean(0)
i_best = int(np.argmin(TEST))

RS = [1, 2, 5, 10, 20, 50]
SEL = {}
rng2 = np.random.default_rng(7)
for R in RS:
    lam_sel = np.empty(NDRAW)
    for d in range(NDRAW):
        idx = rng2.choice(POOL, size=R, replace=False)
        lam_sel[d] = LAMS[int(np.argmin(CURVES[idx].mean(0)))]
    SEL[R] = lam_sel

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"자료 하나 고정 (n = {N}, p = {P}, 참 변수 {K_TRUE} 개).")
print(f"남는 우연은 겹 나누기뿐이다. 분할 {POOL} 개를 모아 두고 "
      f"R 개씩 뽑아 {NDRAW:,} 번 되풀이했다.")
print(f"따로 둔 시험집합 {NTE:,} 개로 잰 참 최적 lambda = {LAMS[i_best]:.4f}\n")

print("① 반복 횟수 R 에 따른 「고른 lambda」")
print("     R   중앙값   하위 5 %   상위 95 %   비(상위/하위)   log lambda 의 sd")
for R in RS:
    s = SEL[R]
    lo, hi = np.percentile(s, [5, 95])
    print(f"   {R:>3}   {np.median(s):>6.4f}   {lo:>8.4f}   {hi:>9.4f}   "
          f"{hi/lo:>13.2f}   {np.std(np.log(s)):>16.3f}")
r1 = np.percentile(SEL[1], 95) / np.percentile(SEL[1], 5)
r50 = np.percentile(SEL[50], 95) / np.percentile(SEL[50], 5)
print(f"   R = 1 에서 두 분석자가 고르는 lambda 가 {r1:.1f} 배까지 벌어지고, "
      f"R = 50 에서 {r50:.1f} 배다.")

sd1, sd50 = np.std(np.log(SEL[1])), np.std(np.log(SEL[50]))
print(f"\n② 흩어짐이 1/sqrt(R) 를 따르는가")
print(f"   sd(log lambda) 가 {sd1:.3f} 에서 {sd50:.3f} 로 "
      f"{sd1/sd50:.2f} 배 줄었다. 1/sqrt(50) 의 역수는 {np.sqrt(50):.2f} 다.")

print("\n③ 그래서 시험오차가 좋아지는가")
print("     R   고른 lambda 에서의 시험 MSE (중앙값)   참 최적 대비")
for R in RS:
    mse = np.array([TEST[int(np.argmin(np.abs(LAMS - l)))] for l in SEL[R]])
    print(f"   {R:>3}   {np.median(mse):>33.4f}   {np.median(mse)/TEST[i_best]:>11.4f}")
m1 = np.median([TEST[int(np.argmin(np.abs(LAMS - l)))] for l in SEL[1]])
m50 = np.median([TEST[int(np.argmin(np.abs(LAMS - l)))] for l in SEL[50]])
print(f"   R 을 {RS[0]} 에서 {RS[-1]} 로 쉰 배 늘려 얻은 것은 "
      f"시험 MSE {m1:.4f} -> {m50:.4f}, **{(1-m50/m1)*100:.2f} %** 다.")
lam_conv = np.median(SEL[RS[-1]])
print(f"\n④ 어디로 모이는가")
print(f"   R = {RS[-1]} 에서 고르는 lambda 는 {lam_conv:.4f} 에 모인다.")
print(f"   참 최적은 {LAMS[i_best]:.4f} 다 — {lam_conv/LAMS[i_best]:.2f} 배 크다.")
print(f"   반복을 늘려도 그 자리는 움직이지 않는다 "
      f"(R = 1 의 중앙값 {np.median(SEL[1]):.4f}).")
print("   **반복은 분산을 줄일 뿐 치우침을 건드리지 않는다. "
      "같은 답을 더 확신하게 만든다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B = "#1f77b4", "#d62728"

SHOW = [1, 5, 20]
parts = ax1.violinplot([np.log10(SEL[R]) for R in SHOW], showextrema=False,
                       widths=0.8)
for pc in parts["bodies"]:
    pc.set_facecolor(C_A)
    pc.set_alpha(0.55)
for i, R in enumerate(SHOW, start=1):
    ax1.plot([i], [np.median(np.log10(SEL[R]))], "o", color="#222222", ms=9)
ax1.axhline(np.log10(LAMS[i_best]), color=C_B, lw=2.4, ls="--",
            label=f"참 최적 $\\lambda = {LAMS[i_best]:.3f}$")
ax1.set_xticks(range(1, len(SHOW) + 1))
ax1.set_xticklabels([f"$R = {R}$" for R in SHOW])
ax1.set_ylabel("$\\log_{10}$ (고른 $\\lambda$)")
ax1.set_title("① 반복을 늘리면 좁아진다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25, axis="y")

sds = np.array([np.std(np.log(SEL[R])) for R in RS])
ax2.loglog(RS, sds, "o-", color=C_A, lw=2.8, ms=10, label="잰 값")
ax2.loglog(RS, sds[0] / np.sqrt(np.array(RS)), "--", color="#333333", lw=2.2,
           label="$1/\\sqrt{R}$")
ax2.set_xlabel("반복 횟수 $R$")
ax2.set_ylabel("$\\mathrm{sd}(\\log \\lambda)$")
# 로그 축의 음수 지수는 mathtext 를 거치므로 눈금을 십진수로 직접 적는다
_yt = [0.02, 0.05, 0.1]
ax2.set_yticks(_yt)
ax2.set_yticklabels([f"${v:g}$" for v in _yt])
ax2.set_xticks(RS)
ax2.set_xticklabels([str(r) for r in RS])
ax2.minorticks_off()
ax2.set_title("② 흩어짐은 $1/\\sqrt{R}$ 로 준다", fontsize=12)
ax2.legend(loc="upper right", fontsize=11)
ax2.grid(alpha=0.25)

med = np.array([np.median([TEST[int(np.argmin(np.abs(LAMS - l)))]
                           for l in SEL[R]]) for R in RS])
p90 = np.array([np.percentile([TEST[int(np.argmin(np.abs(LAMS - l)))]
                               for l in SEL[R]], 90) for R in RS])
ax3.semilogx(RS, med, "o-", color=C_A, lw=2.8, ms=10, label="중앙값")
ax3.semilogx(RS, p90, "s:", color=C_A, lw=2.2, ms=8, alpha=0.7,
             label="상위 $10\\%$")
ax3.axhline(TEST[i_best], color=C_B, lw=2.4, ls="--",
            label=f"참 최적에서 {TEST[i_best]:.3f}")
ax3.set_xlabel("반복 횟수 $R$")
ax3.set_ylabel("고른 $\\lambda$ 에서의 시험 MSE")
ax3.set_xticks(RS)
ax3.set_xticklabels([str(r) for r in RS])
ax3.minorticks_off()
ax3.set_title("③ 그런데 오차는 거의 그대로다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/tuning/img/repeated_cv.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
