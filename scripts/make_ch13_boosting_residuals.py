r"""「잔차에 무늬가 보이면 모형을 고쳐라」를 알고리즘으로 옮긴 것이 부스팅이다.

선형모형으로 시작해 잔차에 작은 나무를 하나씩 얹어 간다. 잔차의 무늬는 **이 장의
도구**로 잰다 — 잔차를 $x$ 로 $10$ 칸 나누어 일원분산분석 $F$ 를 돌리면, 무늬가 없을 때
$F \approx 1$ 이다.

① 훈련·교차검증·시험 오차가 라운드에 따라 갈라진다.
② 훈련 잔차의 $F$ 는 $1$ 에서 멈추지 않고 **$1$ 아래로** 계속 내려간다. $F < 1$ 은
   「무늬가 없다」가 아니라 「잔차가 지나치게 고르다」 — 외워 버렸다는 뜻이다.
③ 과적합한 자리에서 훈련 잔차 그림은 완벽한데 시험 잔차는 그대로 넓다.

만드는 파일: docs/ch13/diagnostics/img/boosting_residuals.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats                                    # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor     # noqa: E402
from sklearn.linear_model import LinearRegression          # noqa: E402
from sklearn.model_selection import KFold                  # noqa: E402

RNG = np.random.default_rng(20261021)

N, NTEST = 400, 20000
SIG = 0.45
ROUNDS = 1500
LR, DEPTH = 0.05, 3
NBIN = 10            # 잔차를 x 로 나눌 칸 수


def truth(x):
    return np.sin(2 * np.pi * x) + 0.8 * x


def make(n):
    x = RNG.uniform(0, 1, n)
    return x, truth(x) + RNG.normal(0, SIG, n)


def pattern_f(x, r, nbin=NBIN):
    """잔차를 x 로 나누어 일원분산분석 F 를 돌린다 — 무늬가 없으면 1 근처."""
    e = np.digitize(x, np.quantile(x, np.linspace(0, 1, nbin + 1)[1:-1]))
    g = [r[e == b] for b in range(nbin)]
    g = [v for v in g if len(v) > 1]
    return float(stats.f_oneway(*g).statistic)


xtr, ytr = make(N)
xte, yte = make(NTEST)
Xtr, Xte = xtr[:, None], xte[:, None]


def boost_stages(Xa, ya, Xb):
    """Xa·ya 로 선형 + 부스팅을 학습하고, Xb 에서 라운드마다의 예측을 돌려준다."""
    lin = LinearRegression().fit(Xa, ya)
    res = ya - lin.predict(Xa)
    gb = GradientBoostingRegressor(n_estimators=ROUNDS, learning_rate=LR,
                                   max_depth=DEPTH, random_state=0).fit(Xa, res)
    base = lin.predict(Xb)
    return np.array([base + s for s in gb.staged_predict(Xb)])


pred_tr = boost_stages(Xtr, ytr, Xtr)      # (ROUNDS, N)
pred_te = boost_stages(Xtr, ytr, Xte)      # (ROUNDS, NTEST)

# 교차검증 잔차 — 폴드마다 따로 학습해 남겨 둔 쪽을 예측한다
oof = np.zeros((ROUNDS, N))
for itr, iva in KFold(5, shuffle=True, random_state=0).split(Xtr):
    oof[:, iva] = boost_stages(Xtr[itr], ytr[itr], Xtr[iva])

mse_tr = ((pred_tr - ytr) ** 2).mean(axis=1)
mse_te = ((pred_te - yte) ** 2).mean(axis=1)
mse_oof = ((oof - ytr) ** 2).mean(axis=1)

f_tr = np.array([pattern_f(xtr, ytr - pred_tr[i]) for i in range(ROUNDS)])
f_oof = np.array([pattern_f(xtr, ytr - oof[i]) for i in range(ROUNDS)])

best = int(np.argmin(mse_te)) + 1
best_oof = int(np.argmin(mse_oof)) + 1
# 훈련 잔차의 F 가 1 을 가로지르는 곳, 그리고 1 아래로 깊이 내려간 곳
clean = int(np.argmax(f_tr < 1.0)) + 1
under = int(np.argmax(f_tr < 0.5)) + 1

print(f"자료 {N} 개, 잡음 표준편차 {SIG}, 나무 깊이 {DEPTH}, 학습률 {LR}")
print(f"{'라운드':>7}{'훈련 MSE':>11}{'교차검증':>11}{'시험 MSE':>11}"
      f"{'훈련 잔차 F':>13}{'교차검증 F':>12}")
for r in (1, 10, 50, clean, best, 300, 800, ROUNDS):
    i = r - 1
    print(f"{r:>7}{mse_tr[i]:>11.4f}{mse_oof[i]:>11.4f}{mse_te[i]:>11.4f}"
          f"{f_tr[i]:>13.2f}{f_oof[i]:>12.2f}")

print(f"\n훈련 잔차의 F 가 1 을 가로지르는 라운드   : {clean}")
print(f"훈련 잔차의 F 가 0.5 아래로 내려간 라운드 : {under}  "
      f"(그때 시험 MSE {mse_te[under - 1]:.4f})")
print(f"시험 MSE 가 가장 작은 라운드             : {best}  "
      f"(MSE {mse_te[best - 1]:.4f})")
print(f"교차검증 MSE 가 가장 작은 라운드          : {best_oof}  "
      f"(시험 MSE {mse_te[best_oof - 1]:.4f})")
print(f"잡음의 분산 {SIG ** 2:.4f} — 시험 MSE 의 바닥")
print(f"마지막 라운드의 훈련 MSE {mse_tr[-1]:.5f}, 시험 MSE {mse_te[-1]:.4f} "
      f"(최적보다 {100 * (mse_te[-1] / mse_te[best - 1] - 1):+.1f}%)")
print(f"마지막 라운드의 훈련 잔차 F {f_tr[-1]:.2f} — 「깨끗하다」고 말한다")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
R = np.arange(1, ROUNDS + 1)

ax1.loglog(R, mse_tr, color="#1f77b4", lw=2.4, label="훈련 MSE")
ax1.loglog(R, mse_oof, color="#2ca02c", lw=2.2, ls="--", label="교차검증 MSE")
ax1.loglog(R, mse_te, color="#d62728", lw=2.6, label="시험 MSE")
ax1.axhline(SIG ** 2, color="#999999", lw=1.3, ls=":")
ax1.text(1.4, SIG ** 2 * 1.14, f"잡음의 분산 ${SIG ** 2:.2f}$", fontsize=10,
         color="#555555")
ax1.axvline(best, color="#d62728", lw=1.4, ls="-.")
ax1.plot(best, mse_te[best - 1], "o", color="#d62728", ms=10)
ax1.text(best * 1.25, mse_te[best - 1] * 1.5, f"${best}$ 라운드", fontsize=10,
         color="#d62728")
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_xticklabels(["$1$", "$10$", "$100$", "$1{,}000$"], fontsize=9.5)
ax1.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
ax1.set_yticklabels(["$0.02$", "$0.05$", "$0.1$", "$0.2$", "$0.4$"])
ax1.minorticks_off()
ax1.set_xlabel("부스팅 라운드")
ax1.set_ylabel("평균제곱오차")
ax1.set_title("① 어디선가 갈라진다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25, which="major")

ax2.semilogx(R, f_tr, color="#1f77b4", lw=2.6, label="훈련 잔차의 $F$")
ax2.semilogx(R, f_oof, color="#2ca02c", lw=2.2, ls="--",
             label="교차검증 잔차의 $F$")
ax2.axhline(1.0, color="black", lw=1.3, ls=":")
ax2.text(1.4, 1.35, "무늬가 없으면 $F \\approx 1$", fontsize=10)
ax2.axvline(best, color="#d62728", lw=1.6, ls="-.")
ax2.annotate(f"시험 MSE 가 가장 작은\n${best}$ 라운드", xy=(best, 1.0),
             xytext=(3.0, 0.11), fontsize=10, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.axhspan(f_tr.min() * 0.8, 1.0, color="#d62728", alpha=0.10)
ax2.text(420, 4.5, "$F < 1$ — 무늬가 없는 것이 아니라\n잔차가 지나치게 고르다",
         ha="center", fontsize=10, color="#aa2222")
ax2.set_yscale("log")
ax2.set_xticks([1, 10, 100, 1000])
ax2.set_xticklabels(["$1$", "$10$", "$100$", "$1{,}000$"], fontsize=9.5)
ax2.set_yticks([0.05, 0.2, 1, 5, 20, 100])
ax2.set_yticklabels(["$0.05$", "$0.2$", "$1$", "$5$", "$20$", "$100$"])
ax2.minorticks_off()
ax2.set_xlabel("부스팅 라운드")
ax2.set_ylabel("잔차를 $x$ 로 나눈 일원분산분석 $F$")
ax2.set_title("② $1$ 을 지나도 멈추지 않는다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25, which="major")

sub = RNG.choice(NTEST, 1200, replace=False)
ax3.scatter(xte[sub], (yte - pred_te[-1])[sub], s=9, alpha=0.3,
            color="#d62728", edgecolors="none",
            label=f"시험 잔차 (RMS ${np.sqrt(mse_te[-1]):.2f}$)")
ax3.scatter(xtr, ytr - pred_tr[-1], s=16, alpha=0.85, color="#1f77b4",
            edgecolors="none",
            label=f"훈련 잔차 (RMS ${np.sqrt(mse_tr[-1]):.3f}$)")
ax3.axhline(0, color="black", lw=1.2, ls=":")
ax3.set_xlabel("$x$")
ax3.set_ylabel("잔차")
ax3.set_title(f"③ ${ROUNDS}$ 라운드에서 본 두 잔차", fontsize=12)
ax3.legend(loc="upper right", fontsize=10, markerscale=1.6)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/diagnostics/img/boosting_residuals.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
