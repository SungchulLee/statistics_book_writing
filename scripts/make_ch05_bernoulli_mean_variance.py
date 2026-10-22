r"""평균을 알면 분산을 아는 분포 — 그래서 분류에는 $\sigma^2$ 가 없다.

베르누이에서 $\operatorname{Var}(X) = p(1-p)$ 는 **평균의 함수**다. 그래서
$\bar X$ 를 알면 $S^2$ 를 따로 추정할 일이 없다. 정규에서는 정반대로 둘이
**독립**이다. 이 차이가 분류와 회귀의 생김새를 가른다.

① $\bar X$ 와 $S^2$ 를 찍으면 — 베르누이는 포물선 위, 정규는 구름.
② 정규에서는 그 매임이 없어 분산을 따로 맞혀야 한다.
③ 공짜로 따라오는 점검 — 예측확률의 합이 오류 수의 분산을 정한다.

만드는 파일: docs/ch05/applications/img/bernoulli_mean_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)

# === ① 평균과 분산의 관계 ============================================
N1, R1, P1 = 40, 1200, 0.35
xb_b = RNG.binomial(N1, P1, R1) / N1
s2_b = xb_b * (1 - xb_b) * N1 / (N1 - 1)
# 정규 쪽은 같은 평균·분산으로 눈금을 맞춰야 한 그림에 겹쳐 볼 수 있다
SD1 = np.sqrt(P1 * (1 - P1))
z = P1 + SD1 * RNG.standard_normal((R1, N1))
xb_n, s2_n = z.mean(1), z.var(1, ddof=1)
print(f"① n = {N1}, 반복 {R1:,}")
print(f"   베르누이 : S^2 = n/(n-1) * Xbar(1-Xbar) 가 **정확히** 성립한다 — "
      f"최대 어긋남 {np.abs(s2_b - xb_b * (1 - xb_b) * N1 / (N1 - 1)).max():.2e}")
print(f"   정규     : corr(Xbar, S^2) = {np.corrcoef(xb_n, s2_n)[0, 1]:+.4f} "
      f"(이론값 0 — 둘은 독립이다)")

# === ② 회귀는 둘이 들고 분류는 하나로 끝난다 ==========================
N2, R2 = 4000, 300
print("\n② 회귀 — 평균과 분산이 따로 노는 쪽")
XR = np.linspace(0, 1, N2)
SIG = 0.25 + 1.75 * XR                    # 오른쪽으로 갈수록 흩어진다
BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cov = np.zeros(len(BINS) - 1)
for _ in range(R2):
    y = np.zeros(N2) + SIG * RNG.standard_normal(N2)   # 참 평균은 0 으로 안다
    sg = y.std(ddof=1)                     # 하나뿐인 sigma 추정값
    inside = np.abs(y) <= 1.96 * sg
    for b in range(len(BINS) - 1):
        m = (XR >= BINS[b]) & (XR < BINS[b] + 0.2)
        cov[b] += inside[m].mean()
cov /= R2
print(f"{'x 구간':>10}{'참 sigma':>11}{'95% 구간의 실제 포함률':>24}")
for b in range(len(BINS) - 1):
    m = (XR >= BINS[b]) & (XR < BINS[b] + 0.2)
    print(f"{f'{BINS[b]:.1f}-{BINS[b]+0.2:.1f}':>10}{SIG[m].mean():>11.2f}"
          f"{cov[b]:>24.3f}")
print("   분산을 하나로 잡으면 한쪽은 넘치고 한쪽은 모자란다. "
      "베르누이에서는 이런 일이 **있을 수 없다** — 분산이 평균에 매여 있어서다.")

# === ③ 공짜로 따라오는 점검 ==========================================
NTEST, R3 = 2000, 4000
print(f"\n③ 시험자료 {NTEST:,} 개에서 '틀린 개수'의 분산 (반복 {R3:,})")
print(f"{'모형':>14}{'틀린 개수 평균':>16}{'관측 분산':>12}{'sum p(1-p)':>13}")
LBL, OBS_V, PRED_V, CNT = [], [], [], []
base = RNG.uniform(0.02, 0.98, NTEST)
for nm, temp in (("잘 맞춘 모형", 1.0), ("과신한 모형", 0.35),
                 ("과소신한 모형", 2.5)):
    logit = np.log(base / (1 - base)) / temp
    phat = 1 / (1 + np.exp(-logit))          # 모형이 **내놓는** 확률
    wrong = np.empty(R3)
    for i in range(R3):
        y = RNG.random(NTEST) < base         # 진짜는 늘 base 에서 나온다
        pred = phat > 0.5
        wrong[i] = np.sum(pred != y)
    pw = np.where(phat > 0.5, 1 - base, base)
    LBL.append(nm)
    OBS_V.append(float(wrong.var()))
    PRED_V.append(float(np.sum(phat * (1 - phat))))
    CNT.append(float(wrong.mean()))
    print(f"{nm:>14}{CNT[-1]:>16.1f}{OBS_V[-1]:>12.1f}{PRED_V[-1]:>13.1f}")
true_v = float(np.sum(base * (1 - base)))
print(f"   참 확률로 센 sum p(1-p) = {true_v:.1f}")
print("   모형이 내놓는 확률로 센 값이 관측 분산과 맞아야 한다. "
      "과신하면 작게, 과소신하면 크게 나온다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_B, C_N, C_G = "#d62728", "#1f77b4", "#2ca02c"

g = np.linspace(0, 1, 200)
ax1.plot(g, g * (1 - g) * N1 / (N1 - 1), color="#333333", lw=2.0,
         label="$\\frac{n}{n-1}\\bar X(1-\\bar X)$")
ax1.scatter(xb_b, s2_b, s=14, color=C_B, alpha=0.5, linewidths=0,
            label="베르누이 표본")
ax1.scatter(xb_n, s2_n, s=14, color=C_N, alpha=0.5, linewidths=0,
            label=f"정규 표본 (평균 {P1}, 같은 분산으로 맞춤)")
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 0.42)
ax1.set_xlabel("$\\bar X$")
ax1.set_ylabel("$S^2$")
ax1.set_title(f"① 한쪽은 선, 한쪽은 구름 ($n = {N1}$)", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25)

ctr = BINS[:-1] + 0.1
ax2.bar(ctr, cov, 0.17, color=C_N, label="실제 포함률")
ax2.axhline(0.95, color=C_B, lw=1.8, ls="--")
ax2.text(0.99, 0.935, "말한 $95\\%$", color=C_B, fontsize=10.5,
         ha="right", va="top")
for i, c in enumerate(cov):
    ax2.text(ctr[i], c + 0.006, f"${c:.3f}$", ha="center", fontsize=10)
ax2.set_xlabel("$x$ — 오른쪽으로 갈수록 참 $\\sigma$ 가 크다")
ax2.set_ylabel("$95\\%$ 라 말한 구간의 포함률")
ax2.set_ylim(0.78, 1.025)
ax2.set_title("② 회귀는 분산을 따로 맞혀야 한다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10.5)
ax2.grid(alpha=0.25, axis="y")

xs = np.arange(3)
w = 0.38
ax3.bar(xs - w / 2, OBS_V, w, color="#555555", label="관측한 분산")
ax3.bar(xs + w / 2, PRED_V, w, color=C_G,
        label="$\\sum_i \\hat p_i(1-\\hat p_i)$")
for i in range(3):
    ax3.text(xs[i] + w / 2, PRED_V[i] + 8, f"${PRED_V[i]:.0f}$", ha="center",
             fontsize=10)
    ax3.text(xs[i] - w / 2, OBS_V[i] + 8, f"${OBS_V[i]:.0f}$", ha="center",
             fontsize=10)
ax3.set_xticks(xs)
ax3.set_xticklabels(LBL, fontsize=10.5)
ax3.set_ylabel("틀린 개수의 분산")
ax3.set_ylim(0, max(max(OBS_V), max(PRED_V)) * 1.25)
ax3.set_title(f"③ 공짜로 따라오는 점검 ($n = {NTEST:,})".replace(",", "{,}")
              .replace("$n", "$n") + "$", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/bernoulli_mean_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
