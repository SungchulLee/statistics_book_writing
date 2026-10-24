r"""건강한 사용자 편향 — 교란 하나가 방향을 뒤집는 데 얼마나 센 힘이 필요한가.

관측되지 않는 **건강 의식** $U \sim N(0,1)$ 이 둘을 함께 움직인다.

$$\Pr(T=1 \mid U) = \sigma(\alpha U), \qquad
  \operatorname{logit}\Pr(Y=1) = \beta_0 + \beta_T T - \gamma U$$

참 인과효과를 $\exp(\beta_T) = 1.29$ (해롭다) 로 두었다.

① $\alpha = \gamma = 1$ 이면 조잡 상대위험이 $0.56$ 으로 나온다 — 방향이 뒤집힌다.
② 대리변수로 보정해도 그 질 $\varrho$ 가 $1$ 이 아니면 끝까지 돌아오지 않는다.
③ 배정만 무작위로 바꾸면 보정을 하나도 하지 않아도 참값이 나온다.

만드는 파일: docs/ch10/confounding/img/healthy_user.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
import statsmodels.api as sm  # noqa: E402

RNG = np.random.default_rng(20261022)

N = 400000
RR_TRUE = 1.29                 # 참 인과 상대위험 (해롭다)
BT = np.log(RR_TRUE)
ALPHA, GAMMA, B0 = 1.0, 1.0, -5.0
RHOS = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
RR_OBS_HIST = 0.56             # 스탬퍼·콜디츠(1991)의 종합 상대위험


def generate(randomized=False, n=N):
    """U 가 복용과 발병을 함께 움직인다. randomized 면 복용을 동전으로 정한다."""
    u = RNG.standard_normal(n)
    p_t = 0.5 * np.ones(n) if randomized else 1 / (1 + np.exp(-ALPHA * u))
    t = (RNG.random(n) < p_t).astype(float)
    lo = B0 + BT * t - GAMMA * u
    y = (RNG.random(n) < 1 / (1 + np.exp(-lo))).astype(float)
    return u, t, y


def logit_rr(y, X):
    return float(np.exp(sm.Logit(y, sm.add_constant(X)).fit(disp=0).params[1]))


u, t, y = generate()
rr_crude = logit_rr(y, t)
rr_adj_u = logit_rr(y, np.c_[t, u])
print(f"참 인과 상대위험 {RR_TRUE}, 발생률 {y.mean():.4f}, 표본 {N:,}")
print(f"  보정 없음          : {rr_crude:.3f}")
print(f"  U 를 알고 보정     : {rr_adj_u:.3f}")
print(f"  (역사 속 관찰값 {RR_OBS_HIST})")

# === ② 대리변수의 질 =================================================
print(f"\n② 대리변수 X = ϱU + √(1-ϱ²)ε 로 보정했을 때")
print(f"{'ϱ':>6}{'보정 후 RR':>13}{'남은 편향(로그)':>17}")
rr_rho = []
for r in RHOS:
    x = r * u + np.sqrt(max(1 - r ** 2, 0.0)) * RNG.standard_normal(N)
    rr_rho.append(logit_rr(y, np.c_[t, x]))
    print(f"{r:>6.2f}{rr_rho[-1]:>13.3f}"
          f"{np.log(rr_rho[-1]) - BT:>17.3f}")
rr_rho = np.array(rr_rho)

# 대리변수를 여러 개 쓰면 나아지는가
print(f"\n   ϱ = 0.7 짜리 대리변수를 여러 개 쓰면")
for m in (1, 2, 4, 8):
    r = 0.7
    X = np.column_stack([r * u + np.sqrt(1 - r ** 2) * RNG.standard_normal(N)
                         for _ in range(m)])
    print(f"     {m} 개 : {logit_rr(y, np.c_[t, X]):.3f}")

# === ③ 무작위배정 ====================================================
ur, tr, yr = generate(randomized=True)
rr_rand = logit_rr(yr, tr)
rr_rand_adj = logit_rr(yr, np.c_[tr, ur])
print(f"\n③ 배정만 무작위로 바꾸면")
print(f"   보정 없음      : {rr_rand:.3f}")
print(f"   U 까지 보정    : {rr_rand_adj:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

names = ["관찰연구\n(보정 없음)", "역사 속\n관찰 종합값", "참 인과효과\n($U$ 를 알 때)"]
vals = [rr_crude, RR_OBS_HIST, rr_adj_u]
cols = ["#d62728", "#ff7f0e", "#1f77b4"]
ax1.barh(np.arange(3)[::-1], vals, 0.6, color=cols)
for i, v in enumerate(vals):
    ax1.text(v + 0.03, 2 - i, f"${v:.3f}$", va="center", fontsize=12)
ax1.axvline(1.0, color="black", lw=1.5, ls=":")
ax1.text(1.03, 2.44, "$1$ — 효과 없음", fontsize=10)
ax1.set_ylim(-0.62, 2.72)
ax1.set_yticks(np.arange(3)[::-1])
ax1.set_yticklabels(names, fontsize=10.5)
ax1.set_xlim(0, 1.62)
ax1.set_xlabel("상대위험")
ax1.set_title("① 교란 하나가 방향을 뒤집는다", fontsize=12)
ax1.grid(alpha=0.25, axis="x")

ax2.plot(RHOS, rr_rho, "o-", color="#1f77b4", lw=2.8, ms=8)
ax2.axhline(RR_TRUE, color="#2ca02c", lw=2.0, ls="--",
            label=f"참값 ${RR_TRUE}$")
ax2.axhline(1.0, color="black", lw=1.3, ls=":")
ax2.text(0.02, 1.02, "$1$ — 효과 없음", fontsize=10)
i9 = int(np.argmin(np.abs(RHOS - 0.9)))
ax2.annotate(f"$\\varrho = 0.9$ 에서도\n${rr_rho[i9]:.2f}$",
             xy=(RHOS[i9], rr_rho[i9]), xytext=(0.42, 1.06), fontsize=11,
             color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xlabel("대리변수가 $U$ 와 갖는 상관 $\\varrho$")
ax2.set_ylabel("보정 후 상대위험")
ax2.set_ylim(0.45, 1.42)
ax2.set_title("② 보정은 대리변수만큼만 듣는다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

labels = ["관찰\n(보정 없음)", "관찰\n($\\varrho=0.9$ 보정)", "무작위배정\n(보정 없음)"]
vv = [rr_crude, rr_rho[i9], rr_rand]
cc = ["#d62728", "#ff7f0e", "#2ca02c"]
ax3.bar(np.arange(3), vv, 0.58, color=cc)
for i, v in enumerate(vv):
    ax3.text(i, v + 0.03, f"${v:.3f}$", ha="center", fontsize=12)
ax3.axhline(RR_TRUE, color="#2ca02c", lw=2.0, ls="--",
            label=f"참값 ${RR_TRUE}$")
ax3.axhline(1.0, color="black", lw=1.3, ls=":")
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_ylim(0, 1.55)
ax3.set_ylabel("상대위험")
ax3.set_title("③ 무작위배정은 $U$ 를 몰라도 된다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch10/confounding/img/healthy_user.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
