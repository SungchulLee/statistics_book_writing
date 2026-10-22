r"""$\sqrt{T}$ 규칙은 독립을 가정한다 — 자기상관이 있으면 얼마나 틀리는가.

이 쪽 $\S 4$ 의 표는 「표준적인 연율화는 수익률이 i.i.d. 라고 가정한다」고 적어 두고
$\sigma_a = \sigma_d\sqrt{252}$ 를 쓴다. 그 가정이 깨지면 얼마나 틀리는지를 잰다.
같은 함정이 **평가 표본이 시간 순서로 묶여 있는** 자리에도 그대로 있다.

① AR(1) 수익률에서 $\sqrt{252}$ 가 참 연간 변동성에서 얼마나 벗어나는가.
② 평균의 표준오차도 같은 인자로 틀린다 — 유효표본수 $n(1-\phi)/(1+\phi)$.
③ 그래서 $95\%$ 라 말한 구간이 몇 퍼센트를 담는가.

만드는 파일: docs/ch05/non_normal/img/sqrt_t_rule.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
T = 252


def ar1(n, phi, rep):
    """정상 AR(1), 주변분산 1."""
    x = np.empty((rep, n))
    x[:, 0] = RNG.standard_normal(rep)
    s = np.sqrt(1 - phi ** 2)
    e = RNG.standard_normal((rep, n - 1)) * s
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t - 1]
    return x


PHIS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# === ① 연율화 ========================================================
REP1 = 4000
print(f"① 일별 분산 1 인 AR(1) 에서 {T} 일 합의 분산")
print(f"{'phi':>7}{'sqrt(T) 규칙':>14}{'참값 (모의)':>14}"
      f"{'비 (참/규칙)':>14}{'식 (1+phi)/(1-phi)':>20}")
RAT, FORM = [], []
for ph in PHIS:
    x = ar1(T, ph, REP1)
    v = float(x.sum(1).var())
    rule = float(T)
    RAT.append(np.sqrt(v / rule))
    FORM.append(np.sqrt((1 + ph) / (1 - ph)))
    print(f"{ph:>7.2f}{np.sqrt(rule):>14.3f}{np.sqrt(v):>14.3f}"
          f"{RAT[-1]:>14.4f}{FORM[-1]:>20.4f}")
print(f"   phi = 0.3 이면 참 연간 변동성이 규칙값의 {RAT[-1]:.3f} 배다 — "
      f"규칙이 참값의 {100 / RAT[-1]:.0f}% 만 말한다(**{100 * (1 - 1 / RAT[-1]):.0f}% "
      f"낮춰 부른다**).")
print(f"   phi = -0.3 이면 거꾸로 {RAT[0]:.3f} 배라 규칙이 "
      f"{100 / RAT[0] - 100:.0f}% **부풀려** 부른다.")

# === ② 유효표본수 ====================================================
N2, REP2 = 1000, 20_000
print(f"\n② 평균의 표준오차 (n = {N2:,}, 반복 {REP2:,})")
print(f"{'phi':>7}{'독립 가정 SE':>14}{'실제 SE':>12}{'비':>9}"
      f"{'n_eff/n = (1-phi)/(1+phi)':>27}")
SE_R, NEFF = [], []
for ph in PHIS:
    x = ar1(N2, ph, REP2)
    m = x.mean(1)
    se_true = float(m.std())
    se_iid = float((x.std(1, ddof=1) / np.sqrt(N2)).mean())
    SE_R.append(se_true / se_iid)
    NEFF.append((1 - ph) / (1 + ph))
    print(f"{ph:>7.2f}{se_iid:>14.5f}{se_true:>12.5f}{SE_R[-1]:>9.4f}"
          f"{NEFF[-1]:>27.4f}")
print(f"   phi = 0.3 에서 실제 표준오차가 {SE_R[-1]:.2f} 배다. "
      f"표본 {N2:,} 개가 사실상 {N2 * NEFF[-1]:.0f} 개 값이다.")

# === ③ 구간의 포함률 ================================================
NS = [100, 250, 1000, 4000]
PH3 = [0.0, 0.1, 0.2, 0.3]
REP3 = 20_000
print(f"\n③ 독립을 가정해 만든 95% 구간의 실제 포함률")
print(f"{'n':>7}" + "".join(f"{f'phi={p}':>12}" for p in PH3))
COV = {p: [] for p in PH3}
for n in NS:
    row = []
    for ph in PH3:
        x = ar1(n, ph, REP3)
        m = x.mean(1)
        hw = stats.t.ppf(0.975, n - 1) * x.std(1, ddof=1) / np.sqrt(n)
        c = float(np.mean(np.abs(m) <= hw))
        COV[ph].append(c)
        row.append(c)
    print(f"{n:>7}" + "".join(f"{v:>12.4f}" for v in row))
print("   n 을 키워도 돌아오지 않는다 — 자기상관은 표본크기로 고쳐지지 않는다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B = "#d62728", "#1f77b4"

ax1.plot(PHIS, RAT, "o", color=C_A, ms=10, label="모의한 비")
g = np.linspace(-0.35, 0.35, 200)
ax1.plot(g, np.sqrt((1 + g) / (1 - g)), "-", color="#333333", lw=2.0,
         label="$\\sqrt{(1+\\phi)/(1-\\phi)}$")
ax1.axhline(1.0, color="#888888", lw=1.4, ls="--")
ax1.text(-0.33, 1.01, "$\\sqrt{252}$ 규칙이 맞는 자리", fontsize=10.5,
         color="#555555")
ax1.set_xlabel("일별 수익률의 자기상관 $\\phi$")
ax1.set_ylabel("참 연간 변동성 / 규칙값")
ax1.set_title("① 규칙이 얼마나 벗어나는가", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.plot(PHIS, SE_R, "o", color=C_A, ms=10, label="실제 SE / 독립 가정 SE")
ax2.plot(g, np.sqrt((1 + g) / (1 - g)), "-", color="#333333", lw=2.0,
         label="$\\sqrt{(1+\\phi)/(1-\\phi)}$")
ax2.axhline(1.0, color="#888888", lw=1.4, ls="--")
ax2.set_xlabel("자기상관 $\\phi$")
ax2.set_ylabel("표준오차가 틀린 배수")
ax2.set_title(f"② 평균의 표준오차도 같은 인자다 ($n = {N2:,}$)"
              .replace(",", "{,}"), fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

for i, ph in enumerate(PH3):
    ax3.semilogx(NS, COV[ph], "o-", color=["#333333", "#9ecae1", "#4292c6",
                                           "#08519c"][i], lw=2.4, ms=9,
                 label=f"$\\phi = {ph}$")
ax3.axhline(0.95, color=C_A, lw=1.8, ls="--")
ax3.text(105, 0.955, "말한 $95\\%$", fontsize=10.5, color=C_A)
ax3.set_xlabel("표본크기 $n$")
ax3.set_ylabel("실제 포함률")
ax3.set_xticks([100, 1000])
ax3.set_xticklabels(["$10^{2}$", "$10^{3}$"])
ax3.minorticks_off()
ax3.set_ylim(0.82, 0.98)
ax3.set_title("③ $n$ 을 키워도 돌아오지 않는다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/non_normal/img/sqrt_t_rule.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
