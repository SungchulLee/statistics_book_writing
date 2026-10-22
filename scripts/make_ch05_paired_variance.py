r"""같은 자료로 잰 두 분산을 $F$ 로 견주면 — 독립이 아닌 두 표본.

이 쪽의 정리 1 은 두 표본이 **독립**인 정규모집단에서 나왔을 때 $S_1^2/S_2^2$ 가
정확히 $F$ 를 따른다고 적는다. 모형 둘을 **같은 시험집합**에 걸어 오차의 흩어짐을
견주는 자리에서는 그 독립이 깨진다 — 같은 문항이 양쪽에 들어가기 때문이다.

① 상관이 생기면 $F$ 검정은 보수적으로 어긋난다.
② 짝을 지어 보는 검정(피트먼–모건)은 상관이 클수록 **힘이 세진다**.
③ 같은 검정력을 내는 데 드는 표본 수가 자릿수로 갈린다.

만드는 파일: docs/ch05/applications/img/paired_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.05


def draw_pair(n, rho, ratio, rep):
    """상관 rho, 분산비 ratio 인 두 열을 rep 번."""
    z1 = RNG.standard_normal((rep, n))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * RNG.standard_normal((rep, n))
    return z1, z2 * np.sqrt(ratio)


def p_f(a, b):
    """독립을 가정한 F 검정 (양측)."""
    n = a.shape[1]
    f = a.var(1, ddof=1) / b.var(1, ddof=1)
    p = 2 * np.minimum(stats.f.cdf(f, n - 1, n - 1),
                       stats.f.sf(f, n - 1, n - 1))
    return np.minimum(p, 1.0)


def p_pm(a, b):
    """피트먼–모건 — corr(a+b, a-b) = 0 을 검정한다."""
    n = a.shape[1]
    u, v = a + b, a - b
    uc = u - u.mean(1, keepdims=True)
    vc = v - v.mean(1, keepdims=True)
    r = (uc * vc).sum(1) / np.sqrt((uc ** 2).sum(1) * (vc ** 2).sum(1))
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), n - 2)


# === ① 1종오류 ======================================================
RHOS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
N1, REP = 50, 100_000
print(f"① 분산이 같을 때의 1종오류 (n = {N1}, 반복 {REP:,})")
print(f"{'상관 rho':>10}{'F 검정':>12}{'피트먼–모건':>14}")
FA_F, FA_P = [], []
for r in RHOS:
    a, b = draw_pair(N1, r, 1.0, REP)
    FA_F.append(float(np.mean(p_f(a, b) < ALPHA)))
    FA_P.append(float(np.mean(p_pm(a, b) < ALPHA)))
    print(f"{r:>10.2f}{FA_F[-1]:>12.4f}{FA_P[-1]:>14.4f}")
print(f"   상관이 커질수록 F 는 {FA_F[0]:.3f} 에서 {FA_F[-1]:.4f} 로 내려간다 — "
      f"**이름값을 지키지 못하는 쪽이 보수적**이다.")
print("   피트먼–모건은 상관이 얼마든 0.05 에 붙어 있다.")

# === ② 검정력 =======================================================
RATIOS = [1.0, 1.1, 1.2, 1.35, 1.5, 2.0]
RHO2 = 0.9
print(f"\n② 검정력 (n = {N1}, rho = {RHO2}, 반복 {REP:,})")
print(f"{'분산비':>9}{'F 검정':>12}{'피트먼–모건':>14}")
PW_F, PW_P = [], []
for rt in RATIOS:
    a, b = draw_pair(N1, RHO2, rt, REP)
    PW_F.append(float(np.mean(p_f(a, b) < ALPHA)))
    PW_P.append(float(np.mean(p_pm(a, b) < ALPHA)))
    print(f"{rt:>9.2f}{PW_F[-1]:>12.4f}{PW_P[-1]:>14.4f}")
print(f"   분산비 1.2 에서 F 는 {PW_F[2]:.3f}, 피트먼–모건은 {PW_P[2]:.3f} 다.")

# === ③ 드는 표본 수 ==================================================
NS = [20, 50, 100, 200, 500, 1000, 2000]
RT3 = 1.2
print(f"\n③ 분산비 {RT3} 를 잡아내는 검정력 (rho = {RHO2})")
print(f"{'n':>7}{'F 검정':>12}{'피트먼–모건':>14}")
CF, CP = [], []
for n in NS:
    rep = REP if n <= 500 else 30_000
    a, b = draw_pair(n, RHO2, RT3, rep)
    CF.append(float(np.mean(p_f(a, b) < ALPHA)))
    CP.append(float(np.mean(p_pm(a, b) < ALPHA)))
    print(f"{n:>7}{CF[-1]:>12.4f}{CP[-1]:>14.4f}")


def need(curve):
    for i, v in enumerate(curve):
        if v >= 0.8:
            return NS[i]
    return None


print(f"   검정력 0.8 에 닿는 n — F {need(CF)}, 피트먼–모건 {need(CP)}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_F, C_P = "#d62728", "#1f77b4"

ax1.plot(RHOS, FA_F, "o-", color=C_F, lw=2.6, ms=9, label="$F$ 검정 (독립 가정)")
ax1.plot(RHOS, FA_P, "s-", color=C_P, lw=2.6, ms=9, label="피트먼–모건 (짝)")
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax1.text(0.02, ALPHA * 1.12, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax1.set_xlabel("두 오차열의 상관 $\\rho$")
ax1.set_ylabel("1종오류")
ax1.set_ylim(0, 0.075)
ax1.set_title(f"① 상관이 생기면 ($n = {N1}$)", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.plot(RATIOS, PW_F, "o-", color=C_F, lw=2.6, ms=9, label="$F$ 검정")
ax2.plot(RATIOS, PW_P, "s-", color=C_P, lw=2.6, ms=9, label="피트먼–모건")
ax2.axhline(ALPHA, color="#888888", lw=1.2, ls=":")
ax2.set_xlabel("참 분산비 $\\sigma_1^2/\\sigma_2^2$")
ax2.set_ylabel("검정력")
ax2.set_ylim(0, 1.02)
ax2.set_title(f"② 같은 자료를 쓸 때의 검정력 ($\\rho = {RHO2}$)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.semilogx(NS, CF, "o-", color=C_F, lw=2.6, ms=9, label="$F$ 검정")
ax3.semilogx(NS, CP, "s-", color=C_P, lw=2.6, ms=9, label="피트먼–모건")
ax3.axhline(0.8, color="#333333", lw=1.6, ls="--")
ax3.text(21, 0.82, "검정력 $0.8$", fontsize=10.5, color="#333333")
ax3.set_xlabel("시험집합의 크기 $n$")
ax3.set_ylabel("검정력")
ax3.set_xticks([20, 100, 1000])
ax3.set_xticklabels(["$20$", "$10^{2}$", "$10^{3}$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.02)
ax3.set_title(f"③ 분산비 ${RT3}$ 를 잡는 데 드는 $n$", fontsize=12)
ax3.legend(loc="center right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/applications/img/paired_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
