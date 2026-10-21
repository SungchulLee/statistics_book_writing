r"""「정상 체온 $37.0$ °C」를 일표본 검정으로 무너뜨린 자리.

$1992$ 년 마코위악 등이 건강한 성인 $148$ 명을 재어 평균이 $36.8$ °C 라고 보고했다.
표준편차는 논문 초록에 없으므로, 같은 초록이 적은 **정상 상한 $37.7$ °C** 에서
되짚어 쓴다 — 상한을 상위 $2.5\%$ 자리로 보면 $s \approx (37.7-36.8)/1.96$ 이다.

① 일표본 $t$ 검정은 이 자리를 쉽게 이긴다.
② 그러나 평균의 구간과 **한 사람의 범위**는 전혀 다르다.
③ 하루 안의 변동이 두 교과서 값의 차이보다 크다.

만드는 파일: docs/ch09/one_sample_tests/img/body_temperature.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

MU0 = 37.0          # 분더리히의 값
XBAR = 36.8         # 마코위악 등(1992)이 보고한 평균
UPPER = 37.7        # 같은 논문이 보고한 정상 상한
N = 148
AMP = 0.5           # 같은 논문이 보고한 하루 변동의 평균 진폭

Z = stats.norm.isf(0.025)
S = (UPPER - XBAR) / Z              # 상한에서 되짚은 표준편차
print(f"상한 {UPPER} 에서 되짚은 표준편차 s = ({UPPER} - {XBAR})/{Z:.3f} "
      f"= {S:.3f} °C")

se = S / np.sqrt(N)
t = (XBAR - MU0) / se
p = 2 * stats.t.sf(abs(t), N - 1)
print(f"\n① n = {N}, SE = {se:.4f}, t = {t:.2f}, p = {p:.2e}")
ci = (XBAR - stats.t.isf(0.025, N - 1) * se, XBAR + stats.t.isf(0.025, N - 1) * se)
print(f"   평균의 95% 구간 [{ci[0]:.3f}, {ci[1]:.3f}] — "
      f"{MU0} 을 담지 않는다")

# === ② 평균의 구간 대 한 사람의 범위 =================================
NS = np.array([10, 30, 148, 1000, 10000])   # 100 은 148 과 눈금이 겹친다
half_mean = stats.norm.isf(0.025) * S / np.sqrt(NS)
half_one = np.full_like(half_mean, Z * S)
print(f"\n② 반너비 (°C)")
print(f"{'n':>8}{'평균의 구간':>14}{'한 사람의 범위':>16}{'비':>8}")
for n, a, b in zip(NS, half_mean, half_one):
    print(f"{n:>8}{a:>14.4f}{b:>16.4f}{b / a:>8.1f}")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

# ---- ① -------------------------------------------------------------
xs = np.linspace(MU0 - 5.2 * se, MU0 + 5.2 * se, 400)
ax1.plot(xs, stats.norm.pdf(xs, MU0, se), color="#9e9e9e", lw=2.4)
ax1.fill_between(xs, stats.norm.pdf(xs, MU0, se), color="#9e9e9e", alpha=0.3)
ax1.axvline(MU0, color="#333333", lw=1.6, ls=":")
ax1.text(MU0 + 0.004, stats.norm.pdf(MU0, MU0, se) * 1.02,
         "분더리히의 $37.0$", fontsize=10.5)
ax1.axvline(XBAR, color="#d62728", lw=2.6)
_m, _e = f"{p:.0e}".split("e")      # 지수표기는 쪼개서 조립한다
ax1.annotate(f"관측된 평균 ${XBAR}$\n$t = {t:.1f}$,  "
             f"$p = {_m} \\times 10^{{{int(_e)}}}$",
             xy=(XBAR, stats.norm.pdf(XBAR, MU0, se)),
             xytext=(MU0 - 4.9 * se, stats.norm.pdf(MU0, MU0, se) * 0.62),
             fontsize=10.5, color="#d62728", ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax1.set_xlabel("표본평균 ($^\\circ$C)")
ax1.set_ylabel("$H_0$ 아래 표본평균의 밀도")
ax1.set_title(f"① $n = {N}$ 이면 $0.2\\,^\\circ$C 도 크다", fontsize=12)
ax1.grid(alpha=0.25)

# ---- ② -------------------------------------------------------------
ax2.loglog(NS, half_mean, "o-", color="#1f77b4", lw=2.6, ms=7,
           label="평균의 $95\\%$ 구간 반너비")
ax2.loglog(NS, half_one, "s--", color="#d62728", lw=2.6, ms=7,
           label="한 사람의 $95\\%$ 범위 반너비")
i148 = int(np.where(NS == N)[0][0])
ax2.annotate(f"$n = {N}$ 에서\n${half_one[i148] / half_mean[i148]:.0f}$ 배 차이",
             xy=(N, np.sqrt(half_mean[i148] * half_one[i148])),
             xytext=(330, 0.06), fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax2.set_xlabel("표본 수 $n$")
ax2.set_ylabel("반너비 ($^\\circ$C)")
ax2.set_title("② 좁아지는 것은 평균뿐이다", fontsize=12)
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax2.set_yticks([0.01, 0.1, 1.0])
ax2.set_yticklabels(["$0.01$", "$0.1$", "$1$"])
ax2.minorticks_off()
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25, which="major")

# ---- ③ -------------------------------------------------------------
hrs = np.linspace(0, 24, 400)
# 논문이 보고한 것 — 6 시 최저, 16–18 시 최고, 평균 진폭 0.5 °C
curve = XBAR + (AMP / 2) * np.cos((hrs - 17) / 24 * 2 * np.pi)
ax3.plot(hrs, curve, color="#1f77b4", lw=2.8)
ax3.axhline(MU0, color="#333333", lw=1.4, ls=":")
ax3.text(0.3, MU0 + 0.012, "분더리히의 $37.0$", fontsize=10)
ax3.axhline(XBAR, color="#d62728", lw=1.4, ls="--")
ax3.text(0.3, XBAR - 0.045, f"마코위악 등의 ${XBAR}$", fontsize=10,
         color="#d62728")
ax3.annotate("", xy=(21.5, curve.max()), xytext=(21.5, curve.min()),
             arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.8))
ax3.text(20.9, XBAR - 0.21, f"하루 안의 변동\n${AMP}\\,^\\circ$C", ha="right",
         va="center", fontsize=11, color="#1f77b4")
ax3.annotate("", xy=(1.7, MU0), xytext=(1.7, XBAR),
             arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.8))
ax3.text(2.1, (MU0 + XBAR) / 2, "두 교과서 값의\n차이 $0.2\\,^\\circ$C",
         va="center", fontsize=11)
ax3.set_xlabel("시각")
ax3.set_ylabel("체온 ($^\\circ$C)")
ax3.set_title("③ 논쟁보다 큰 것이 하루 안에 있다", fontsize=12)
ax3.set_xticks([0, 6, 12, 18, 24])
ax3.set_xticklabels(["$0$", "$6$", "$12$", "$18$", "$24$"])
ax3.set_ylim(XBAR - AMP * 0.78, XBAR + AMP * 0.78)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch09/one_sample_tests/img/body_temperature.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

print(f"\n③ 하루 변동의 진폭 {AMP} °C 대 두 값의 차이 {MU0 - XBAR:.1f} °C "
      f"({AMP / (MU0 - XBAR):.1f} 배)")
