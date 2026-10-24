r"""무작위배정이 깨지면 표본을 아무리 키워도 고쳐지지 않는다.

$F$ 의 분자는 **처리 때문에 생긴 차이**만 담고 있어야 한다. 배정에 사람 손이 닿아
두 군의 출발점이 $\delta$ 만큼 어긋나면, $F$ 는 그 손까지 「효과」로 읽는다.

① 참 효과가 $0$ 인데도 기각률이 $n$ 과 함께 $1$ 로 간다 (무작위배정은 $0.05$).
② 그 까닭 — $E[F] \approx 1 + n\delta^2/(2\sigma^2)$ 이라 $n$ 에 **비례해** 커진다.
③ 편의가 있으면 RMSE 에 바닥 $\lvert\delta\rvert$ 가 생긴다. 그래서 짝지은 소수가
   편의 있는 다수를 이긴다 — Student 가 쌍둥이 $50$ 쌍을 말한 자리다.

만드는 파일: docs/ch12/anova_one_way/img/lanarkshire.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

SIG = 1.0            # 넉 달 동안의 성장의 표준편차
DELTA = 0.2          # 교사 보정이 만든 출발점의 어긋남 (성장 표준편차 단위)
NS = np.array([10, 30, 100, 300, 1000, 3000, 10000])
REPS = 4000
ALPHA = 0.05
PAIRS, RHO = 50, 0.9   # Student 가 말한 일란성 쌍둥이 50 쌍
NREAL = 10000          # 실제 실험에서 한 군의 인원

print(f"참 효과 0, 교사 보정이 만든 편의 {DELTA}σ, {REPS} 회")
print(f"{'군마다 n':>9}{'무작위 기각률':>14}{'보정 기각률':>13}{'평균 F':>10}"
      f"{'이론 E[F]':>11}")
rej_r, rej_b, fbar, theo = [], [], [], []
for n in NS:
    n = int(n)
    a = RNG.normal(0.0, SIG, (REPS, n))
    b = RNG.normal(0.0, SIG, (REPS, n))
    # 무작위배정 — 두 군의 출발점이 같다
    f0 = stats.f_oneway(a, b, axis=1)
    rej_r.append(float((f0.pvalue < ALPHA).mean()))
    # 교사 보정 — 우유군의 출발점이 delta 만큼 낮다
    f1 = stats.f_oneway(a - DELTA, b, axis=1)
    rej_b.append(float((f1.pvalue < ALPHA).mean()))
    fbar.append(float(f1.statistic.mean()))
    theo.append(1 + n * DELTA ** 2 / (2 * SIG ** 2))
    print(f"{n:>9}{rej_r[-1]:>14.3f}{rej_b[-1]:>13.3f}{fbar[-1]:>10.2f}"
          f"{theo[-1]:>11.2f}")
rej_r, rej_b, fbar, theo = map(np.array, (rej_r, rej_b, fbar, theo))

# === ③ RMSE 의 바닥 =================================================
NG = np.unique(np.concatenate([NS, np.logspace(1, 4.3, 60).astype(int)]))
rmse_bias = np.sqrt(DELTA ** 2 + 2 * SIG ** 2 / NG)       # 편의 있는 비짝 설계
rmse_rand = np.sqrt(2 * SIG ** 2 / NG)                    # 편의 없는 비짝 설계
se_pair = np.sqrt(2 * SIG ** 2 * (1 - RHO) / PAIRS)       # 짝지은 50 쌍
i_real = int(np.argmin(np.abs(NG - NREAL)))
print(f"\n③ 추정오차 (RMSE, σ 단위)")
print(f"  편의 {DELTA}σ 가 있는 설계, 군마다 {NREAL} 명 : "
      f"{rmse_bias[i_real]:.4f}   (바닥 {DELTA})")
print(f"  편의 없는 설계,        군마다 {NREAL} 명 : {rmse_rand[i_real]:.4f}")
print(f"  짝지은 쌍둥이 {PAIRS} 쌍 (ρ = {RHO})        : {se_pair:.4f}")
print(f"  -> 짝 {PAIRS} 쌍이 편의 있는 {2 * NREAL} 명보다 "
      f"{rmse_bias[i_real] / se_pair:.1f} 배 정확하다")
rho_need = 1 - PAIRS * (2 * SIG ** 2 / NREAL) / (2 * SIG ** 2)
print(f"  편의가 없었다면 짝 {PAIRS} 쌍이 이기는 데 ρ > {rho_need:.4f} 가 필요하다")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.semilogx(NS, rej_b, "o-", color="#d62728", lw=2.8, ms=8,
             label=f"교사 보정 (편의 ${DELTA}\\sigma$)")
ax1.semilogx(NS, rej_r, "s--", color="#1f77b4", lw=2.2, ms=7,
             label="무작위배정")
ax1.axhline(ALPHA, color="black", lw=1.4, ls=":")
ax1.text(950, 0.105, "명목 $\\alpha = 0.05$", fontsize=10)
ax1.axvline(NREAL, color="#999999", lw=1.6, ls="-.")
ax1.text(6200, 0.55, "실제 실험\n(군마다 $10{,}000$ 명)", fontsize=9.5,
         ha="right", color="#555555")
ax1.set_xticks(NS)
ax1.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax1.minorticks_off()
ax1.set_xlabel("군마다의 인원 $n$")
ax1.set_ylabel("참 효과가 $0$ 인데 기각할 확률")
ax1.set_ylim(0, 1.05)
ax1.set_title("① 더 모아도 고쳐지지 않는다", fontsize=12)
ax1.legend(loc="center left", fontsize=10)
ax1.grid(alpha=0.25)

ax2.loglog(NS, theo, "-", color="#333333", lw=2.4,
           label="이론값 $1 + n\\delta^2/(2\\sigma^2)$")
ax2.loglog(NS, fbar, "o", color="#d62728", ms=9, label="실제 $F$ 의 평균")
ax2.axhline(1.0, color="black", lw=1.2, ls=":")
ax2.text(150, 1.18, "$H_0$ 가 참이면 $1$ 이어야 한다", fontsize=10)
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax2.set_yticks([1, 10, 100])
ax2.set_yticklabels(["$1$", "$10$", "$100$"])
ax2.minorticks_off()
ax2.set_xlabel("군마다의 인원 $n$")
ax2.set_ylabel("$F$ 의 크기")
ax2.set_title("② $n$ 에 비례해 커진다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, which="major")

ax3.loglog(NG, rmse_bias, "-", color="#d62728", lw=2.8,
           label=f"교사 보정 (편의 ${DELTA}\\sigma$)")
ax3.loglog(NG, rmse_rand, "--", color="#1f77b4", lw=2.2,
           label="무작위배정")
ax3.axhline(se_pair, color="#2ca02c", lw=2.4, ls="-",
            label=f"쌍둥이 ${PAIRS}$ 쌍 ($\\varrho = {RHO}$)")
ax3.axhline(DELTA, color="#d62728", lw=1.1, ls=":")
ax3.text(420, DELTA * 1.3, f"바닥 ${DELTA}\\sigma$ — 편의는 줄지 않는다",
         fontsize=10, color="#d62728")
ax3.plot(NREAL, rmse_bias[i_real], "*", color="#d62728", ms=20)
ax3.set_xticks([10, 100, 1000, 10000])
ax3.set_xticklabels(["$10$", "$100$", "$1{,}000$", "$10{,}000$"], fontsize=9.5)
ax3.set_yticks([0.01, 0.1, 1.0])
ax3.set_yticklabels(["$0.01$", "$0.1$", "$1$"])
ax3.minorticks_off()
ax3.set_xlabel("군마다의 인원 $n$")
ax3.set_ylabel("효과 추정의 오차 (RMSE, $\\sigma$ 단위)")
ax3.set_title("③ 그래서 $50$ 쌍이 $2$ 만 명을 이긴다", fontsize=12)
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch12/anova_one_way/img/lanarkshire.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
