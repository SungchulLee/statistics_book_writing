r"""1.2절 「그림으로 보는 통계학」 — 봉투에 든 난수표.

$1948$ 년 영국 의학연구위원회의 스트렙토마이신 결핵 시험은 배정을 **난수표**로 하고
그 계열을 **봉투에 봉해** 아무도 미리 알 수 없게 했다. 환자 $107$ 명, 여섯 달 뒤
사망이 치료군 $55$ 명 중 $4$ 명, 대조군 $52$ 명 중 $14$ 명이었다.

① 그 표를 지금 읽으면 — 위험차와 구간.
② 작은 시험이지만 — 귀무분포에서 그 차이가 놓인 자리.
③ 봉투가 사는 것 — 은폐가 깨지면 **효과 없는 약으로도** 그만한 차이가 만들어진다.

만드는 파일: docs/ch01/classical/img/streptomycin_envelope.png
실행:  python3 scripts/make_ch01_streptomycin.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats

OUT = "docs/ch01/classical/img/streptomycin_envelope.png"
RNG = np.random.default_rng(20261022)

N_S, D_S = 55, 4         # 스트렙토마이신 + 침상안정
N_C, D_C = 52, 14        # 침상안정만
p_s, p_c = D_S / N_S, D_C / N_C
rd = p_s - p_c
se = np.sqrt(p_s * (1 - p_s) / N_S + p_c * (1 - p_c) / N_C)
lo, hi = rd - 1.96 * se, rd + 1.96 * se
odds, p_fisher = stats.fisher_exact([[D_S, N_S - D_S], [D_C, N_C - D_C]])
print(f"① 사망률 {100*p_s:.1f}% ({D_S}/{N_S}) 대 {100*p_c:.1f}% ({D_C}/{N_C})")
print(f"   위험차 {100*rd:+.1f} 백분율점, 95% 구간 [{100*lo:+.1f}, {100*hi:+.1f}]")
print(f"   오즈비 {odds:.3f},  피셔 정확검정 p = {p_fisher:.4f}")

# --- ② 귀무분포 — 표를 주변합에 묶어 두고 다시 섞는다 ------------------
TOT, DEAD = N_S + N_C, D_S + D_C
k = np.arange(0, DEAD + 1)
pmf = stats.hypergeom.pmf(k, TOT, DEAD, N_S)
rd_grid = k / N_S - (DEAD - k) / N_C
print(f"\n② 귀무분포에서 관측값만큼 치우칠 확률(양쪽) = "
      f"{pmf[np.abs(rd_grid) >= abs(rd) - 1e-12].sum():.4f}")

# --- ③ 은폐가 깨지면 ----------------------------------------------------
# 약은 **아무 효과가 없다**. 다만 의사가 예후가 나쁜 환자를 대조군 쪽으로
# 비율 δ 만큼 밀어 넣는다. 중증은 사망위험 0.40, 경증은 0.08 로 둔다.
P_SEV, R_SEV, R_MILD = 0.45, 0.40, 0.08
REPS = 4000
DELTAS = np.linspace(0, 0.6, 13)
mean_rd, q10, q90 = [], [], []
for d in DELTAS:
    out = np.empty(REPS)
    for r in range(REPS):
        sev = RNG.random(TOT) < P_SEV
        # 기본은 반반, 중증이면 대조군으로 갈 확률이 δ 만큼 커진다
        p_ctrl = np.where(sev, 0.5 + d / 2, 0.5 - d / 2)
        ctrl = RNG.random(TOT) < p_ctrl
        risk = np.where(sev, R_SEV, R_MILD)          # 약의 효과는 0 이다
        died = RNG.random(TOT) < risk
        if ctrl.sum() == 0 or (~ctrl).sum() == 0:
            out[r] = np.nan
            continue
        out[r] = died[~ctrl].mean() - died[ctrl].mean()
    mean_rd.append(np.nanmean(out))
    q10.append(np.nanquantile(out, 0.10))
    q90.append(np.nanquantile(out, 0.90))
    print(f"③ δ = {d:.2f}:  관측되는 위험차 평균 {100*mean_rd[-1]:+6.2f} 백분율점  "
          f"(가운데 80% {100*q10[-1]:+.1f} ~ {100*q90[-1]:+.1f})")
mean_rd, q10, q90 = map(np.array, (mean_rd, q10, q90))
hit = DELTAS[np.argmin(np.abs(mean_rd - rd))]
print(f"\n   실제 시험의 {100*rd:+.1f} 백분율점은 δ ≈ {hit:.2f} 로도 만들어진다")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
ax.bar([0, 1], [100 * p_s, 100 * p_c], 0.5, color=[C1, C2])
for i, (v, d, n) in enumerate([(p_s, D_S, N_S), (p_c, D_C, N_C)]):
    ax.text(i, 100 * v + 0.8, f"{100*v:.1f}%\n({d}/{n})", ha="center", fontsize=10)
ax.set_xticks([0, 1])
ax.set_xticklabels(["스트렙토마이신\n+ 침상안정", "침상안정만"], fontsize=10)
ax.set_ylabel("여섯 달 안의 사망률 (%)")
ax.set_ylim(0, 34)
ax.set_title("① 환자 107 명의 표", fontsize=11)
ax.text(0.5, 31, f"위험차 {100*rd:+.1f} 백분율점\n"
        f"95% 구간 [{100*lo:+.1f}, {100*hi:+.1f}]\n"
        f"피셔 정확검정 p = {p_fisher:.4f}",
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[1]
keep = pmf > 1e-5
ax.bar(100 * rd_grid[keep], pmf[keep], width=1.4, color="0.72")
ax.bar(100 * rd_grid[keep][np.abs(rd_grid[keep]) >= abs(rd) - 1e-12],
       pmf[keep][np.abs(rd_grid[keep]) >= abs(rd) - 1e-12], width=1.4, color=C2)
ax.axvline(100 * rd, color=C1, lw=2)
ax.text(100 * rd - 1.5, pmf.max() * 0.85, f"관측값 {100*rd:+.1f}", color=C1,
        fontsize=9, ha="right")
ax.set_xlabel("위험차 (백분율점)")
ax.set_ylabel("확률")
ax.set_title("② 약이 듣지 않았다면 — 귀무분포", fontsize=11)

ax = axes[2]
ax.fill_between(DELTAS, 100 * q10, 100 * q90, color=C2, alpha=0.18,
                label="가운데 80%")
ax.plot(DELTAS, 100 * mean_rd, "o-", color=C2, lw=2, label="관측되는 위험차 평균")
ax.axhline(100 * rd, color=C1, lw=1.8, ls="--")
ax.text(0.02, 100 * rd - 1.2, f"실제 시험이 본 {100*rd:+.1f} 백분율점", color=C1,
        fontsize=9, va="top")
ax.axhline(0, color="0.4", lw=1)
ax.set_xlabel("예후가 나쁜 환자를 대조군으로 미는 정도  δ")
ax.set_ylabel("관측되는 위험차 (백분율점)")
ax.set_title("③ 약의 효과가 0 인데도", fontsize=11)
ax.legend(fontsize=9, loc="lower left")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
