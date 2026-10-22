r"""17.2절 「그림으로 보는 통계학」 — 그 중앙값 한 줄이 생계급여의 문턱이다.

기준 중위소득은 **표본조사(가계금융복지조사, 약 $2$ 만 가구)에서 나온 중앙값**을
바탕으로 정해지고, 거기에 $32\%$ 를 곱한 선이 생계급여 선정 기준이 된다. 표본에서
나온 수이면 표집오차가 있는데, 중앙값에는 평균과 달리 표준오차의 간단한 공식이 없다.

① 표본 하나 — 소득 분포와 세 문턱.
② 중앙값의 붓스트랩 분포 — 공식 없이 잰 표준오차.
③ 그 폭이 $32\%$ 선에서 몇 가구에 해당하는가.

자료는 직접 만든 로그정규 모집단이다(실제 미시자료가 아니다). 중앙값만 고시된
$2026$ 년 $4$ 인 가구 기준 중위소득에 맞추었다.

만드는 파일: docs/ch17/bootstrap_ci/img/median_welfare_line.png
실행:  python3 scripts/make_ch17_median_welfare_line.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/bootstrap_ci/img/median_welfare_line.png"
RNG = np.random.default_rng(20261022)

MEDIAN = 6_494_738        # 2026년 4인 가구 기준 중위소득 (월, 원)
SIGMA = 0.8               # 로그소득의 표준편차 — 모양을 주려고 고른 값이다
N, B = 20_000, 4_000      # 가계금융복지조사 표본 규모와 붓스트랩 횟수
RATES = {"생계급여 32%": 0.32, "의료급여 40%": 0.40,
         "주거급여 48%": 0.48, "교육급여 50%": 0.50}

x = MEDIAN * np.exp(SIGMA * RNG.standard_normal(N))
med_hat = np.median(x)
print(f"모집단 중앙값 {MEDIAN:,} 원,  표본 중앙값 {med_hat:,.0f} 원")

# --- 중앙값의 붓스트랩 표준오차 ----------------------------------------
reps = np.empty(B)
CH = 200
for s in range(0, B, CH):
    idx = RNG.integers(0, N, (CH, N))
    reps[s:s + CH] = np.median(x[idx], axis=1)
se_boot = reps.std(ddof=1)
lo, hi = np.quantile(reps, [0.025, 0.975])
se_formula = 1 / (2 * (0.3989422804 / (MEDIAN * SIGMA)) * np.sqrt(N))
print(f"중앙값의 붓스트랩 SE = {se_boot:,.0f} 원   "
      f"(모집단을 알 때의 점근식 {se_formula:,.0f} 원)")
print(f"중앙값의 95% 붓스트랩 구간 = [{lo:,.0f}, {hi:,.0f}] 원  "
      f"(폭 {hi - lo:,.0f} 원)")

se_mean = x.std(ddof=1) / np.sqrt(N)
print(f"견주기 — 평균의 SE 는 공식으로 {se_mean:,.0f} 원")

# --- 32% 선이 흔들리는 폭과 그 안의 가구 ------------------------------
half = 1.959964 * se_boot
gate = 0.32 * MEDIAN
g_lo, g_hi = 0.32 * (MEDIAN - half), 0.32 * (MEDIAN + half)
share = float(np.mean((x > g_lo) & (x < g_hi)))
below = float(np.mean(x < gate))
print(f"\n생계급여 선정선 = 0.32 × 중앙값 = {gate:,.0f} 원")
print(f"중앙값이 ±{half:,.0f} 원 흔들리면 선정선은 "
      f"[{g_lo:,.0f}, {g_hi:,.0f}] 원 사이를 오간다 (폭 {g_hi - g_lo:,.0f} 원)")
print(f"그 띠 안에 든 가구의 비율 = {share * 100:.3f}%   "
      f"(선 아래 가구는 {below * 100:.2f}%)")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"
M = 1e6   # 백만원 단위로 그린다

ax = axes[0]
ax.hist(x / M, bins=np.linspace(0, 30, 121), color="0.78", edgecolor="white")
ax.axvline(med_hat / M, color=C1, lw=2)
ax.text(med_hat / M * 1.03, ax.get_ylim()[1] * 0.92,
        f"표본 중앙값\n{med_hat/M:.2f} 백만원", color=C1, fontsize=9)
for nm, r in RATES.items():
    ax.axvline(r * MEDIAN / M, color=C2, lw=1.1, ls="--")
ax.annotate("중앙값의 32 · 40 · 48 · 50%\n(생계 · 의료 · 주거 · 교육급여)",
            xy=(0.44 * MEDIAN / M, ax.get_ylim()[1] * 0.42),
            xytext=(9.5, ax.get_ylim()[1] * 0.60), color=C2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C2))
ax.set_xlim(0, 30)
ax.set_xlabel("가구 월소득 (백만원)")
ax.set_ylabel("가구 수")
ax.set_title(f"① 표본 {N:,} 가구와 네 문턱", fontsize=11)

ax = axes[1]
ax.hist(reps / M, bins=40, color=C1, alpha=0.8, density=True)
ax.axvline(lo / M, color="0.3", ls="--", lw=1.2)
ax.axvline(hi / M, color="0.3", ls="--", lw=1.2)
ax.set_xlabel("붓스트랩 표본의 중앙값 (백만원)")
ax.set_ylabel("밀도")
ax.set_title("② 중앙값의 붓스트랩 분포", fontsize=11)
ax.text(0.03, 0.95,
        f"붓스트랩 SE = {se_boot:,.0f} 원\n"
        f"95% 구간 폭 = {hi - lo:,.0f} 원\n"
        f"평균의 SE(공식) = {se_mean:,.0f} 원",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[2]
lo_x, hi_x = gate - 6 * (g_hi - g_lo), gate + 6 * (g_hi - g_lo)
sel = x[(x > lo_x) & (x < hi_x)]
ax.hist(sel / M, bins=40, color="0.78", edgecolor="white")
ax.axvspan(g_lo / M, g_hi / M, color=C2, alpha=0.30)
ax.axvline(gate / M, color=C2, lw=2)
ax.set_xlabel("가구 월소득 (백만원)")
ax.set_ylabel("가구 수")
ax.set_title("③ 선정선이 오가는 띠", fontsize=11)
ax.text(0.03, 0.95,
        f"선정선 {gate:,.0f} 원\n"
        f"띠 폭 {g_hi - g_lo:,.0f} 원\n"
        f"띠 안의 가구 {share*100:.2f}%",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
