r"""답을 바꾼 사람만 말을 한다 — 패널이 좁은 구간을 얻는 까닭.

이 쪽은 $p_1 - p_2$ 의 McNemar 구간을 세우고, 정보가 **불일치 짝 $b + c$ 에만** 있다고
적는다. 그 사실의 뒷면이 덜 알려져 있다 — **사람들이 생각을 잘 바꾸지 않을수록 구간이
좁아진다.** 패널 조사가 같은 사람을 다시 찾아가는 까닭이 그것이다.

① 같은 $600$ 명에게 두 번 묻기 vs 따로 $600$ 명씩 두 번 묻기. 바꾸는 비율을 키우며.
② 두 설계가 뒤집히는 자리 — 절반이 바꾸면 같아진다.
③ 「$b + c \ge 10$ 이면 Wald 를 써도 된다」는 지침이 실제로 $95\%$ 를 주는가.

만드는 파일: docs/ch08/paired_sample_intervals/img/panel_discordant.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
Z = stats.norm.ppf(0.975)
N_PANEL = 600          # 에리 카운티 패널의 크기
REP = 40_000


def cells(n, pi, delta, rep):
    """불일치 비율 pi, 참 차이 delta 인 2x2 표를 rep 개 만든다."""
    b = (pi + delta) / 2
    c = (pi - delta) / 2
    p11 = (1 - pi) / 2                 # 둘 다 찬성
    p00 = 1 - p11 - b - c              # 둘 다 반대
    counts = RNG.multinomial(n, [p11, b, c, p00], size=rep)
    return counts[:, 1], counts[:, 2]  # b, c


def paired_hw(n, pi, delta, rep=REP):
    """McNemar Wald 구간의 반폭 (쪽의 식 그대로)."""
    b, c = cells(n, pi, delta, rep)
    se = np.sqrt(np.maximum(b + c - (b - c) ** 2 / n, 0)) / n
    return Z * se, (b - c) / n, b + c


def indep_hw(n, p1, p2, rep=REP):
    """서로 다른 사람 n 명씩 두 번 — 독립 두 비율 구간의 반폭."""
    x1 = RNG.binomial(n, p1, rep) / n
    x2 = RNG.binomial(n, p2, rep) / n
    se = np.sqrt(x1 * (1 - x1) / n + x2 * (1 - x2) / n)
    return Z * se


# === ① 바꾸는 비율을 키우며 ==========================================
PIS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
DELTA = 0.02
P_BASE = 0.50
print(f"① 패널 {N_PANEL} 명을 두 번 vs 따로 {N_PANEL} 명씩 두 번 "
      f"(참 차이 {DELTA}, 바탕 비율 {P_BASE})")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   바꾼 비율   대응 반폭   독립 반폭      비   불일치 짝")
HP, HI, MM = [], [], []
for pi in PIS:
    hw_p, _, m = paired_hw(N_PANEL, pi, DELTA)
    hw_i = indep_hw(N_PANEL, P_BASE + DELTA / 2, P_BASE - DELTA / 2)
    HP.append(hw_p.mean())
    HI.append(hw_i.mean())
    MM.append(m.mean())
    print(f"   {pi:>9.2f}   {HP[-1]:>9.4f}   {HI[-1]:>9.4f}   "
          f"{HI[-1] / HP[-1]:>5.2f}   {MM[-1]:>8.1f}")
print(f"   {int(100 * PIS[1])}% 만 바꾸면 대응 쪽 구간이 "
      f"{HI[1] / HP[1]:.1f} 배 좁다.")
print("   **적게 바꿀수록 좁아진다** — 바꾸지 않은 사람들이 차이를 못 박아 준다.")

# === ② 뒤집히는 자리 =================================================
print(f"\n② 두 설계가 같아지는 자리")
print("   대응 SE ~ sqrt(pi/n),  독립 SE ~ sqrt(2p(1-p)/n)")
print(f"   p = 0.5 이면 2p(1-p) = 0.5 이므로 **pi = 0.5 에서 같아진다.**")
for pi in PIS:
    tag = "대응이 좁다" if pi < 0.5 else ("같다" if pi == 0.5 else "독립이 좁다")
    print(f"   바꾼 비율 {pi:.2f} → {tag}")

# === ③ b + c >= 10 지침 ==============================================
TARGET_M = [4, 6, 8, 10, 15, 20, 30, 50]
print(f"\n③ 불일치 짝 수에 따른 Wald 구간의 실제 포함률 (이름값 95%)")
print("   불일치 짝   실제 포함률   정확 이항 구간")
COV_W, COV_E = [], []
for m_target in TARGET_M:
    # 불일치 짝의 기댓값이 m_target 이 되도록 n 을 맞춘다
    pi = 0.10
    n = int(round(m_target / pi))
    b, c = cells(n, pi, DELTA, REP)
    m = b + c
    ok = m > 0
    se = np.sqrt(np.maximum(m - (b - c) ** 2 / n, 0)) / n
    hw = Z * se
    est = (b - c) / n
    cov_w = float(np.mean(np.abs(est[ok] - DELTA) <= hw[ok]))
    # 정확 구간: b | m ~ Bin(m, theta), theta = b/(b+c) 를 Clopper-Pearson 으로
    lo = stats.beta.ppf(0.025, b[ok], m[ok] - b[ok] + 1)
    hi = stats.beta.ppf(0.975, b[ok] + 1, m[ok] - b[ok])
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.nan_to_num(hi, nan=1.0)
    d_lo = (2 * lo - 1) * m[ok] / n
    d_hi = (2 * hi - 1) * m[ok] / n
    cov_e = float(np.mean((d_lo <= DELTA) & (DELTA <= d_hi)))
    COV_W.append(cov_w)
    COV_E.append(cov_e)
    print(f"   {m_target:>9}   {cov_w:>11.4f}   {cov_e:>13.4f}")
print("   지침이 말하는 10 언저리에서 Wald 가 이름값에 못 미치는지 본다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_P, C_I = "#1f77b4", "#d62728"

ax1.plot(PIS, HP, "o-", color=C_P, lw=2.8, ms=10,
         label=f"대응 — 같은 ${N_PANEL}$ 명에게 두 번")
ax1.plot(PIS, HI, "s-", color=C_I, lw=2.8, ms=10,
         label=f"독립 — 따로 ${N_PANEL}$ 명씩")
ax1.axvline(0.5, color="#333333", lw=1.6, ls=":")
ax1.text(0.515, 0.028, "절반이 바꾸면\n두 설계가 같다", fontsize=10,
         color="#333333")
ax1.set_xlabel("생각을 바꾼 사람의 비율")
ax1.set_ylabel("$95\\%$ 구간의 반폭")
ax1.set_title("① 적게 바꿀수록 좁아진다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ratio = [i / p for p, i in zip(HP, HI)]
ax2.plot(PIS, ratio, "o-", color=C_P, lw=2.8, ms=10)
ax2.axhline(1.0, color="#333333", lw=1.8, ls="--")
ax2.axvline(0.5, color="#333333", lw=1.6, ls=":")
ax2.fill_between([0, 0.5], 1.0, max(ratio) * 1.05, color=C_P, alpha=0.10)
ax2.text(0.06, max(ratio) * 0.88, "대응이 이긴다", fontsize=11, color=C_P)
ax2.set_xlabel("생각을 바꾼 사람의 비율")
ax2.set_ylabel("독립 반폭 / 대응 반폭")
ax2.set_xlim(0, 0.72)
ax2.set_ylim(0.8, max(ratio) * 1.05)
ax2.set_title("② 몇 배 좁은가", fontsize=12)
ax2.grid(alpha=0.25)

ax3.plot(TARGET_M, COV_W, "o-", color=C_I, lw=2.8, ms=10,
         label="Wald (쪽의 식)")
ax3.plot(TARGET_M, COV_E, "s-", color=C_P, lw=2.6, ms=9,
         label="정확 이항 구간")
ax3.axhline(0.95, color="#333333", lw=1.8, ls="--")
ax3.axvline(10, color="#2ca02c", lw=2.0, ls=":")
ax3.text(10.6, 0.905, "지침의 $b + c = 10$", fontsize=10.5, color="#2ca02c")
ax3.set_xlabel("불일치 짝의 수 $b + c$")
ax3.set_ylabel("실제 포함률")
ax3.set_ylim(0.88, 1.005)
ax3.set_title("③ 지침은 지켜지는가", fontsize=12)
ax3.legend(loc="lower right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/paired_sample_intervals/img/panel_discordant.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
