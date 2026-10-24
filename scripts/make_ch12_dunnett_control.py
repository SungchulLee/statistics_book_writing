r"""대조군 하나를 모두가 나누어 쓰면, 비교들이 상관 $1/2$ 로 묶인다.

처리 $k$ 개와 대조군을 각 $n$ 씩 두면 $d_i = \bar y_i - \bar y_0$ 들은
$\operatorname{Cov}(d_i, d_j) = \sigma^2/n$, $\operatorname{Var}(d_i) = 2\sigma^2/n$
이라 상관이 정확히 $1/2$ 다. 본페로니는 이 상관을 모르고 최악을 가정해 깎는다.

① 임계값 — Dunnett 이 본페로니보다 낮고, 모든 쌍을 보는 Tukey 보다 훨씬 낮다.
② 그 차이가 검정력으로 얼마나 돌아오는가.
③ 대조군에 몇 배를 배정해야 하는가 (총 표본을 고정하고 직접 찾는다).

만드는 파일: docs/ch12/post_hoc/img/dunnett_control.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

ALPHA = 0.05
N = 10                       # 집단마다의 표본 수 (①②)
KS = np.arange(1, 9)         # 처리 수 (대조군은 따로)
DELTA = 1.2                  # 처리 하나만 이만큼 좋다 (표준편차 단위)
REPS_C = 120000              # 임계값을 잡는 데 쓰는 반복
REPS_P = 40000               # 검정력을 재는 데 쓰는 반복


def maxabs_t(k, n0, nt, reps, shift=0.0):
    """대조군과의 t 통계량 k 개 가운데 절댓값 최대. shift 는 처리 1 에만 준다."""
    c = RNG.standard_normal((reps, n0))
    t = RNG.standard_normal((reps, k, nt))
    t[:, 0, :] += shift
    m0 = c.mean(axis=1)
    mi = t.mean(axis=2)
    ss = ((c - m0[:, None]) ** 2).sum(axis=1) \
        + ((t - mi[:, :, None]) ** 2).sum(axis=(1, 2))
    nu = (n0 - 1) + k * (nt - 1)
    s = np.sqrt(ss / nu)
    tv = (mi - m0[:, None]) / (s[:, None] * np.sqrt(1.0 / nt + 1.0 / n0))
    return np.abs(tv).max(axis=1), nu


# === 상관이 정말 1/2 인가 (검산) =====================================
c = RNG.standard_normal((40000, N))
t = RNG.standard_normal((40000, 3, N))
d = t.mean(axis=2) - c.mean(axis=1)[:, None]
print("검산 — 대조군과의 차들 사이의 상관")
print(np.round(np.corrcoef(d.T), 4))

# === ① 임계값 ========================================================
print(f"\n① 양측 {ALPHA} 임계값 (집단마다 n = {N})")
print(f"{'처리 k':>7}{'Dunnett':>10}{'본페로니':>10}{'Tukey':>10}")
cv_d, cv_b, cv_t = [], [], []
for k in KS:
    mx, nu = maxabs_t(k, N, N, REPS_C)
    cv_d.append(float(np.quantile(mx, 1 - ALPHA)))
    cv_b.append(float(stats.t.isf(ALPHA / (2 * k), nu)))
    # Tukey 는 k+1 개 집단의 모든 쌍을 본다
    cv_t.append(float(stats.studentized_range.isf(ALPHA, k + 1, nu) / np.sqrt(2)))
    print(f"{k:>7}{cv_d[-1]:>10.3f}{cv_b[-1]:>10.3f}{cv_t[-1]:>10.3f}")
cv_d, cv_b, cv_t = map(np.array, (cv_d, cv_b, cv_t))

# === ② 검정력 ========================================================
print(f"\n② 처리 하나가 {DELTA}σ 좋을 때 그것을 잡아낼 확률")
print(f"{'처리 k':>7}{'Dunnett':>10}{'본페로니':>10}{'Tukey':>10}")
pw_d, pw_b, pw_t = [], [], []
for i, k in enumerate(KS):
    c0 = RNG.standard_normal((REPS_P, N))
    tr = RNG.standard_normal((REPS_P, k, N))
    tr[:, 0, :] += DELTA
    m0, mi = c0.mean(axis=1), tr.mean(axis=2)
    ss = ((c0 - m0[:, None]) ** 2).sum(axis=1) \
        + ((tr - mi[:, :, None]) ** 2).sum(axis=(1, 2))
    nu = (N - 1) + k * (N - 1)
    s = np.sqrt(ss / nu)
    t1 = np.abs((mi[:, 0] - m0) / (s * np.sqrt(2.0 / N)))
    pw_d.append(float((t1 > cv_d[i]).mean()))
    pw_b.append(float((t1 > cv_b[i]).mean()))
    pw_t.append(float((t1 > cv_t[i]).mean()))
    print(f"{k:>7}{pw_d[-1]:>10.3f}{pw_b[-1]:>10.3f}{pw_t[-1]:>10.3f}")

# === ③ 대조군에 얼마를 줄 것인가 =====================================
KA, TOTAL = 4, 100
NTS = np.arange(12, 24)          # 처리군마다의 표본 수 — 나머지가 대조군이다
REPS_A = 150000                  # ③ 은 봉우리가 평평해서 더 많이 돌린다
print(f"\n③ 처리 {KA} 개, 총 표본 {TOTAL} 명을 나눌 때 (√k = {np.sqrt(KA):.2f})")
print(f"{'배율':>6}{'n0':>5}{'nt':>5}{'임계값':>9}{'검정력':>9}")
alloc, pw_a = [], []
for nt in NTS:
    nt = int(nt)
    n0 = TOTAL - KA * nt
    if nt < 3 or n0 < 3:
        continue
    mx, nu = maxabs_t(KA, n0, nt, REPS_C)
    cv = float(np.quantile(mx, 1 - ALPHA))
    c0 = RNG.standard_normal((REPS_A, n0))
    tr = RNG.standard_normal((REPS_A, KA, nt))
    tr[:, 0, :] += DELTA
    m0, mi = c0.mean(axis=1), tr.mean(axis=2)
    ss = ((c0 - m0[:, None]) ** 2).sum(axis=1) \
        + ((tr - mi[:, :, None]) ** 2).sum(axis=(1, 2))
    s = np.sqrt(ss / nu)
    t1 = np.abs((mi[:, 0] - m0) / (s * np.sqrt(1.0 / nt + 1.0 / n0)))
    alloc.append(n0 / nt)
    pw_a.append(float((t1 > cv).mean()))
    print(f"{n0 / nt:>6.2f}{n0:>5}{nt:>5}{cv:>9.3f}{pw_a[-1]:>9.3f}")
alloc, pw_a = np.array(alloc), np.array(pw_a)
best = int(np.argmax(pw_a))
print(f"  최적 배율 {alloc[best]:.2f} (검정력 {pw_a[best]:.3f}), "
      f"√k = {np.sqrt(KA):.2f}")
eq = int(np.argmin(np.abs(alloc - 1.0)))
print(f"  똑같이 나눌 때 {alloc[eq]:.2f} 배 -> 검정력 {pw_a[eq]:.3f} "
      f"({pw_a[best] - pw_a[eq]:+.3f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.plot(KS, cv_t, "^-", color="#9467bd", lw=2.4, ms=8,
         label="Tukey (모든 쌍)")
ax1.plot(KS, cv_b, "s-", color="#d62728", lw=2.4, ms=8, label="본페로니")
ax1.plot(KS, cv_d, "o-", color="#1f77b4", lw=2.8, ms=9, label="Dunnett")
ax1.set_xlabel("처리 수 $k$ (대조군은 따로)")
ax1.set_ylabel(f"양측 ${ALPHA}$ 임계값")
ax1.set_title("① 상관을 아는 만큼 덜 깎는다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25)

ax2.plot(KS, pw_t, "^-", color="#9467bd", lw=2.4, ms=8, label="Tukey")
ax2.plot(KS, pw_b, "s-", color="#d62728", lw=2.4, ms=8, label="본페로니")
ax2.plot(KS, pw_d, "o-", color="#1f77b4", lw=2.8, ms=9, label="Dunnett")
ax2.annotate(f"$k = 8$ 에서\n${pw_d[-1] - pw_t[-1]:+.3f}$",
             xy=(8, (pw_d[-1] + pw_t[-1]) / 2), xytext=(5.5, 0.655),
             fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax2.set_xlabel("처리 수 $k$")
ax2.set_ylabel(f"${DELTA}\\sigma$ 짜리 효과를 잡아낼 확률")
ax2.set_title("② 그만큼 더 찾아낸다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(alloc, pw_a, "o-", color="#1f77b4", lw=2.6, ms=7)
ax3.axvline(np.sqrt(KA), color="#d62728", lw=2.0, ls="--",
            label=f"$\\sqrt{{k}} = {np.sqrt(KA):.0f}$")
ax3.axvline(1.0, color="#999999", lw=1.6, ls=":")
ax3.text(1.06, pw_a.min() + 0.004, "똑같이 나눌 때", fontsize=10,
         color="#555555")
flat = alloc[pw_a >= pw_a.max() - 0.003]     # 봉우리가 평평한 구간
ax3.axvspan(flat.min(), flat.max(), color="#ff7f0e", alpha=0.2,
            label=f"꼭대기에서 $0.003$ 안\n(${flat.min():.2f}$–${flat.max():.2f}$ 배)")
ax3.set_xlabel("대조군이 처리군의 몇 배인가")
ax3.set_ylabel(f"${DELTA}\\sigma$ 짜리 효과를 잡아낼 확률")
ax3.set_title(f"③ 처리 ${KA}$ 개, 총 ${TOTAL}$ 명을 나눈다면", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/post_hoc/img/dunnett_control.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
