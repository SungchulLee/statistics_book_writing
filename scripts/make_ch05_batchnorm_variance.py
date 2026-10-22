r"""분산 추정이 흔들리면 눈금이 흔들린다 — 정규화 층이 배치를 떠난 까닭.

이 쪽 개요가 「분산 추정값은 평균 추정값보다 훨씬 심하게 요동친다」로 시작한다.
신경망의 **정규화 층**은 바로 그 $S^2$ 로 값을 나눈다. 그러면 나누는 눈금 자체가
표본마다 달라진다 — 몇 개로 재느냐가 전부다.

① $S$ 의 상대표준편차는 $1/\sqrt{2(n-1)}$ 쯤이다. $\bar X$ 쪽과 견준다.
② 그 눈금으로 나누면 같은 입력이 배치마다 다른 값이 된다.
③ 축을 바꾸면 $n$ 이 바뀐다 — 배치 $8$ 개와 특징 $1{,}024$ 개의 차이.

만드는 파일: docs/ch05/applications/img/batchnorm_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
REP = 200_000

# === ① S 와 X_bar 의 흔들림 =========================================
NS = np.array([2, 3, 4, 8, 16, 32, 64, 128, 256, 1024])
print("① 정규모집단에서 n 개로 잰 눈금이 얼마나 흔들리는가")
print(f"{'n':>7}{'S 의 상대 sd':>14}{'1/sqrt(2(n-1))':>17}{'X_bar 의 sd':>14}")
RSD_S, RSD_M = [], []
for n in NS:
    x = RNG.standard_normal((REP, n))
    s = x.std(axis=1, ddof=1)
    RSD_S.append(float(s.std() / s.mean()))
    RSD_M.append(float(x.mean(axis=1).std()))
    print(f"{n:>7}{RSD_S[-1]:>14.4f}{1 / np.sqrt(2 * (n - 1)):>17.4f}"
          f"{RSD_M[-1]:>14.4f}")
print("   n = 2 에서 눈금이 표본마다 60% 넘게 흔들린다. "
      "n = 1024 라야 2% 로 내려온다.")

# === ② 그 눈금으로 나누면 ============================================
print("\n② 같은 입력 하나를 배치 통계로 정규화하면")
print(f"{'배치 크기 m':>12}{'정규화된 값의 sd':>18}{'이상값(|z|>3) 비율':>20}")
MS = [2, 4, 8, 16, 32, 64, 256]
SD_Z, OUT_Z = [], []
for m in MS:
    x = RNG.standard_normal((REP, m))
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1, keepdims=True)
    z = (x - mu) / sd
    SD_Z.append(float(z[:, 0].std()))
    OUT_Z.append(float(np.mean(np.abs(z[:, 0]) > 3)))
    print(f"{m:>12}{SD_Z[-1]:>18.4f}{OUT_Z[-1]:>20.5f}")
print("   m 이 작으면 정규화된 값의 폭이 1 보다 **작다** — 자기 자신이 평균과 "
      "표준편차에 들어가 있어서다.")
print(f"   m = 2 면 z 가 ±1/sqrt(2) = ±{1 / np.sqrt(2):.4f} 두 값밖에 될 수 없다 "
      f"(일반으로 |z| <= (m-1)/sqrt(m)).")
for m in (2, 4, 8):
    x = RNG.standard_normal((20000, m))
    z = (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)
    print(f"      m = {m:>3} : 재어 본 |z| 의 최댓값 {np.abs(z).max():.4f}, "
          f"한계 {(m - 1) / np.sqrt(m):.4f}")

# === ③ 축을 바꾸면 n 이 바뀐다 =======================================
print("\n③ 같은 공식을 어느 축에 거느냐")
CASES = [("배치 $2$", 2), ("배치 $8$", 8), ("배치 $32$", 32),
         ("특징 $256$", 256), ("특징 $1{,}024$", 1024)]
BAR = []
for nm, n in CASES:
    r = 1 / np.sqrt(2 * (n - 1))
    BAR.append(r)
    print(f"   {nm.replace('$', '').replace('{,}', ','):>12} : "
          f"눈금의 상대 sd {100 * r:>6.2f}%")
print("   같은 식인데 n 이 배치 크기냐 특징 수냐에 따라 자릿수가 달라진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_M, C_L = "#d62728", "#1f77b4", "#2ca02c"

ax1.loglog(NS, RSD_S, "o", color=C_S, ms=9, label="$S$ 의 상대 표준편차")
ax1.loglog(NS, 1 / np.sqrt(2 * (NS - 1)), "-", color="#333333", lw=2.0,
           label="$1/\\sqrt{2(n-1)}$")
ax1.loglog(NS, RSD_M, "s--", color=C_M, lw=2.0, ms=7,
           label="$\\bar X$ 의 표준편차 ($1/\\sqrt{n}$)")
ax1.set_xlabel("몇 개로 재는가 ($n$)")
ax1.set_ylabel("흔들리는 정도")
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax1.set_yticks([0.01, 0.1, 1])
ax1.set_yticklabels(["$1\\%$", "$10\\%$", "$100\\%$"])
ax1.minorticks_off()
ax1.set_title("① 눈금은 평균보다 심하게 흔들린다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25)

ax2.semilogx(MS, SD_Z, "o-", color=C_S, lw=2.6, ms=9,
             label="정규화된 값의 표준편차")
ax2.axhline(1.0, color="#333333", lw=1.6, ls="--")
ax2.text(2.1, 1.02, "$1$ 이 되기를 바라는 값", fontsize=10.5, color="#333333")
ax2.set_xlabel("배치 크기 $m$")
ax2.set_ylabel("정규화 뒤의 표준편차")
ax2.set_xticks([2, 10, 100])
ax2.set_xticklabels(["$2$", "$10$", "$10^{2}$"])
ax2.minorticks_off()
ax2.set_ylim(0.6, 1.08)
ax2.annotate(f"$m = 2$ 에서 ${SD_Z[0]:.3f}$", xy=(2, SD_Z[0]),
             xytext=(4.5, 0.70), fontsize=10.5, color=C_S,
             arrowprops=dict(arrowstyle="->", color=C_S, lw=1.1))
ax2.set_title("② 나눈 값이 바라던 눈금이 아니다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10.5)
ax2.grid(alpha=0.25)

cols = [C_S, C_S, C_S, C_L, C_L]
ax3.bar(range(len(CASES)), [100 * b for b in BAR], 0.58, color=cols)
for i, b in enumerate(BAR):
    ax3.text(i, 100 * b * 1.15, f"${100 * b:.2f}\\%$", ha="center",
             fontsize=10.5)
ax3.set_yscale("log")
ax3.set_xticks(range(len(CASES)))
ax3.set_xticklabels([c[0] for c in CASES], fontsize=10)
ax3.set_ylabel("눈금의 상대 표준편차")
ax3.set_yticks([1, 10, 100])
ax3.set_yticklabels(["$1\\%$", "$10\\%$", "$100\\%$"])
ax3.minorticks_off()
ax3.set_ylim(1.2, 180)
ax3.set_title("③ 같은 식, 다른 축", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/batchnorm_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
