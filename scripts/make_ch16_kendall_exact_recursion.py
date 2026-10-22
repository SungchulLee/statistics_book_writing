r"""16.6 Kendall의 타우(재론) 쪽의 상자에 들어가는 그림을 만든다.

비부합쌍의 수는 순열의 뒤바뀜(inversion) 수와 같고, 뒤바뀜 수의 분포는 되풀이식으로
쌓인다. 그래서 tau 의 정확 귀무분포는 n 이 커도 바로 나온다 — r_s 가 n = 10 에서
멈춰야 했던 것과 대조된다. 그 정확분포로 「n >= 10 이면 정규근사」를 검사한다.

만드는 파일:
  ch16/correlation/img/kendall_exact_recursion.png

실행:  python3 scripts/make_ch16_kendall_exact_recursion.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools
import math
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/correlation/img/"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
ZC = stats.norm.ppf(1 - ALPHA / 2)


def inversion_pmf(n):
    """뒤바뀜 수 D 의 확률분포. I_n = I_{n-1} 을 길이 n 의 창으로 밀며 더한 것."""
    pmf = np.array([1.0])
    for m in range(2, n + 1):
        pad = np.zeros(len(pmf) + m - 1)
        csum = np.concatenate(([0.0], np.cumsum(pmf)))
        for k in range(len(pad)):
            lo = max(0, k - m + 1)
            pad[k] = csum[min(k, len(pmf) - 1) + 1] - csum[lo]
        pmf = pad / m
    return pmf                      # 이미 확률로 정규화되어 있다


def tau_null(n):
    """(tau 값, 확률) — 정확 귀무분포."""
    pmf = inversion_pmf(n)
    d = np.arange(len(pmf))
    tau = 1.0 - 4.0 * d / (n * (n - 1))
    return tau, pmf


def normal_size(n):
    """Z = S / sqrt(Var S) 를 그대로 썼을 때의 실제 양측 크기."""
    tau, pmf = tau_null(n)
    S = tau * (n * (n - 1) / 2)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    return pmf[np.abs(S) / np.sqrt(var) > ZC].sum()


# ------------------------------------------------- 되풀이식이 맞는지 전수로 확인
for n in (6, 8):
    base = np.arange(n)
    exact = np.zeros(n * (n - 1) // 2 + 1)
    for p in itertools.permutations(base):
        p = np.array(p)
        inv = sum((p[i] > p[j]) for i in range(n) for j in range(i + 1, n))
        exact[inv] += 1
    exact /= exact.sum()
    print(f"n = {n}: 되풀이식과 {math.factorial(n):,} 가지 전수 열거의 "
          f"최대 어긋남 = {np.abs(exact - inversion_pmf(n)).max():.3e}")

# ------------------------------------------------- Daniels 의 부등식도 확인
n = 8
base = np.arange(1, n + 1)
perms = np.array(list(itertools.permutations(base)), dtype=np.int16)
d2 = ((perms - base) ** 2).sum(axis=1)
rs = 1.0 - 6.0 * d2 / (n * (n * n - 1))
inv = np.array([sum((p[i] > p[j]) for i in range(n) for j in range(i + 1, n))
                for p in perms])
tau8 = 1.0 - 4.0 * inv / (n * (n - 1))
dan = 3 * tau8 - 2 * rs
print(f"n = 8 의 {len(perms):,} 가지에서 3*tau - 2*r_s 의 범위 = "
       f"[{dan.min():.4f}, {dan.max():.4f}]   (Daniels: [-1, 1])")

# ------------------------------------------------- 크기를 n 에 대해 잰다
NS = list(range(4, 61))
sizes = np.array([normal_size(n) for n in NS])
print("\n  n   정규근사의 실제 크기")
for n in [4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]:
    print(f"{n:3d}      {normal_size(n):.4f}")
big = [n for n in NS if n >= 10]
print(f"n >= 10 구간에서 실제 크기의 범위 = "
      f"[{min(normal_size(n) for n in big):.4f}, "
      f"{max(normal_size(n) for n in big):.4f}]")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4))

# ------------------------------------------------- 왼쪽: n = 8 과 n = 40
for n_show, color, fill, off in [(8, BLUE, BLUE_F, 0), (40, GREEN, "#DCEDC8", 0)]:
    tau, pmf = tau_null(n_show)
    S = tau * (n_show * (n_show - 1) / 2)
    z = S / np.sqrt(n_show * (n_show - 1) * (2 * n_show + 5) / 18.0)
    stepz = z[0] - z[1]
    axL.bar(z, pmf / stepz, width=stepz * 0.95, color=fill, edgecolor=color,
            linewidth=0.3, zorder=2,
            label=f"$n = {n_show}$ 정확분포 (되풀이식)")
zg = np.linspace(-4, 4, 600)
axL.plot(zg, stats.norm.pdf(zg), color=ORANGE, lw=2.2, zorder=4,
         label="$\\mathcal{N}(0,1)$")
axL.axvline(ZC, color=RED, lw=1.1, ls="--", zorder=3)
axL.axvline(-ZC, color=RED, lw=1.1, ls="--", zorder=3)
axL.set_xlim(-4, 4)
axL.set_ylim(0, 0.46)
axL.set_xlabel("$Z = S / \\sqrt{\\mathrm{Var}(S)}$")
axL.set_ylabel("밀도")
axL.set_title("$40! \\approx 8 \\times 10^{47}$ 을 세지 않고 얻은 분포",
              color=INK, fontsize=12, pad=9)
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------- 오른쪽: 실제 크기
axR.axhline(ALPHA, color=RED, lw=1.2, ls="--", zorder=1, label="겨눈 수준 $0.05$")
axR.plot(NS, sizes, "-", color=BLUE, lw=1.6, zorder=3,
         label="$Z$ 를 그대로 썼을 때의 실제 크기")
axR.plot(NS, sizes, "o", color=BLUE, ms=3.2, zorder=4)
axR.axvline(10, color=GREEN, lw=1.2, ls=":", zorder=2)
axR.text(10.6, 0.0795, "쪽이 적어 둔 문턱 $n = 10$", fontsize=9, color=GREEN)
axR.axvspan(4, 10, color=MUTED, alpha=0.14, zorder=0)
axR.annotate("$r_s$ 는 이 언저리에서\n셈이 끊긴다 ($n! $ 을 다 세야 한다)",
             xy=(7.2, 0.0135), xytext=(13.5, 0.0120), fontsize=9, color=INK,
             va="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axR.set_xlim(4, 60)
axR.set_ylim(0, 0.088)
axR.set_xlabel("$n$")
axR.set_ylabel("귀무가설에서의 실제 기각률")
axR.set_title("$n = 4$ 부터 $60$ 까지 모두 정확히 잰 값", color=INK,
              fontsize=12, pad=9)
axR.legend(loc="lower right", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "kendall_exact_recursion.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
