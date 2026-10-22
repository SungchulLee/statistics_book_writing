r"""16.1 검정력 비교 쪽의 상자에 들어가는 그림을 만든다.

호지스-레만(1956)이 Wilcoxon 순위합의 ARE 를 가장 낮추는 밀도로 찾아낸
뒤집힌 포물선이, 열세 해 뒤 커널밀도추정의 에파네치니코프 커널과 같은 꼴임을
한 장에 나란히 놓는다.

만드는 파일:
  ch16/foundations/img/worst_case_density.png

실행:  python3 scripts/make_ch16_worst_case_density.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import integrate

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

OUT = "docs/ch16/foundations/img/"
os.makedirs(OUT, exist_ok=True)

ROOT5 = np.sqrt(5.0)


def epan(x):
    """분산이 1 인 뒤집힌 포물선 밀도 (|x| <= sqrt 5)."""
    out = np.zeros_like(np.asarray(x, dtype=float))
    m = np.abs(x) <= ROOT5
    out[m] = 3.0 / (4.0 * ROOT5) * (1.0 - np.asarray(x, dtype=float)[m] ** 2 / 5.0)
    return out


def normal(x):
    return np.exp(-np.asarray(x, dtype=float) ** 2 / 2.0) / np.sqrt(2.0 * np.pi)


def laplace(x):
    b = 1.0 / np.sqrt(2.0)          # 분산 2b^2 = 1
    return np.exp(-np.abs(x) / b) / (2.0 * b)


def uniform(x):
    a = np.sqrt(3.0)                # 분산 a^2/3 = 1
    return np.where(np.abs(x) <= a, 1.0 / (2.0 * a), 0.0)


def roughness(f, lo=-40.0, hi=40.0):
    """적분 f^2 을 수치로 구한다."""
    val, _ = integrate.quad(lambda t: f(np.array([t]))[0] ** 2, lo, hi, limit=400)
    return val


def are(f):
    """ARE(Wilcoxon 순위합, t) = 12 * sigma^2 * (적분 f^2)^2, 분산은 1 로 맞춰 둠."""
    return 12.0 * roughness(f) ** 2


# ---------------------------------------------------------------- 수 확인
for name, f in [("포물선", epan), ("정규", normal), ("Laplace", laplace),
                ("균일", uniform)]:
    r = roughness(f)
    print(f"{name:8s}  적분f^2 = {r:.6f}   ARE = {12 * r ** 2:.6f}")
print(f"포물선 닫힌꼴: 적분f^2 = {3 * ROOT5 / 25:.6f}, ARE = {108 / 125:.6f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.3))

# ---------------------------------------------------------------- 왼쪽
x = np.linspace(-3.4, 3.4, 1400)
axL.fill_between(x, 0, epan(x), color=BLUE_F, zorder=1)
axL.plot(x, epan(x), color=BLUE, lw=2.6, zorder=4,
         label=f"포물선   ARE $= 108/125 = {108 / 125:.3f}$")
axL.plot(x, normal(x), color=INK, lw=1.9, zorder=3,
         label=r"정규   ARE $= 3/\pi = 0.955$")
axL.plot(x, laplace(x), color=ORANGE, lw=1.9, ls="--", zorder=3,
         label=r"Laplace   ARE $= 3/2 = 1.500$")
axL.plot(x, uniform(x), color=GREEN, lw=1.5, ls=":", zorder=2,
         label=f"균일   ARE $= {are(uniform):.3f}$")

axL.axvline(-ROOT5, color=MUTED, lw=0.8, ls=":", zorder=0)
axL.axvline(ROOT5, color=MUTED, lw=0.8, ls=":", zorder=0)
axL.annotate(r"$\pm\sqrt{5}\,\sigma$ 에서 끊긴다", xy=(ROOT5, 0.012),
             xytext=(2.45, 0.135), color=BLUE, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
axL.set_xlim(-3.4, 3.4)
axL.set_ylim(0, 0.76)
axL.set_xlabel("분산을 $1$ 로 맞춘 눈금")
axL.set_ylabel("밀도")
axL.set_title("순위합이 가장 불리해지는 밀도", color=INK, fontsize=12, pad=9)
axL.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------------------------- 오른쪽
rng = np.random.default_rng(16)
data = np.sort(np.concatenate([rng.normal(-1.25, 0.40, 5),
                               rng.normal(1.45, 0.55, 4)]))
h = 0.30
grid = np.linspace(-3.2, 3.6, 1200)
total = np.zeros_like(grid)
for i, d in enumerate(data):
    bump = epan((grid - d) / h) / (h * len(data))
    total += bump
    axR.plot(grid, bump, color=BLUE, lw=1.0, alpha=0.55, zorder=2,
             label="낱낱의 포물선" if i == 0 else None)
axR.plot(grid, total, color=INK, lw=2.6, zorder=4, label="다 더한 밀도추정")
axR.plot(data, np.zeros_like(data), "|", color=ORANGE, ms=14, mew=1.8,
         zorder=5, label=f"자료 $n = {len(data)}$")

axR.set_xlim(-3.2, 3.6)
axR.set_ylim(0, total.max() * 1.30)
axR.set_xlabel("$x$")
axR.set_ylabel("밀도추정값")
axR.set_title("같은 포물선, 열세 해 뒤의 일자리", color=INK, fontsize=12, pad=9)
axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "worst_case_density.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
