r"""11.6 「가정 위반의 처리」 쪽의 상자에 들어가는 그림을 만든다.

Box 와 Cox 가 1982 년 반박문에서 든 보기를 그대로 되살린다 — 한 집단은 995 언저리,
다른 집단은 1005 언저리, 집단 안 표준편차는 1. 이 자료에서 lambda 는 거의 결정되지
않고, 집단 차이를 y^(lambda) 눈금에 적으면 자릿수가 통째로 바뀐다. 그러나 야코비안을
넣은 z(lambda) 눈금에서는 어느 lambda 에서나 10 근처로 같다.

만드는 파일:
  ch12/diagnostics/img/boxcox_rebutted.png

실행:  python3 scripts/make_ch11_boxcox_rebutted.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch12/diagnostics/img/"
os.makedirs(OUT, exist_ok=True)

# Box 와 Cox 가 적은 그대로 — 995 언저리와 1005 언저리, 집단 안 표준편차 1
rng = np.random.default_rng(1964)
y1 = 995.0 + rng.normal(0, 1, 10)
y2 = 1005.0 + rng.normal(0, 1, 10)
y = np.concatenate([y1, y2])
ly = np.log(y)
LG = ly.mean()                      # 기하평균의 로그


def diff_raw(lam):
    """y^(lambda) = (y^lam - 1)/lam 눈금에서 잰 두 집단 평균의 차이.
       -1/lam 은 상수라 차이에서 사라지므로 y^lam/lam 으로 계산해도 같다."""
    if abs(lam) < 1e-12:
        return np.log(y2).mean() - np.log(y1).mean()
    # 로그공간에서 — y^lam 은 lam 이 커지면 바로 넘치거나 밑돈다
    a1, a2 = lam * np.log(y1), lam * np.log(y2)
    A = max(a1.max(), a2.max())
    return np.sign(lam) * np.exp(A - np.log(abs(lam))) * (
        np.exp(a2 - A).mean() - np.exp(a1 - A).mean())


def diff_norm(lam):
    """Box 와 Cox 가 반박문에서 권한 z(lambda) = (y^lam - 1)/(lam * gdot^(lam-1)) 눈금."""
    if abs(lam) < 1e-12:
        return (np.log(y2).mean() - np.log(y1).mean()) * np.exp(LG)
    a1 = lam * np.log(y1) - np.log(abs(lam)) - (lam - 1) * LG
    a2 = lam * np.log(y2) - np.log(abs(lam)) - (lam - 1) * LG
    A = max(a1.max(), a2.max())
    return np.sign(lam) * np.exp(A) * (np.exp(a2 - A).mean() - np.exp(a1 - A).mean())


def profile(lam):
    """정규화 눈금 위의 프로파일 로그가능도 (야코비안이 흡수된다). 로그공간에서 안정하게."""
    if abs(lam) < 1e-12:
        t1, t2 = np.log(y1) * np.exp(LG), np.log(y2) * np.exp(LG)
        r = np.concatenate([t1 - t1.mean(), t2 - t2.mean()])
        return -len(y) / 2 * np.log((r ** 2).mean())
    a1 = lam * np.log(y1) - np.log(abs(lam)) - (lam - 1) * LG
    a2 = lam * np.log(y2) - np.log(abs(lam)) - (lam - 1) * LG
    A = max(a1.max(), a2.max())
    e1, e2 = np.exp(a1 - A), np.exp(a2 - A)
    r = np.concatenate([e1 - e1.mean(), e2 - e2.mean()])
    # RSS = e^{2A} * sum(r^2)  ->  log(RSS/n) = 2A + log(mean r^2)
    return -len(y) / 2 * (2 * A + np.log((r ** 2).mean()))


LAMS = np.linspace(-120, 120, 9601)
pl = np.array([profile(l) for l in LAMS])
pmax = pl.max()
lhat = LAMS[pl.argmax()]
inside = LAMS[pl > pmax - 1.92]

print(f"기하평균 = {np.exp(LG):.3f}")
print(f"lambda hat = {lhat:+.2f},  95% 프로파일 구간 = "
      f"[{inside.min():+.2f}, {inside.max():+.2f}]")
print(f"\n{'lambda':>8s} {'y^(lam) 눈금의 차이':>24s} {'z(lam) 눈금의 차이':>20s}")
for lam in (-4, -2, -1, 0, 1, 2, 4):
    print(f"{lam:8.1f} {diff_raw(lam):24.6e} {diff_norm(lam):20.4f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ------------------------------------------------ 왼쪽: lambda 가 정해지지 않는다
axL.plot(LAMS, pl - pmax, color=BLUE, lw=2.2, zorder=3)
axL.axhline(-1.92, color=RED, lw=1.3, ls="--", zorder=2)
axL.fill_between(LAMS, -1.92, pl - pmax, where=(pl - pmax > -1.92),
                 color=BLUE, alpha=0.18, zorder=1)
axL.axvline(1, color=GREEN, lw=1.2, ls=":", zorder=2)
axL.axvline(-1, color=ORANGE, lw=1.2, ls=":", zorder=2)
axL.annotate("$\\lambda = 1$ (변환 않음)", xy=(1, -3.2), xytext=(22, -3.9),
             fontsize=9, color=GREEN,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
axL.annotate("$\\lambda = -1$ (역수)", xy=(-1, -2.3), xytext=(-105, -3.1),
             fontsize=9, color=ORANGE,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
axL.text(-115, -1.80, "가능도가 $1.92$ 밖에 떨어지지 않는 구간", fontsize=9,
         color=RED, va="bottom")
axL.set_xlim(-120, 120)
axL.set_ylim(-5.2, 0.35)
axL.set_xlabel("$\\lambda$")
axL.set_ylabel("프로파일 로그가능도 (최댓값 대비)")
axL.set_title(f"$95\\%$ 구간이 $[{inside.min():.0f},\\ {inside.max():.0f}]$ — "
              "자료가 $\\lambda$ 를 고르지 못한다", color=INK, fontsize=11.5, pad=9)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------ 오른쪽: 두 눈금의 차이
grid = np.linspace(-4, 4, 401)
raw = np.array([abs(diff_raw(l)) for l in grid])
nrm = np.array([abs(diff_norm(l)) for l in grid])
axR.semilogy(grid, raw, color=RED, lw=2.4, zorder=3,
             label="$y^{(\\lambda)}$ 눈금에서 잰 차이")
axR.semilogy(grid, nrm, color=GREEN, lw=2.4, zorder=3,
             label="$z(\\lambda)$ 눈금에서 잰 차이")
axR.axhline(10, color=MUTED, lw=1.0, ls=":", zorder=1)
# 로그 눈금의 지수는 mathtext 라 평문으로 박는다
ticks = [1e-12, 1e-8, 1e-4, 1, 1e4, 1e8, 1e12]
axR.set_yticks(ticks)
axR.set_yticklabels(["0.000000000001", "0.00000001", "0.0001", "1",
                     "10000", "100000000", "1000000000000"], fontsize=8)
axR.minorticks_off()
axR.set_ylim(1e-13, 1e13)
axR.set_xlim(-4, 4)
axR.set_xlabel("$\\lambda$")
axR.set_ylabel("두 집단 평균의 차이 (절댓값)")
axR.set_title("같은 자료, 같은 효과 — 자만 바꾸었다", color=INK,
              fontsize=12, pad=9)
axR.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
axR.text(1.6, 22, f"$z$ 눈금에서는 어느 $\\lambda$ 에서나 ${diff_norm(1):.1f}$",
         fontsize=9, color=GREEN)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "boxcox_rebutted.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
