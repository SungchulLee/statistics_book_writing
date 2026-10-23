r"""2.3절 왜도와 첨도 쪽의 「AI로 보는 통계학」 그림 — 꼬리의 두께가 양자화 눈금을 정한다.

만드는 파일:
  docs/ch02/shape/img/kurtosis_quantization.png

실행:  python3 scripts/make_ch02_kurtosis_quantization.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다. 로그 눈금의 마이너스도 mathtext 라
       한글 글꼴에서 두부가 되므로 눈금을 직접 적는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
MUTED = "#90A4AE"
COLORS = ["#1565C0", "#33691E", "#E65100", "#6A1B9A"]

OUT = "docs/ch02/shape/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
n = 1_000_000


def unit(x):
    return x / x.std()


dists = [
    ("정규", unit(rng.normal(0, 1, n))),
    ("라플라스", unit(rng.laplace(0, 1, n))),
    ("t(5)", unit(rng.standard_t(5, n))),
    ("1 퍼센트가 8 배", unit(np.where(rng.random(n) < 0.01,
                                  rng.normal(0, 8, n), rng.normal(0, 1, n)))),
]


def snr_db(w, clip, bits):
    """대칭 균일 양자화. 눈금을 clip 에 맞추고 그 밖은 자른다."""
    q = 2 ** (bits - 1) - 1
    s = clip / q
    wq = np.clip(np.round(w / s), -q, q) * s
    return 10 * np.log10(np.var(w) / np.mean((w - wq) ** 2))


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

# ── ① 꼬리 ──────────────────────────────────────────────────────────────
ax = axes[0]
grid = np.linspace(0, 25, 400)
for (name, w), c in zip(dists, COLORS):
    tail = (np.abs(w)[:, None] > grid[None, :]).mean(0)
    k = stats.kurtosis(w)
    ax.plot(grid, np.maximum(tail, 1e-7), color=c, lw=1.7,
            label=f"{name} — 초과첨도 {k:.1f}")
    mx = np.abs(w).max()
    ax.plot([mx], [1 / n], "o", ms=5, color=c)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: {1e-6: "1/100만", 1e-4: "1/1만",
                  1e-2: "1/100", 1: "1"}.get(v, "")))
ax.set_ylim(5e-7, 1.6)
ax.set_xlim(0, 25)
ax.set_xlabel("표준편차의 몇 배 (절댓값)", fontsize=9.5, color=INK)
ax.set_ylabel("그보다 큰 값의 비율", fontsize=9.5, color=INK)
ax.set_title("① 같은 표준편차, 다른 꼬리 (점이 최댓값)", fontsize=10.5, color=INK)
ax.legend(fontsize=8.3, frameon=False, loc="upper right")
tidy(ax)

# ── ②③ 자르는 지점에 따른 신호 대 잡음비 ─────────────────────────────────
for ax, bits in ((axes[1], 4), (axes[2], 8)):
    qs = np.linspace(0.90, 1.0, 81)
    for (name, w), c in zip(dists, COLORS):
        cs = np.quantile(np.abs(w), qs)
        ss = np.array([snr_db(w, cc, bits) for cc in cs])
        ax.plot(cs, ss, color=c, lw=1.7)
        i = int(ss.argmax())
        ax.plot([cs[i]], [ss[i]], "o", ms=5, color=c)
        mx = np.abs(w).max()
        ax.plot([mx], [snr_db(w, mx, bits)], "x", ms=6, color=c, mew=1.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("눈금 끝을 어디에 둘 것인가 (표준편차의 몇 배)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("신호 대 잡음비 (dB)", fontsize=9.5, color=INK)
    ax.set_title(f"{'②' if bits == 4 else '③'} {bits} 비트 — 동그라미가 가장 좋은 자리,"
                 f" 가위표가 최댓값", fontsize=10.5, color=INK)
    tidy(ax)
axes[1].set_ylim(0, 21)
axes[2].set_ylim(0, 45)

fig.tight_layout()
path = OUT + "kurtosis_quantization.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)

print("분포            초과첨도  max/sd   4비트최대  4비트최적(자리)   8비트최대")
for name, w in dists:
    k = stats.kurtosis(w)
    mx = np.abs(w).max()
    cs = np.quantile(np.abs(w), np.linspace(0.90, 1.0, 81))
    ss4 = np.array([snr_db(w, c, 4) for c in cs])
    i = int(ss4.argmax())
    print(f"{name:<14}{k:9.2f}{mx:8.2f}{snr_db(w, mx, 4):10.1f}"
          f"{ss4[i]:9.1f} ({cs[i]:.2f})   {snr_db(w, mx, 8):8.1f}")
