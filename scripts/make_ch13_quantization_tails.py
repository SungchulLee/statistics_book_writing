r"""정규성이 싼 곳과 비싼 곳을 가르는 것은 **평균을 보느냐 최댓값을 보느냐**다.

$8$ 비트 양자화(absmax)는 $[-\max|x|,\ \max|x|]$ 를 $255$ 칸으로 나눈다. 칸 너비가
$2\max|x|/255$ 이므로 **오차를 정하는 것이 최댓값 하나**다. 평균에는 중심극한정리가
받쳐 주지만 최댓값에는 받쳐 줄 것이 없다.

① 분산이 같아도 꼬리가 두꺼우면 양자화 신호대잡음비가 무너진다.
② 이상치 **하나**가 나머지 $4{,}095$ 좌표의 유효 비트를 몇 개 깎는가.
③ 이상치 차원을 $16$ 비트로 빼내면 얼마나 돌아오는가 (LLM.int8() 이 하는 일).

만드는 파일: docs/ch13/assumptions/img/quantization_tails.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

D = 4096            # 특징 차원 수
LEV8 = 255          # 8 비트 대칭 양자화의 칸 수
REPS = 400


def quant(x, levels=LEV8, axis=-1):
    """absmax 대칭 양자화 — 벡터마다 최댓값으로 눈금을 정한다."""
    a = np.max(np.abs(x), axis=axis, keepdims=True)
    step = 2 * a / levels
    return np.round(x / step) * step


def snr_db(x, xq, mask=None):
    """신호대잡음비 (dB). mask 를 주면 그 좌표들만 센다."""
    e = x - xq
    if mask is not None:
        x, e = x[..., mask], e[..., mask]
    return float(10 * np.log10((x ** 2).mean() / (e ** 2).mean()))


# === ① 꼬리 모양에 따라 =============================================
def unit(z):
    """분산을 1 로 맞춘다 — 비교의 조건을 같게 한다."""
    return z / z.std(axis=-1, keepdims=True)


shapes = {}
shapes["정규"] = unit(RNG.standard_normal((REPS, D)))
shapes["라플라스"] = unit(RNG.laplace(0, 1, (REPS, D)))
shapes["$t_5$"] = unit(RNG.standard_t(5, (REPS, D)))
shapes["$t_3$"] = unit(RNG.standard_t(3, (REPS, D)))
g = RNG.standard_normal((REPS, D))
g[:, 0] = 20.0 * np.sign(RNG.standard_normal(REPS))   # 20 배짜리 이상치 하나
shapes["정규 $+$\n이상치 $1$ 개"] = unit(g)

print(f"① 같은 분산, 다른 꼬리 — 8 비트 양자화 (차원 {D}, {REPS} 벌)")
print(f"{'모양':>16}{'평균 |x|max':>13}{'SNR (dB)':>11}")
snr1, amax1 = [], []
for name, x in shapes.items():
    xq = quant(x)
    snr1.append(snr_db(x, xq))
    amax1.append(float(np.abs(x).max(axis=-1).mean()))
    print(f"{name.replace(chr(10), ' '):>16}{amax1[-1]:>13.2f}{snr1[-1]:>11.2f}")

# === ② 이상치 하나가 깎는 비트 ======================================
# 먼저 깨끗한 정규 자료에서 칸 수 L 과 오차의 관계를 재 둔다 (되짚을 눈금).
LGRID = np.unique(np.round(np.logspace(np.log2(7), np.log2(255), 60,
                                       base=2)).astype(int))
clean = RNG.standard_normal((REPS, D))
err_of_L = np.array([np.sqrt(((clean - quant(clean, L)) ** 2).mean())
                     for L in LGRID])
print(f"\n② 깨끗한 정규 자료의 눈금 — 칸 수 L 과 RMS 오차")
for L in (7, 15, 31, 63, 127, 255):
    i = int(np.argmin(np.abs(LGRID - L)))
    print(f"   L = {LGRID[i]:>3} ({np.log2(LGRID[i] + 1):.0f} 비트)"
          f"  RMS 오차 {err_of_L[i]:.5f}")

MAGS = np.array([3.5, 5, 7, 10, 14, 20, 28, 40, 60])
print(f"\n   이상치 하나가 들어올 때 나머지 {D - 1} 좌표의 유효 비트")
print(f"{'이상치 크기':>11}{'RMS 오차':>11}{'유효 비트':>11}")
bits = []
rest = np.arange(1, D)
for m in MAGS:
    x = RNG.standard_normal((REPS, D))
    x[:, 0] = m * np.sign(RNG.standard_normal(REPS))
    e = (x - quant(x))[:, rest]
    rms = float(np.sqrt((e ** 2).mean()))
    # 같은 오차를 내는 깨끗한 자료의 칸 수로 되짚는다
    Leq = float(np.interp(-rms, -err_of_L, LGRID))
    bits.append(np.log2(Leq + 1))
    print(f"{m:>11.1f}{rms:>11.5f}{bits[-1]:>11.2f}")
bits = np.array(bits)

# === ③ 이상치 차원을 빼내면 =========================================
NOUT, MAG = 6, 20.0        # 논문이 보고한 자리 — 6 개 차원, 최대 20 배
KS = np.arange(0, 9)
x3 = RNG.standard_normal((REPS, D))
odim = np.arange(NOUT)
x3[:, odim] = MAG * np.sign(RNG.standard_normal((REPS, NOUT)))
print(f"\n③ {D} 차원 가운데 {NOUT} 개가 {MAG:.0f} 배 "
      f"(전체의 {100 * NOUT / D:.3f}%) 일 때")
print(f"{'16 비트로 뺀 차원 수':>19}{'나머지 SNR (dB)':>17}")
snr3 = []
for k in KS:
    keep = np.arange(k, D)                  # k 개를 빼고 나머지를 8 비트로
    sub = x3[:, keep]
    snr3.append(snr_db(sub, quant(sub)))
    print(f"{k:>19}{snr3[-1]:>17.2f}")
snr3 = np.array(snr3)
print(f"   {NOUT} 개를 빼면 {snr3[0]:.1f} dB -> {snr3[NOUT]:.1f} dB "
      f"({snr3[NOUT] - snr3[0]:+.1f} dB)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

names = list(shapes)
cols = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#8c564b"]
ax1.bar(np.arange(len(names)), snr1, 0.62, color=cols)
for i, (v, a) in enumerate(zip(snr1, amax1)):
    ax1.text(i, v + 0.7, f"${v:.1f}$", ha="center", fontsize=11)
    ax1.text(i, 2.0, f"$|x|_{{\\max}}$\n${a:.1f}$", ha="center", fontsize=9.5,
             color="white")
ax1.set_xticks(np.arange(len(names)))
ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylabel("$8$ 비트 양자화의 SNR (dB)")
ax1.set_ylim(0, max(snr1) * 1.16)
ax1.set_title("① 분산이 같아도 꼬리가 정한다", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(MAGS, bits, "o-", color="#d62728", lw=2.7, ms=8)
ax2.axhline(8, color="#1f77b4", lw=2.0, ls="--", label="명목 $8$ 비트")
ax2.annotate(f"$20$ 배짜리 하나에\n${bits[list(MAGS).index(20)]:.1f}$ 비트",
             xy=(20, bits[list(MAGS).index(20)]), xytext=(23, 7.1),
             fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xscale("log")
ax2.set_xticks(MAGS)
ax2.set_xticklabels([f"${v:g}$" for v in MAGS], fontsize=9.5)
ax2.minorticks_off()
ax2.set_xlabel("이상치 하나의 크기 (표준편차 단위)")
ax2.set_ylabel(f"나머지 ${D - 1}$ 좌표의 유효 비트")
ax2.set_ylim(3.5, 8.6)
ax2.set_title("② 하나가 전부를 깎는다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(KS, snr3, "o-", color="#1f77b4", lw=2.7, ms=8)
ax3.axvline(NOUT, color="#d62728", lw=2.0, ls="--",
            label=f"이상치 차원 ${NOUT}$ 개")
ax3.annotate(f"${snr3[NOUT] - snr3[0]:+.1f}$ dB",
             xy=(NOUT, snr3[NOUT]), xytext=(2.6, snr3[NOUT] - 1.5),
             fontsize=11.5, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
ax3.set_xlabel(f"$16$ 비트로 따로 뺀 차원 수 (전체 ${D}$ 중)")
ax3.set_ylabel("나머지 좌표의 SNR (dB)")
ax3.set_title("③ 여섯 개를 빼면 돌아온다", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/assumptions/img/quantization_tails.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
