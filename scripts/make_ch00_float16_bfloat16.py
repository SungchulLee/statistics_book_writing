r"""같은 $2$ 바이트를 다르게 나눈다 — float16 과 bfloat16.

이 쪽 $\S 2$ 의 표는 `dtype` 을 「원소의 자료형」이라 적고 `nbytes` 로 닫는다.
자료형을 바꾸면 **자리만** 바뀌는 것이 아니다. $16$ 비트에는 두 가지가 있고,
같은 두 바이트를 **지수와 가수에 다르게 나눈다.**

① 비트를 어떻게 나누는가 — float32 · float16 · bfloat16.
② $1$ 을 계속 더하면 어디서 멈추는가 — $256$ · $2{,}048$ · $16{,}777{,}216$.
③ 크기에 따라 상대오차가 어떻게 달라지는가 — 언더플로와 오버플로의 벽.

numpy 에 bfloat16 이 없으므로 float32 의 가수를 $7$ 비트로 **반올림해** 흉내 낸다.

만드는 파일: docs/ch00/tools/img/float16_bfloat16.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def to_bf16(x):
    """float32 의 가수를 7 비트로 짝수 반올림한다 (bfloat16 흉내)."""
    u = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    r = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


BF_MAX = float(np.array([0x7F7F0000], np.uint32).view(np.float32)[0])
BF_TINY = float(np.finfo(np.float32).tiny)
BF_EPS = 2.0 ** -8

F16, F32 = np.finfo(np.float16), np.finfo(np.float32)
SPEC = [
    ("float32", 1, 8, 23, 4, F32.max, F32.tiny, F32.eps),
    ("float16", 1, 5, 10, 2, float(F16.max), float(F16.tiny), float(F16.eps)),
    ("bfloat16", 1, 8, 7, 2, BF_MAX, BF_TINY, BF_EPS),
]
print("① 같은 자리를 다르게 나눈다")
print(f"{'자료형':>9}{'바이트':>7}{'지수':>6}{'가수':>6}{'가장 큰 수':>13}"
      f"{'가장 작은 정규수':>18}{'eps':>12}")
for nm, s, e, m, by, mx, tn, ep in SPEC:
    print(f"{nm:>9}{by:>7}{e:>6}{m:>6}{mx:>13.4g}{tn:>18.4g}{ep:>12.4g}")
print("   bfloat16 은 지수가 float32 와 같아 **범위가 같다**. 가수만 잘랐다.")

# === ② 1 을 계속 더하면 ==============================================
print("\n② 1 을 계속 더하면 어디서 멈추는가")


def running_native(dt, n):
    s = np.array(0, dt)
    one = np.array(1, dt)
    out = np.empty(n)
    for i in range(n):
        s = (s + one).astype(dt)
        out[i] = s.item()
    return out


def running_bf(n):
    s = np.float32(0)
    out = np.empty(n)
    for i in range(n):
        s = to_bf16(s + np.float32(1))
        out[i] = s.item()
    return out


NSHOW = 6000
CURVE = {
    "float16": running_native(np.float16, NSHOW),
    "bfloat16": running_bf(NSHOW),
    "float32": running_native(np.float32, NSHOW),
}
STALL = {}
for nm, c in CURVE.items():
    flat = np.flatnonzero(np.diff(c) == 0)
    STALL[nm] = float(c[flat[0]]) if len(flat) else float("inf")
STALL["float32"] = 2.0 ** 24                 # 손으로 따져 둔 값 — 아래에서 확인한다
s = np.float32(2 ** 24)
print(f"   float32 에서 2^24 = {2 ** 24:,} 일 때 s + 1 == s 인가 : "
      f"{bool(np.float32(s + np.float32(1)) == s)}")
for nm in ("bfloat16", "float16", "float32"):
    print(f"   {nm:>9} : {STALL[nm]:>12,.0f} 에서 멈춘다")
print("   가수가 짧을수록 일찍 멈춘다 — bfloat16 이 float16 보다 **먼저** 멈춘다.")

# === ③ 크기에 따른 상대오차 ==========================================
print("\n③ 크기를 바꾸며 왕복시켜 상대오차를 잰다")
MAG = np.logspace(-10, 6, 200)
rng = np.random.default_rng(0)
base = MAG[:, None] * (1.0 + 0.3 * rng.standard_normal((len(MAG), 400)))
base = np.abs(base)                       # float64 가 참값이다


def rel_err(x, cast):
    y = cast(np.asarray(x, np.float32))
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.abs(y.astype(np.float64) - x.astype(np.float64)) / x.astype(
            np.float64)
    e = np.where(np.isfinite(y), e, 1.0)      # 오버플로는 전부 잃은 것으로 센다
    return np.median(e, axis=1)


with np.errstate(over="ignore"):      # 65504 를 넘는 값은 inf 가 된다 — 그것이 요점
    F16_RT = base.astype(np.float16).astype(np.float32)
ERR = {
    "float16": rel_err(base, lambda a: F16_RT),
    "bfloat16": rel_err(base, to_bf16),
    "float32": rel_err(base, lambda a: a),       # float64 -> float32 왕복
}
for nm in ("float16", "bfloat16"):
    lo = MAG[np.flatnonzero(ERR[nm] < 0.5)[0]]
    hi = MAG[np.flatnonzero(ERR[nm] < 0.5)[-1]]
    print(f"   {nm:>9} : 상대오차가 50% 아래인 구간이 "
          f"{lo:.2g} ~ {hi:.2g} ({np.log10(hi / lo):.0f} 자릿수)")
    mid = (MAG > 1e-3) & (MAG < 1e2)
    print(f"{'':>13}그 안쪽에서 상대오차 중앙값 {np.median(ERR[nm][mid]):.2e}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_E, C_M = "#555555", "#d62728", "#1f77b4"
COL = {"float32": "#2ca02c", "float16": "#d62728", "bfloat16": "#1f77b4"}

for i, (nm, sg, ex, ma, by, mx, tn, ep) in enumerate(SPEC):
    y = 2 - i
    tot = sg + ex + ma
    x = 0.0
    for w, c, lab in ((sg, C_S, "부호"), (ex, C_E, "지수"), (ma, C_M, "가수")):
        ax1.add_patch(Rectangle((x, y - 0.3), w, 0.6, facecolor=c,
                                edgecolor="white", lw=1.2))
        if w >= 4:
            ax1.text(x + w / 2, y, f"{lab} {w}", ha="center", va="center",
                     color="white", fontsize=10.5)
        x += w
    ax1.text(-0.8, y, nm, ha="right", va="center", fontsize=11.5)
    m10 = int(np.floor(np.log10(ep)))
    ax1.text(tot + 0.8, y, f"${by}$ 바이트\n"
             f"$\\varepsilon = {ep / 10 ** m10:.1f}\\times 10^{{{m10}}}$",
             ha="left", va="center", fontsize=10, color="#333333")
ax1.set_xlim(-11, 44)
ax1.set_ylim(-0.8, 2.8)
ax1.set_yticks([])
ax1.set_xticks([0, 8, 16, 24, 32])
ax1.set_xlabel("비트")
ax1.plot([9, 9], [-0.05, 2.45], ls="--", color=C_E, lw=1.3, alpha=0.8)
ax1.text(0.0, -0.45, "지수 8 비트가 같은 자리에서 끝난다 — 범위가 같다",
         fontsize=10.5, color=C_E, va="center")
ax1.set_title("① 같은 두 바이트를 다르게 나눈다", fontsize=12)
for side in ("left", "right", "top"):
    ax1.spines[side].set_visible(False)

ks = np.arange(1, NSHOW + 1)
for nm in ("float32", "float16", "bfloat16"):
    ax2.loglog(ks, np.maximum(CURVE[nm], 1), color=COL[nm], lw=2.6, label=nm)
ax2.loglog(ks, ks, "--", color="#888888", lw=1.4, label="참값 (더한 횟수)")
for nm, c in (("bfloat16", COL["bfloat16"]), ("float16", COL["float16"])):
    ax2.annotate(f"${STALL[nm]:,.0f}$ 에서 멈춘다".replace(",", "{,}"),
                 xy=(NSHOW * 0.95, STALL[nm]), xytext=(-0.4, 0.5),
                 textcoords="offset fontsize", fontsize=10.5, color=c,
                 ha="right")
ax2.set_xlabel("$1$ 을 더한 횟수")
ax2.set_ylabel("누적합")
ax2.set_xticks([1, 10, 100, 1000])
ax2.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax2.set_yticks([1, 10, 100, 1000])
ax2.set_yticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax2.minorticks_off()
ax2.set_title("② 더하기가 멈추는 자리", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

for nm in ("float32", "bfloat16", "float16"):
    ax3.loglog(MAG, np.maximum(ERR[nm], 1e-9), color=COL[nm], lw=2.6,
               label=nm)
ax3.axvline(float(F16.tiny), color=COL["float16"], lw=1.3, ls=":")
ax3.axvline(float(F16.max), color=COL["float16"], lw=1.3, ls=":")
ax3.text(float(F16.tiny) * 1.4, 2e-9, "여기부터\nfloat16 은 비정규수", fontsize=9.5,
         color=COL["float16"], va="bottom")
ax3.text(float(F16.max) * 0.55, 2e-9, "$65{,}504$", fontsize=9.5,
         color=COL["float16"], va="bottom", ha="right")
ax3.set_xlabel("담으려는 수의 크기")
ax3.set_ylabel("왕복시킨 뒤의 상대오차 (중앙값)")
ax3.set_xticks([1e-10, 1e-6, 1e-2, 1e2, 1e6])
ax3.set_xticklabels(["$10^{-10}$", "$10^{-6}$", "$10^{-2}$", "$10^{2}$",
                     "$10^{6}$"])
ax3.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
ax3.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$",
                     "$1$"])
ax3.minorticks_off()
ax3.set_ylim(1e-9, 4)
ax3.set_title("③ 범위를 사면 정밀도를 판다", fontsize=12)
ax3.legend(loc="center left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/tools/img/float16_bfloat16.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
