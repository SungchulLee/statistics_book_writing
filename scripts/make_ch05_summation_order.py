r"""표본을 다시 뽑지 않았는데도 평균이 달라진다 — 더하는 차례.

이 쪽은 통계량이 흔들리는 까닭을 **누가 표본에 뽑혔는가**로 둔다. 기계에서
계산할 때는 까닭이 하나 더 있다. 부동소수의 덧셈은 결합법칙이 성립하지 않으므로,
**같은 수들을 다른 차례로 더하면 다른 값**이 나온다.

① 같은 자료를 차례만 바꿔 백 번 더해 본다.
② 표본의 흔들림은 $1/\sqrt n$ 로 줄고 차례의 흔들림은 커진다 — 어디선가 만난다.
③ 바닥을 정하는 것은 담는 자료형이 아니라 누적하는 자료형이다.

만드는 파일: docs/ch05/foundations/img/summation_order.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)


def to_bf16(x):
    u = np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)
    r = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def seq_mean(x, dt):
    """순차 누적. np.cumsum 이 곧 순차 더하기다 (아래에서 확인한다)."""
    return float(np.cumsum(x, dtype=dt)[-1]) / len(x)


def pw_mean(x, dt):
    """numpy 의 np.sum — 쌍별(이진 트리) 더하기다."""
    return float(np.sum(x, dtype=dt)) / len(x)


# 순차 누적이 맞는지 한 번 확인한다
_x = RNG.lognormal(0, 1, 500).astype(np.float32)
_s = np.float32(0)
for _v in _x:
    _s = _s + _v
print(f"순차 루프 == np.cumsum[-1] : {float(_s) == seq_mean(_x, np.float32) * 500}")

# === ① 차례만 바꾼다 =================================================
N1, R1 = 1_000_000, 200
X64 = RNG.lognormal(0, 1, N1)
X32 = X64.astype(np.float32)
EXACT = float(np.sum(X64.astype(np.longdouble))) / N1
SEQ, PW = [], []
for _ in range(R1):
    p = RNG.permutation(N1)
    SEQ.append(seq_mean(X32[p], np.float32))
    PW.append(pw_mean(X32[p], np.float32))
SEQ, PW = np.array(SEQ), np.array(PW)
print(f"\n① n = {N1:,} 짜리 자료 하나를 차례만 바꿔 {R1} 번 더했다 (float32)")
print(f"   참 평균          {EXACT:.9f}")
print(f"   순차 더하기  — 서로 다른 값 {len(set(SEQ)):>3} 개, "
       f"폭 {SEQ.ptp():.3e} (상대 {SEQ.ptp() / EXACT:.2e})")
print(f"   쌍별 더하기  — 서로 다른 값 {len(set(PW)):>3} 개, "
      f"폭 {PW.ptp():.3e} (상대 {PW.ptp() / EXACT:.2e})")
print(f"   같은 자료다. 다시 뽑은 적이 없다.")

# === ② 두 흔들림의 경주 ==============================================
NS = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 3 * 10 ** 7]
R2 = 12
print(f"\n② 표본의 흔들림과 차례의 흔들림 (float32, 상대값)")
print(f"{'n':>12}{'표본 SE/평균':>15}{'순차 폭/평균':>15}{'쌍별 폭/평균':>15}")
SAMP, OSEQ, OPW = [], [], []
for n in NS:
    x64 = RNG.lognormal(0, 1, n)
    x32 = x64.astype(np.float32)
    ex = float(np.sum(x64.astype(np.longdouble))) / n
    cv = x64.std() / x64.mean()
    s = np.array([seq_mean(x32[RNG.permutation(n)], np.float32)
                  for _ in range(R2)])
    w = np.array([pw_mean(x32[RNG.permutation(n)], np.float32)
                  for _ in range(R2)])
    SAMP.append(cv / np.sqrt(n))
    OSEQ.append(s.ptp() / ex)
    OPW.append(w.ptp() / ex)
    print(f"{n:>12,}{SAMP[-1]:>15.2e}{OSEQ[-1]:>15.2e}{OPW[-1]:>15.2e}")
print(f"   표본 쪽은 1/sqrt(n) 로 줄고 차례 쪽은 큰다. "
      f"n = {NS[-1]:,} 에서 둘이 같은 자릿수다.")

# === ③ 무엇으로 담고 무엇으로 더하는가 ================================
N3, R3 = 1_000_000, 12
x64 = RNG.lognormal(0, 1, N3)
ex3 = float(np.sum(x64.astype(np.longdouble))) / N3
cv3 = x64.std() / x64.mean()
print(f"\n③ n = {N3:,} 을 순차로 더한다 — 담는 자료형과 누적 자료형을 바꾸며")
print(f"{'담기':>10}{'누적':>10}{'폭/평균':>13}{'치우침/평균':>14}")
CASES = []
for nm, xx, dt in (("float64", x64, np.float64),
                   ("float32", x64.astype(np.float32), np.float32),
                   ("float32", x64.astype(np.float32), np.float64)):
    vals = np.array([seq_mean(xx[RNG.permutation(N3)], dt) for _ in range(R3)])
    CASES.append((nm, np.dtype(dt).name, float(vals.ptp() / ex3),
                  float(abs(vals.mean() - ex3) / ex3)))
    print(f"{CASES[-1][0]:>10}{CASES[-1][1]:>10}{CASES[-1][2]:>13.2e}"
          f"{CASES[-1][3]:>14.2e}")
print(f"   같은 n 에서 표본 SE/평균 = {cv3 / np.sqrt(N3):.2e} 다.")
print("   담는 자료형이 아니라 **누적하는 자료형**이 바닥을 정한다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_P, C_E = "#d62728", "#1f77b4", "#2ca02c"

off = (SEQ - EXACT) / EXACT * 1e6
offp = (PW - EXACT) / EXACT * 1e6
bins = np.linspace(min(off.min(), offp.min()) - 0.05,
                   max(off.max(), offp.max()) + 0.05, 45)
ax1.hist(off, bins=bins, color=C_S, alpha=0.85, label="순차 더하기")
ax1.hist(offp, bins=bins, color=C_P, alpha=0.85, label="쌍별 더하기 ($\\mathtt{np.sum}$)")
ax1.axvline(0, color="#333333", lw=1.6, ls="--")
ax1.text(0.04, 0.95, "참값", transform=ax1.transAxes, fontsize=10.5,
         color="#333333", va="top")
ax1.set_xlabel("참 평균에서 벗어난 정도 (백만분율)")
ax1.set_ylabel("$200$ 번 가운데")
ax1.set_title(f"① 차례만 바꿨다 ($n = 10^6$, float32)", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

ax2.loglog(NS, SAMP, "o-", color=C_E, lw=2.6, ms=8, label="표본의 흔들림 (SE/평균)")
ax2.loglog(NS, OSEQ, "s-", color=C_S, lw=2.6, ms=8, label="차례의 흔들림 — 순차")
ax2.loglog(NS, OPW, "^-", color=C_P, lw=2.6, ms=8, label="차례의 흔들림 — 쌍별")
ax2.set_xlabel("자료의 수 $n$")
ax2.set_ylabel("평균에 대한 상대적 흔들림")
ax2.set_xticks([1e4, 1e6, 1e8])
ax2.set_xticklabels(["$10^{4}$", "$10^{6}$", "$10^{8}$"])
ax2.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
ax2.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$"])
ax2.minorticks_off()
ax2.annotate("여기서 만난다", xy=(NS[-1], SAMP[-1]), xytext=(2.5e6, 2.5e-3),
             fontsize=10.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
ax2.set_title("② 하나는 줄고 하나는 큰다", fontsize=12)
ax2.legend(loc="lower left", fontsize=9.5)
ax2.grid(alpha=0.25)

nms = [f"{c[0]}\n+ {c[1]} 누적" for c in CASES]
vals = [max(c[2], 1e-16) for c in CASES]
ax3.bar(np.arange(3), vals, 0.56, color=["#2ca02c", "#d62728", "#1f77b4"])
for i, (v, c) in enumerate(zip(vals, CASES)):
    if c[2] == 0.0:                      # 열두 번이 모두 똑같은 값을 냈다
        ax3.text(i, 2e-16, f"폭 $0$\n({R3} 번 모두 같은 값)", ha="center",
                 fontsize=10, va="bottom")
    else:
        e = int(np.floor(np.log10(v)))
        ax3.text(i, v * 2.0, f"${v / 10 ** e:.1f}\\times 10^{{{e}}}$",
                 ha="center", fontsize=10.5)
ax3.axhline(cv3 / np.sqrt(N3), color=C_E, lw=1.8, ls="--")
ax3.text(2.45, cv3 / np.sqrt(N3) * 1.35, "표본의 흔들림", color=C_E,
         fontsize=10.5, ha="right")
ax3.set_yscale("log")
ax3.set_xticks(range(3))
ax3.set_xticklabels(nms, fontsize=9.5)
ax3.set_ylabel("차례가 만드는 상대적 흔들림")
ax3.set_yticks([1e-14, 1e-10, 1e-6, 1e-2])
ax3.set_yticklabels(["$10^{-14}$", "$10^{-10}$", "$10^{-6}$", "$10^{-2}$"])
ax3.minorticks_off()
ax3.set_ylim(1e-16, 1e-1)
ax3.set_title("③ 바닥을 정하는 것은 누적 자료형이다", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/foundations/img/summation_order.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
