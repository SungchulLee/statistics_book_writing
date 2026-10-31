"""2.2절 ECDF 와 Q-Q 그림 — 「직선이면 기준 분포」의 눈금.

Q-Q 그림이 곧으냐는 눈으로 재는데, **바르게 맞는 모형도 작은 표본에서는
구불거린다.** 그 흔들림의 크기를 띠로 그려 보이고, 띠를 모르면 어떤 판정이
되는지를 센다.

표준화한 순서통계량 z_(i) = (x_(i) - xbar)/s 의 분포는 귀무가설(정규) 아래에서
모수에 의존하지 않는다. 그래서 정규에서 뽑은 모의로 띠를 정확히 얻을 수 있다.
띠는 D = max_i |z_(i) - m_i| / s_i 의 95 분위 c 로 m_i +- c s_i 를 쓴다 ---
동시 포함률이 정확히 0.95 가 된다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch02/exploration/img/"

R = 40_000
NS = [10, 20, 50, 100, 200, 500]
N0 = 20          # 그림 왼쪽·가운데가 쓰는 표본 크기
DF_T = 5         # 참 분포로 쓸 t 의 자유도


def zorder(x):
    """표본을 제 평균과 표준편차로 표준화한 뒤 정렬한다."""
    x = np.sort(x, axis=-1)
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, ddof=1, keepdims=True)


def band(n, rng):
    """정규 아래에서 동시 포함률 0.95 인 띠."""
    z = zorder(rng.standard_normal((R, n)))
    m, s = z.mean(0), z.std(0, ddof=1)
    d = np.max(np.abs(z - m) / s, axis=1)
    c = np.quantile(d, 0.95)
    return m, s, c


def inside_rate(z, m, s, c):
    return np.mean(np.max(np.abs(z - m) / s, axis=1) <= c)


rng = np.random.default_rng(2026)
m0, s0, c0 = band(N0, rng)
lo0, hi0 = m0 - c0 * s0, m0 + c0 * s0

pp = (np.arange(1, N0 + 1) - 0.375) / (N0 + 0.25)     # 블롬의 자리잡기
q0 = stats.norm.ppf(pp)

# 비율은 크게 재고(모의 오차를 줄인다), 그리는 것은 그 가운데 120 개만 쓴다
z_norm = zorder(rng.standard_normal((R, N0)))
z_t = zorder(rng.standard_t(DF_T, size=(R, N0)))

rate_t_in = inside_rate(z_t, m0, s0, c0)              # t(5) 인데 띠 안에 머문다
rate_n_in = inside_rate(z_norm, m0, s0, c0)           # 정규가 띠 안에 머문다 (~0.95)

# 보일 표본 둘: 띠를 「간신히」 벗어난 정규 표본과, 띠 한가운데 얌전한 t(5) 표본
d_t = np.max(np.abs(z_t - m0) / s0, axis=1)
d_n = np.max(np.abs(z_norm - m0) / s0, axis=1)
i_t = int(np.argmin(d_t))
i_n = int(np.argmin(np.abs(d_n - np.quantile(d_n, 0.992))))

# n 을 키우면 띠가 무엇을 걸러내는가
rows = []
for n in NS:
    mm, ss, cc = band(n, rng)
    zt = zorder(rng.standard_t(DF_T, size=(20_000, n)))
    rows.append((n, inside_rate(zt, mm, ss, cc), cc))


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
ax1, ax2, ax3 = axes

for ax, Z, lab, col in [(ax1, z_norm[:120], "참 정규", "tab:blue"),
                        (ax2, z_t[:120], f"참 t({DF_T})", "tab:red")]:
    ax.fill_between(q0, lo0, hi0, color="0.86", zorder=0,
                    label="정규 아래 95% 동시 띠")
    for z in Z:
        ax.plot(q0, z, "-", color=col, lw=0.6, alpha=0.18, zorder=1)
    ax.plot(q0, q0, "--", color="0.35", lw=1.3, zorder=2)
    ax.set_xlabel("정규 분위수")
    ax.set_ylabel("표준화한 자료 분위수")
    ax.grid(alpha=0.2)
    ax.set_ylim(-3.6, 3.6)

ax1.plot(q0, z_norm[i_n], "-o", color="tab:blue", lw=2.0, ms=3.6, zorder=3)
ax1.set_title(f"참으로 정규인 자료 · n = {N0}\n굵은 선은 띠를 벗어난 표본이다",
              fontsize=11.5, color="tab:blue")
ax1.text(0.035, 0.965, f"띠 안에 머문 비율 {rate_n_in * 100:.1f}%", fontsize=10.5,
         color="tab:blue", fontweight="bold", va="top",
         transform=ax1.transAxes)

ax2.plot(q0, z_t[i_t], "-o", color="tab:red", lw=2.0, ms=3.6, zorder=3)
ax2.set_title(f"참으로 t({DF_T}) 인 자료 · n = {N0}\n"
              "굵은 선은 띠 안에 얌전히 들어온 표본이다",
              fontsize=11.5, color="tab:red")
ax2.text(0.035, 0.965, f"띠 안에 머문 비율 {rate_t_in * 100:.1f}%", fontsize=10.5,
         color="tab:red", fontweight="bold", va="top",
         transform=ax2.transAxes)
ax1.legend(fontsize=9.5, loc="lower right", framealpha=0.95)

ns = [r[0] for r in rows]
rt = [r[1] * 100 for r in rows]
ax3.plot(ns, rt, "-o", color="tab:red", lw=2.2, ms=7)
for n, r in zip(ns, rt):
    ha = "left" if n == ns[0] else ("right" if n == ns[-1] else "center")
    dx = {"left": -2, "right": 2, "center": 0}[ha]
    ax3.annotate(f"{r:.1f}%", (n, r), textcoords="offset points",
                 xytext=(dx, 11), ha=ha, fontsize=10,
                 fontweight="bold", color="tab:red")
ax3.axhline(5, color="0.4", ls="--", lw=1.3)
ax3.text(11, 8.5, "띠가 다 걸러낸 자리", fontsize=9.5, color="0.4")
ax3.set_xscale("log")
ax3.set_xlim(8.2, 620)
ax3.set_xticks(ns, [str(n) for n in ns])
ax3.set_ylim(-4, 104)
ax3.set_xlabel("표본 크기 n")
ax3.set_ylabel("정규로 보이는 비율 (%)")
ax3.set_title(f"t({DF_T}) 자료가 정규의 띠 안에 머무는 비율", fontsize=11.5)
ax3.grid(alpha=0.25)

fig.suptitle("「직선이면 기준 분포」 — 얼마나 곧아야 곧은지는 n 이 정한다",
             fontsize=13.5, y=1.015)
plt.tight_layout()
path = OUT + "qq_how_straight.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n띠: 정규 아래 동시 포함률 0.95, 모의 {R:,} 회")
print(f"n = {N0} 에서 c = {c0:.4f}")
print(f"  참 정규 표본이 띠 안: {rate_n_in:.4f}")
print(f"  참 t({DF_T}) 표본이 띠 안: {rate_t_in:.4f}")
print(f"\n{'n':>5} {'t(5) 가 띠 안에 머무는 비율':>10}   c")
for n, r, c in rows:
    print(f"{n:>5} {r:>14.4f}   {c:.4f}")
