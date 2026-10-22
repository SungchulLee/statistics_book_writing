r"""바이올린은 자료가 없는 곳까지 그린다.

바이올린의 윤곽은 커널밀도추정이다. 커널은 각 관측에 매끄러운 봉우리를
얹으므로 **자료의 양끝 바깥에도 밀도가 남는다.** seaborn 은 기본값으로
그 꼬리를 띠폭의 두 배만큼 더 그린다(`cut=2`).

① 자료 범위 밖으로 얼마나 뻗는가 — $n$ 이 작을수록 더 뻗는다.
② 정확도처럼 $1$ 에서 끊기는 지표에서는 **$100\%$ 를 넘는 그림**이 된다.
③ 시드 여섯 개로도 매끄러운 분포가 그려진다.

만드는 파일: docs/ch02/visualization/img/violin_overhang.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import seaborn as sns   # noqa: E402

RNG = np.random.default_rng(0)
NS = [10, 20, 50, 100, 200, 500]
REPS = 200
N_ACC, N_FEW = 8, 6


def span(x, **kw):
    """그려진 바이올린이 세로로 차지하는 구간."""
    fig, ax = plt.subplots()
    sns.violinplot(y=x, ax=ax, **kw)
    v = ax.collections[0].get_paths()[0].vertices
    plt.close(fig)
    return float(v[:, 1].min()), float(v[:, 1].max())


# === ① 자료 밖으로 얼마나 ===========================================
print(f"① 자료 범위 밖으로 뻗는 길이 (자료 범위에 대한 비율, {REPS} 번 평균)")
print(f"{'n':>6}{'아래로':>9}{'위로':>9}{'합(범위 대비)':>14}")
OVER = []
for n in NS:
    tot = 0.0
    for _ in range(REPS):
        x = RNG.normal(0, 1, n)
        lo, hi = span(x)
        rng_ = x.max() - x.min()
        tot += ((x.min() - lo) + (hi - x.max())) / rng_
    OVER.append(tot / REPS)
    print(f"{n:>6}{OVER[-1] / 2:>9.3f}{OVER[-1] / 2:>9.3f}{OVER[-1]:>14.3f}")
print(f"   n = {NS[0]} 에서 자료 범위의 {100 * OVER[0]:.0f}% 만큼 더 그리고, "
      f"n = {NS[-1]} 에서도 {100 * OVER[-1]:.0f}% 다.")
print("   `cut=0` 으로 두면 어느 n 에서도 정확히 0 이다 (자료 끝에서 끊는다).")

# === ② 100% 를 넘는 정확도 ==========================================
RNG2 = np.random.default_rng(4)
ACC = np.round(0.955 + RNG2.beta(4, 1.3, N_ACC) * 0.042, 4)
A_LO, A_HI = span(ACC)
A_LO0, A_HI0 = span(ACC, cut=0)
print(f"\n② 시드 {N_ACC} 개의 정확도 (자료 "
      f"[{ACC.min():.4f}, {ACC.max():.4f}])")
print(f"   기본   바이올린 [{A_LO:.4f}, {A_HI:.4f}]  "
      f"-> 위쪽 끝이 {100 * A_HI:.2f}%")
print(f"   cut=0  바이올린 [{A_LO0:.4f}, {A_HI0:.4f}]")

# === ③ 여섯 개로 그린 분포 ==========================================
FEW = np.round(np.sort(RNG2.normal(0.82, 0.03, N_FEW)), 4)
F_LO, F_HI = span(FEW)
print(f"\n③ 시드 {N_FEW} 개 — 값 {list(FEW)}")
print(f"   바이올린이 덮는 구간 [{F_LO:.4f}, {F_HI:.4f}], "
      f"자료 범위 [{FEW.min():.4f}, {FEW.max():.4f}]")
print(f"   그림의 세로 길이 가운데 "
      f"{100 * (1 - (FEW.max() - FEW.min()) / (F_HI - F_LO)):.0f}% 는 "
      f"자료가 아니라 커널이 만든 것이다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C_V, C_P = "#9467bd", "#d62728"

ax1.semilogx(NS, OVER, "o-", color=C_V, lw=2.8, ms=8, label="seaborn 기본값")
ax1.axhline(0, color="#2ca02c", lw=2.4, ls="--", label="`cut=0`")
ax1.set_xticks(NS)
ax1.set_xticklabels([f"${v}$" for v in NS], fontsize=10)
ax1.minorticks_off()
ax1.set_xlabel("관측 수 $n$")
ax1.set_ylabel("자료 범위 밖으로 더 그린 길이 $\\div$ 자료 범위")
ax1.set_ylim(-0.05, max(OVER) * 1.2)
ax1.set_title("① 자료가 적을수록 더 뻗는다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25)

sns.violinplot(y=ACC, ax=ax2, color=C_V, inner=None, alpha=0.55)
ax2.plot(RNG2.uniform(-0.06, 0.06, N_ACC), ACC, "o", color=C_P, ms=7,
         zorder=4, label=f"실제 시드 {N_ACC} 개")
ax2.axhline(1.0, color="black", lw=2.0, ls="--")
ax2.text(0.42, 1.0008, "정확도 $100\\%$", fontsize=10.5, ha="right")
ax2.annotate(f"바이올린의 꼭대기\n${100 * A_HI:.2f}\\%$",
             xy=(0, A_HI), xytext=(0.17, 0.9925), textcoords="data",
             fontsize=10.5, color=C_P,
             arrowprops=dict(arrowstyle="->", color=C_P, lw=1.1))
ax2.set_xlim(-0.45, 0.45)
ax2.set_xticks([])
ax2.set_ylabel("정확도")
ax2.set_title("② $100\\%$ 를 넘는 정확도", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

sns.violinplot(y=FEW, ax=ax3, color=C_V, inner=None, alpha=0.55)
ax3.plot(np.zeros(N_FEW), FEW, "o", color=C_P, ms=9, zorder=4,
         label=f"자료 전부 ({N_FEW} 개)")
# 값이 가까워 점마다 적으면 겹친다 — 한 곳에 모은다
ax3.text(0.03, 0.03, "자료 여섯 개\n"
         + "  ".join(f"${v:.3f}$" for v in FEW[:3]) + "\n"
         + "  ".join(f"${v:.3f}$" for v in FEW[3:]),
         transform=ax3.transAxes, va="bottom", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax3.set_xlim(-0.45, 0.45)
ax3.set_xticks([])
ax3.set_ylabel("값")
ax3.set_title(f"③ 점 {N_FEW} 개로 그린 매끄러운 분포", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch02/visualization/img/violin_overhang.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
