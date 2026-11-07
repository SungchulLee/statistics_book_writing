"""8.2절 μ의 신뢰구간 — 산포를 아는가가 폭을 정한다.

들머리의 주장은 **sigma 를 아는가**다. 마코위악 외(1992)의 초록에서 평균
36.8 과 정상 상한 37.7 을 읽고, 상한을 "평균 + 1.96 x 산포" 로 되짚어
산포 0.46 을 얻은 뒤(이 되짚기는 이 책의 가정이다), 그 하나를 아는 값으로
칠 때와 추정해야 할 값으로 칠 때를 두 표본크기에서 맞댄다.

오른쪽은 "넓어진다" 의 더 날카로운 꼴이다. sigma 를 알면 구간의 폭은 자료를
모으기 전에 이미 정해진 수인데, 모르면 **폭 자체가 표본마다 달라진다.**
처음에 여기에 "몇 명을 재야 교과서를 반박하나" 를 그렸는데 그것은 표본크기
이야기여서 들머리의 주장이 아니었다. 중첩도 보아야 한다 --- 임계값의 비는
1 절의 t_vs_z_cost 가, 한 표본에서 넓어진 몫을 둘로 쪼개는 일은
width_decomposition 이 이미 그린다. 폭의 **분포**는 어느 쪽도 그리지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/one_sample_intervals/img/"

XBAR = 36.8        # 초록의 평균
SIG = 0.46         # 상한 37.7 을 1.96 배로 되짚은 산포
BOOK = 37.0        # 백이십 년 동안 교과서가 적어 온 수
Z = stats.norm.ppf(0.975)

ROWS = []
for n in (148, 5):
    se = SIG / np.sqrt(n)
    for known, crit in (("σ 를 모를 때 (t 구간)", stats.t.ppf(0.975, n - 1)),
                        ("σ 를 알 때 (z 구간)", Z)):
        h = crit * se
        ROWS.append((n, known, crit, XBAR - h, XBAR + h, 2 * h))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ── (좌) 네 구간과 37.0 ────────────────────────────────────────────
ys = [3, 2.3, 1.0, 0.3]
for y, (n, lab, crit, lo, hi, w) in zip(ys, ROWS):
    col = "tab:blue" if n == 148 else "tab:orange"
    ax1.plot([lo, hi], [y, y], "-", lw=5.0, color=col, solid_capstyle="butt")
    for x in (lo, hi):
        ax1.plot([x, x], [y - 0.09, y + 0.09], "-", lw=3.0, color=col)
    ax1.plot([XBAR], [y], "o", ms=7, color="white", mec=col, mew=2.0)
    ax1.text((lo + hi) / 2, y + 0.13, f"({lo:.4f}, {hi:.4f})", ha="center",
             va="bottom", fontsize=10, color=col)
    ax1.text(hi + 0.02, y, f"폭 {w:.4f}", ha="left", va="center",
             fontsize=10, color="0.35")
ax1.text(36.15, 3.60, "건강한 성인 148 명", fontsize=11.5, fontweight="bold",
         color="tab:blue")
ax1.text(36.15, 1.58, "같은 산포로 5 명만 재었다면", fontsize=11.5,
         fontweight="bold", color="tab:orange")
for y, (n, lab, *_rest) in zip(ys, ROWS):
    ax1.text(36.15, y - 0.24, lab, fontsize=9.5, color="0.4")
ax1.axvline(BOOK, color="tab:red", lw=2.2)
ax1.text(BOOK + 0.02, 3.90, f"교과서의 {BOOK:.1f} °C", fontsize=11.5,
         color="tab:red", fontweight="bold")
ax1.axvline(XBAR, color="0.45", ls="--", lw=1.4)
ax1.text(XBAR, -0.42, f"표본평균 {XBAR}", ha="center", va="top", fontsize=10,
         color="0.35")
ax1.annotate("", xy=(BOOK, 2.72), xytext=(ROWS[0][4], 2.72),
             arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.5))
ax1.text(BOOK + 0.03, 2.72, "37.0 이 밖에 있다", ha="left", va="center",
         fontsize=10.5, color="tab:red", fontweight="bold")
ax1.text(BOOK + 0.03, 0.66, "둘 다 37.0 을 담는다", ha="left",
         fontsize=10.5, color="tab:orange", fontweight="bold")
ax1.set_xlim(36.12, 37.74)
ax1.set_ylim(-0.75, 4.15)
ax1.set_yticks([])
ax1.set_xlabel("체온 (°C)", fontsize=11)
ax1.set_title("같은 평균·같은 산포 — 아는 값이냐 아니냐", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 모르면 폭 자체가 흔들린다 ────────────────────────────────
# 아는 쪽의 반폭을 1 로 놓아 두 표본크기를 한 축에 올린다
rng = np.random.default_rng(802)
REPS = 200_000
stat = {}
for n, col in ((5, "tab:orange"), (148, "tab:blue")):
    zh = Z * SIG / np.sqrt(n)                      # 늘 같은 수
    sd = rng.normal(XBAR, SIG, (REPS, n)).std(axis=1, ddof=1)
    ratio = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) / zh
    lo_, hi_ = np.percentile(ratio, [2.5, 97.5])
    stat[n] = (zh, ratio, float(np.median(ratio)), float(lo_), float(hi_),
               float((ratio < 1).mean()))
    ax2.hist(ratio, bins=np.linspace(0, 3, 260), density=True, color=col,
             alpha=0.62 if n == 5 else 0.78,
             label=f"n = {n} · σ 를 모를 때")
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.32)      # 글자 자리를 위에 낸다
ax2.axvline(1.0, color="0.15", lw=2.6)
ax2.annotate("σ 를 알 때 — 늘 이 자리다", xy=(1.02, 1.1),
             xytext=(1.42, ax2.get_ylim()[1] * 0.30), ha="left",
             va="center", fontsize=11, color="0.15", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.15", lw=1.6))
ax2.axvspan(0, 1, color="0.5", alpha=0.10)
ax2.annotate(f"다섯 명이면 {stat[5][3]:.2f} 배에서 {stat[5][4]:.2f} 배 사이를\n"
             f"오가고, 넷에 하나꼴로 아는 쪽보다 **좁다**".replace("**", ""),
             xy=(1.95, 0.30), xytext=(1.26, ax2.get_ylim()[1] * 0.52),
             fontsize=10.8, color="tab:orange", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.7))
ax2.annotate(f"백마흔여덟이면 {stat[148][3]:.2f} 에서 {stat[148][4]:.2f} 로\n"
             f"거의 정해진 수와 같다",
             xy=(0.90, 3.4), xytext=(0.05, ax2.get_ylim()[1] * 0.78),
             fontsize=10.8, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.7))
ax2.text(0.06, ax2.get_ylim()[1] * 0.30,
         f"이 영역은 아는 쪽보다 **좁은** 구간이다\n"
         f"다섯 명에서 {stat[5][5]:.0%} · "
         f"백마흔여덟에서 {stat[148][5]:.0%}".replace("**", ""),
         ha="left", va="top", fontsize=10.5, color="0.35")
ax2.set_ylim(0, ax2.get_ylim()[1])
ax2.set_xlim(0, 2.8)
ax2.set_xlabel("구간의 반폭 ÷ σ 를 알 때의 반폭", fontsize=11)
ax2.set_ylabel("밀도", fontsize=11)
ax2.legend(fontsize=10.5, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title("σ 를 알면 폭이 정해져 있고, 모르면 폭마저 흔들린다",
              fontsize=12.5)

fig.suptitle("산포를 모른다는 사실이 폭으로 돌아온다 — 표본이 작을수록 비싸다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "body_temp_intervals.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n평균 {XBAR}, 되짚은 산포 {SIG}, 교과서 {BOOK}")
for n, lab, crit, lo, hi, w in ROWS:
    print(f"  n={n:3d}  {lab:<22s} 임계값 {crit:.4f}  ({lo:.4f}, {hi:.4f})  "
          f"폭 {w:.4f}  37.0 을 {'담는다' if lo <= BOOK <= hi else '담지 않는다'}")
w148 = [r[5] for r in ROWS if r[0] == 148]
w5 = [r[5] for r in ROWS if r[0] == 5]
print(f"  폭의 차: n=148 에서 {100*(w148[0]/w148[1]-1):.2f}% · "
      f"n=5 에서 {100*(w5[0]/w5[1]-1):.1f}%")
print("  σ 를 알 때의 반폭(늘 같다)과, 모를 때 그 몇 배가 되는지")
for n in (5, 148):
    zh, _r, med, lo_, hi_, under = stat[n]
    print(f"    n={n:3d}: 아는 쪽 반폭 {zh:.4f} · 모르는 쪽은 가운데값 "
          f"{med:.3f} 배 · 95% 범위 [{lo_:.3f}, {hi_:.3f}] 배 · "
          f"아는 쪽보다 좁게 나오는 비율 {under:.3f}")
