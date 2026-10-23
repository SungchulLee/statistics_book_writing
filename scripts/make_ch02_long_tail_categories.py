r"""2.5절 막대그림 쪽의 「AI로 보는 통계학」 그림 — 범주가 수만 개면 막대를 그릴 수 없다.

만드는 파일:
  docs/ch02/visualization/img/long_tail_categories.png

실행:  python3 scripts/make_ch02_long_tail_categories.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다. 로그 눈금은 평문으로 적는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#2E7D32"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(11)
K, N, ALPHA = 20_000, 200_000, 1.07          # 범주 수, 관측 수, 지프 지수
w = 1.0 / np.arange(1, K + 1) ** ALPHA
w /= w.sum()
draw = rng.choice(K, size=N, p=w)
cnt = np.bincount(draw, minlength=K)
srt = np.sort(cnt)[::-1]
seen = int((cnt > 0).sum())
print(f"범주 {K:,} 개 가운데 한 번이라도 나온 것 {seen:,} 개")

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


# ── ① 막대그림을 그려 보면 ──────────────────────────────────────────────
ax = axes[0]
top = 30
ax.bar(np.arange(1, top + 1), srt[:top], color=BLUE_F, edgecolor=BLUE, lw=0.8)
rest = srt[top:].sum()
ax.bar([top + 3], [rest], color=ORANGE_F, edgecolor=ORANGE, lw=1.2)
ax.text(top + 3, rest * 1.04, f"나머지 {seen - top:,} 개를\n모두 더한 막대",
        ha="center", fontsize=9, color=ORANGE)
ax.set_xlabel("도수가 큰 차례로 센 범주", fontsize=9.5, color=INK)
ax.set_ylabel("관측 수", fontsize=9.5, color=INK)
ax.set_title("① 막대 서른 개와 나머지 전부", fontsize=10.5, color=INK)
ax.set_ylim(0, max(srt[0], rest) * 1.25)
tidy(ax)

# ── ② 몇 개가 자료를 덮는가 ─────────────────────────────────────────────
ax = axes[1]
cum = np.cumsum(srt) / N
ks = np.arange(1, seen + 1)
ax.plot(ks, cum[:seen] * 100, color=BLUE, lw=2.0)
for k in (10, 100, 1000):
    ax.plot([k], [cum[k - 1] * 100], "o", ms=6, color=BLUE)
    ax.annotate(f"{k:,} 개 — {cum[k-1]*100:.1f} 퍼센트", xy=(k, cum[k - 1] * 100),
                xytext=(8, -4), textcoords="offset points", fontsize=8.8,
                color=BLUE)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("상위 몇 개를 남기는가 (로그 눈금)", fontsize=9.5, color=INK)
ax.set_ylabel("그 범주들이 덮는 자료의 비율 (퍼센트)", fontsize=9.5, color=INK)
ax.set_title("② 꼬리는 길고 얇다", fontsize=10.5, color=INK)
ax.set_ylim(0, 105)
tidy(ax)

# ── ③ 임계빈도로 자르면 ─────────────────────────────────────────────────
ax = axes[2]
thr = np.arange(1, 51)
kept = np.array([(cnt >= m).sum() for m in thr])
lost = np.array([cnt[cnt < m].sum() / N * 100 for m in thr])
ax.plot(thr, kept, color=BLUE, lw=2.0, label="남는 범주 수 (왼쪽)")
ax.set_xlabel("이 횟수보다 적게 나오면 「기타」로 묶는다", fontsize=9.5, color=INK)
ax.set_ylabel("남는 범주 수", fontsize=9.5, color=INK)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
tidy(ax)
ax2 = ax.twinx()
ax2.plot(thr, lost, color=ORANGE, lw=2.0, ls="--")
ax2.set_ylabel("「기타」로 묶이는 자료의 비율 (퍼센트)", fontsize=9.5, color=ORANGE)
ax2.tick_params(labelsize=9, colors=ORANGE, length=3)
ax2.spines["top"].set_visible(False)
for s in ax2.spines.values():
    s.set_color(MUTED)
ax2.set_ylim(0, max(lost) * 1.25)
for m in (5, 20):
    ax.plot([m], [kept[m - 1]], "o", ms=6, color=BLUE)
    ax2.plot([m], [lost[m - 1]], "o", ms=6, color=ORANGE)
    ax.annotate(f"{m} 번 미만을 묶으면\n범주 {kept[m-1]:,} 개", xy=(m, kept[m - 1]),
                xytext=(6, 10), textcoords="offset points", fontsize=8.8,
                color=BLUE)
ax.set_title("③ 꼬리를 묶으면 범주는 급히 준다", fontsize=10.5, color=INK)

fig.tight_layout()
path = OUT + "long_tail_categories.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
for k in (10, 100, 1000):
    print(f"  상위 {k:5,d} 개가 덮는 비율 {cum[k-1]*100:6.2f} 퍼센트")
for m in (5, 20):
    print(f"  {m:2d} 번 미만을 묶으면 남는 범주 {kept[m-1]:6,d} 개, "
          f"「기타」가 자료의 {lost[m-1]:.2f} 퍼센트")
once = int((cnt == 1).sum())
print(f"  딱 한 번 나온 범주 {once:,} 개 — 본 범주의 {100*once/seen:.1f} 퍼센트, "
      f"자료의 {100*once/N:.2f} 퍼센트")
