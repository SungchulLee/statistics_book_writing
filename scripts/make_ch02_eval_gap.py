r"""2.5절 계단 그림 쪽의 「AI로 보는 통계학」 그림 — 평가 사이를 이으면 없던 변화를 만든다.

만드는 파일:
  docs/ch02/visualization/img/eval_gap.png

실행:  python3 scripts/make_ch02_eval_gap.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다.
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
GREEN = "#2E7D32"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(5)
STEPS = 10_000
t = np.arange(1, STEPS + 1)

# 참 학습 곡선: 꾸준히 내려가다 한 번 치솟고 돌아온다
loss = 2.6 * np.exp(-t / 2600) + 0.42 + 0.012 * rng.normal(size=STEPS)
spike = 1.15 * np.exp(-((t - 4300) ** 2) / (2 * 260 ** 2))
loss = loss + spike

GAP = 1000
ev = np.arange(GAP, STEPS + 1, GAP)
ev_loss = loss[ev - 1]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), width_ratios=[1.35, 1])


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


# ── ① 세 가지 그림 ──────────────────────────────────────────────────────
ax = axes[0]
ax.plot(t, loss, color=MUTED, lw=0.9, label="걸음마다 잰 참값")
ax.plot(ev, ev_loss, color=BLUE, lw=2.0, marker="o", ms=5,
        label=f"{GAP:,} 걸음마다 재어 선으로 이은 것")
ax.step(np.r_[0, ev], np.r_[ev_loss[0], ev_loss], where="post", color=ORANGE,
        lw=1.8, ls="--", label="같은 점을 계단으로")
# 선 보간이 참값에서 가장 크게 벗어난 자리를 짚는다.
# 첫 평가점 앞은 보간이 아니라 바깥쪽 연장이므로 빼고 본다.
interp_all = np.interp(t, ev, ev_loss)
inside = t >= ev[0]
i = int(np.flatnonzero(inside)[np.abs(interp_all - loss)[inside].argmax()])
lin = float(interp_all[i])
ax.annotate(f"참값 {loss[i]:.3f}\n({t[i]:,} 걸음)", xy=(t[i], loss[i]),
            xytext=(t[i] + 600, loss[i] + 0.22), fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate(f"선이 말하는 값 {lin:.3f}", xy=(t[i], lin),
            xytext=(t[i] + 900, lin - 0.30), fontsize=9, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax.set_xlabel("걸음", fontsize=9.5, color=INK)
ax.set_ylabel("손실", fontsize=9.5, color=INK)
ax.set_title("① 평가 사이에 일어난 일은 그림에 없다", fontsize=10.5, color=INK)
ax.legend(fontsize=8.8, frameon=False, loc="upper right")
tidy(ax)

# ── ② 평가 간격을 줄이면 ────────────────────────────────────────────────
ax = axes[1]
gaps = np.array([100, 200, 300, 500, 700, 1000, 1500, 2000])
# 평가를 어느 걸음에서 시작하느냐에 따라 봉우리를 놓치기도 잡기도 한다.
# 시작점을 간격 안에서 모두 옮겨 보고 가운뎃값과 최소~최대를 함께 그린다.
med, lo, hi = [], [], []
for g in gaps:
    ws = []
    for off in range(0, g, max(1, g // 25)):
        e = np.arange(g + off, STEPS + 1, g)
        if len(e) < 3:
            continue
        approx = np.interp(t, e, loss[e - 1])
        ins = t >= e[0]
        ws.append(np.abs(approx - loss)[ins].max())
    ws = np.array(ws)
    med.append(np.median(ws)); lo.append(ws.min()); hi.append(ws.max())
med, lo, hi = np.array(med), np.array(lo), np.array(hi)
ax.fill_between(gaps, lo, hi, color="#DCEBFB", lw=0)
ax.plot(gaps, med, color=BLUE, lw=2.0, marker="o", ms=5)
ax.axhline(spike.max(), color=RED, ls=":", lw=1.2)
ax.text(2000, spike.max() * 1.03, "봉우리의 높이 1.15", ha="right", fontsize=9,
        color=RED)
for g, m in zip(gaps, med):
    if g in (300, 1000, 2000):
        ax.annotate(f"{g:,} 걸음 — {m:.2f}", xy=(g, m), xytext=(6, 10),
                    textcoords="offset points", fontsize=8.8, color=BLUE)
ax.text(0.03, 0.92, "띠는 평가를 시작하는 걸음에 따른 최소~최대",
        transform=ax.transAxes, fontsize=8.8, color=MUTED)
ax.set_xlabel("평가 간격 (걸음)", fontsize=9.5, color=INK)
ax.set_ylabel("선 보간이 참값에서 벗어난 최대 폭", fontsize=9.5, color=INK)
ax.set_title("② 간격이 넓을수록, 그리고 운에 따라", fontsize=10.5, color=INK)
ax.set_ylim(0, 1.35)
tidy(ax)

fig.tight_layout()
path = OUT + "eval_gap.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
print(f"가장 크게 어긋나는 자리 {t[i]:,} 걸음: 참값 {loss[i]:.4f}, "
      f"선이 말하는 값 {lin:.4f}, 차이 {loss[i]-lin:.4f}")
print(f"평가점 {len(ev)} 개의 최댓값 {ev_loss.max():.4f} "
      f"({ev[int(ev_loss.argmax())]:,} 걸음)")
for g, m, l, h in zip(gaps, med, lo, hi):
    print(f"  간격 {g:5,d}: 어긋남 가운뎃값 {m:.3f}  (최소 {l:.3f}, 최대 {h:.3f})")
