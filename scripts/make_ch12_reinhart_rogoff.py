r"""한 해가 $1/7$ 을 차지하면 — 라인하트·로고프의 $-0.1\%$ 가 만들어진 경로.

모든 수는 헌던·애시·폴린($2013$)이 보고한 **나라별 값**에서 직접 다시 계산한 것이다.

① 일곱 나라의 자료 수와 평균 성장률. 나라별 균등가중이면 모두 $1/7$ 이다.
② $-0.1$ 에서 $+2.2$ 까지의 폭포 — 무엇을 고치면 얼마가 움직이는가.

만드는 파일: docs/ch12/diagnostics/img/reinhart_rogoff.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# === 라인하트·로고프가 실제로 쓴 일곱 나라 ==========================
# (이름, 90% 초과 구간의 해 수, 그 해들의 평균 실질성장률 %)
RR7 = [
    ("그리스", 19, 2.9),
    ("아일랜드", 7, 2.4),
    ("이탈리아", 10, 1.0),
    ("일본", 11, 0.7),
    ("뉴질랜드", 1, -7.6),
    ("영국", 19, 2.4),
    ("미국", 4, -2.0),
]
NZ_TRANSCRIBED = -7.9  # 옮겨 적는 과정에서 -7.6 이 -7.9 가 되었다
NZ_CORRECT = (5, 2.6)  # 빠진 네 해를 되살리면 5 해 평균 +2.6
HAP_FINAL = 2.2  # 엑셀 오류로 빠진 세 나라까지 되살린 발표값

names = [c[0] for c in RR7]
yrs = np.array([c[1] for c in RR7], dtype=float)
avg = np.array([c[2] for c in RR7])


def eq_mean(a):
    return float(np.mean(a))


def cy_mean(a, w):
    return float(np.sum(np.array(w) * np.array(a)) / np.sum(w))


# ① RR 이 발표한 값 — 전사 오류가 들어간 채로, 나라별 균등가중
a0 = avg.copy()
a0[4] = NZ_TRANSCRIBED
step0 = eq_mean(a0)
# ② 전사 오류만 고침
step1 = eq_mean(avg)
# ③ 뉴질랜드의 빠진 네 해를 되살림 (여전히 나라별 균등가중)
a2 = avg.copy()
a2[4] = NZ_CORRECT[1]
step2 = eq_mean(a2)
# ④ 나라-해별 가중으로 바꿈
y2 = yrs.copy()
y2[4] = NZ_CORRECT[0]
step3 = cy_mean(a2, y2)

STEPS = [
    ("라인하트·로고프\n발표값", step0),
    ("전사 오류\n($-7.9 \\to -7.6$)", step1),
    ("뉴질랜드의\n빠진 네 해", step2),
    ("나라별 균등 →\n나라-해별 가중", step3),
    ("엑셀 오류로 빠진\n세 나라까지", HAP_FINAL),
]

# === 그림 ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

col = ["#d62728" if n == "뉴질랜드" else "#1f77b4" for n in names]
ax1.scatter(yrs, avg, s=190, color=col, zorder=5, edgecolors="white", linewidths=1.2)
for n, x, y in zip(names, yrs, avg):
    dy = -1.35 if n in ("영국", "미국", "이탈리아") else 0.95
    ax1.text(x, y + dy, n, ha="center", fontsize=10.5)
ax1.axhline(0, color="#555555", lw=1.0, ls="--")
ax1.axhline(step0, color="#d62728", lw=1.6, ls=":")
ax1.text(
    19.6,
    step0 + 0.5,
    f"일곱 나라의 단순평균 ${step0:.2f}$",
    ha="right",
    fontsize=10,
    color="#d62728",
)
ax1.set_xlabel("$90\\%$ 초과 구간에 머문 해의 수")
ax1.set_ylabel("그 해들의 평균 실질성장률 ($\\%$)")
ax1.set_title("① 일곱 나라에 똑같이 $1/7$ 씩", fontsize=12)
ax1.set_xlim(-1.5, 21.5)
ax1.set_ylim(-9.6, 5.2)
ax1.grid(alpha=0.25)
ax1.annotate(
    "한 해짜리가 열아홉 해짜리와\n같은 무게를 갖는다",
    xy=(1, -7.6),
    xytext=(9.5, -8.4),
    ha="center",
    fontsize=10,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
)

vals = [s[1] for s in STEPS]
xs = np.arange(len(STEPS))
colors = ["#d62728", "#e377c2", "#ff7f0e", "#2ca02c", "#1f77b4"]
ax2.bar(xs, vals, color=colors, width=0.62, zorder=3)
for x, v in zip(xs, vals):
    ax2.text(
        x,
        v + (0.13 if v >= 0 else -0.28),
        f"${v:+.2f}$",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
    )
for x in range(len(STEPS) - 1):
    ax2.plot([x + 0.31, x + 0.69], [vals[x], vals[x + 1]], color="#999999", lw=1.0, ls=":")
ax2.axhline(0, color="black", lw=1.0)
ax2.axhline(3.2, color="#555555", lw=1.3, ls="--")
ax2.text(
    4.42, 3.32, "$60$–$90\\%$ 구간의 값 $3.2$", ha="right", fontsize=9.5, color="#555555"
)
ax2.set_xticks(xs)
ax2.set_xticklabels([s[0] for s in STEPS], fontsize=9.5)
ax2.set_ylabel("$90\\%$ 초과 구간의 평균 성장률 ($\\%$)")
ax2.set_title("② 무엇을 고치면 얼마가 움직이는가", fontsize=12)
ax2.set_ylim(-0.75, 3.9)
ax2.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch12/diagnostics/img/reinhart_rogoff.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① 라인하트·로고프가 실제로 쓴 일곱 나라")
print(f"{'나라':<8}{'해':>5}{'평균 성장률':>12}{'RR 가중치':>11}{'나라-해 가중치':>14}")
tot = yrs.sum()
for n, y, a in zip(names, yrs, avg):
    print(f"{n:<8}{int(y):>5}{a:>11.1f}%{1 / 7 * 100:>10.1f}%{y / tot * 100:>13.1f}%")
print(f"{'합계':<8}{int(tot):>5}{'':>12}{'100.0%':>11}{'100.0%':>14}")
print(f"  뉴질랜드의 한 해가 받는 가중치: 1/7 = {100 / 7:.1f}%  (나라-해로는 {1 / tot * 100:.1f}%)")
print(f"  → {100 / 7 / (100 / tot):.1f} 배")

print("\n② 폭포")
prev = None
for lab, v in STEPS:
    d = "" if prev is None else f"   ({v - prev:+.2f})"
    print(f"  {lab.replace(chr(10), ' '):<28}{v:+7.2f}%{d}")
    prev = v
print(f"\n  발표값 {step0:+.2f} 는 (2.9+2.4+1.0+0.7-7.9+2.4-2.0)/7 = {sum([2.9,2.4,1.0,0.7,-7.9,2.4,-2.0]) / 7:+.4f}")
print(f"  뉴질랜드를 빼고 여섯 나라만 평균내면: {eq_mean(np.delete(a0, 4)):+.3f}")
print(f"  90% 초과와 60-90% 의 참 간격: 3.2 - 2.2 = 1.0 (RR 은 3.2 - (-0.1) = 3.3)")
