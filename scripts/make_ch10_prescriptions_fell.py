"""11장 개요 — 이 일은 논문 안에서 끝나지 않았다.

들머리의 마지막 beat 를 그린다. 쪽의 출처 줄에 이미 확인해 적어 둔 수
말고는 아무것도 쓰지 않는다. 통계 어휘를 하나도 요구하지 않는 그림이고,
장 개요는 독자가 「보정」도 「무작위배정」도 아직 배우지 않은 자리라
그것이 요건이다.

쓰는 수 (모두 Hersh · Stefanick · Stafford, 「National Use of
Postmenopausal Hormone Therapy」, JAMA 291(2004) 47 쪽):

  1995 년        5,800 만 건
  1999 년        8,960 만 건
  2002 년 6 월까지 유지
  발표 뒤        38% 감소
  2003 년 추정   5,700 만 건
  에스트로겐+프로게스틴을 쓰던 여성  560 만 → 250 만 명

보고된 해가 띄엄띄엄이므로 그 사이를 이은 선은 **보간**이고, 그렇게
보이도록 보고된 자리에만 점을 찍고 사이는 옅게 긋는다. 떨어진 폭은
내가 두 끝에서 다시 계산하지 않고 출처가 적은 38% 를 그대로 적는다.

만드는 파일: docs/ch11/img/prescriptions_fell.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch11/img/"

# 출처가 적은 자리만 (연도, 백만 건, 이름표)
POINTS = [(1995.0, 58.0, "1995 년\n5,800 만 건"),
          (1999.0, 89.6, "1999 년\n8,960 만 건"),
          (2002.5, 89.6, "2002 년 6 월까지\n그대로였다"),
          (2003.5, 57.0, "2003 년 추정\n5,700 만 건")]
ANNOUNCE = 2002.5
USERS = (5.6, 2.5)          # 에스트로겐+프로게스틴을 쓰던 여성 (백만 명)

xs = np.array([p[0] for p in POINTS])
ys = np.array([p[1] for p in POINTS])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# ── (좌) 처방 건수 ─────────────────────────────────────────────────
ax1.axvspan(ANNOUNCE, 2005.4, color="tab:red", alpha=0.07)
ax1.plot(xs, ys, "-", lw=2.2, color="0.72", zorder=1)
ax1.plot(xs[:3], ys[:3], "-", lw=3.4, color="tab:blue", zorder=2)
ax1.plot(xs[2:], ys[2:], "-", lw=3.4, color="tab:red", zorder=2)
ax1.plot(xs, ys, "o", ms=12, color="white", mec="0.15", mew=2.6, zorder=5)

for (x, y, lab), (dx, dy, ha, va) in zip(
        POINTS, [(-0.08, 5, "left", "bottom"), (0.0, 6, "center", "bottom"),
                 (-0.12, 6, "right", "bottom"), (0.0, -6, "center", "top")]):
    ax1.text(x + dx, y + dy, lab, ha=ha, va=va, fontsize=10.5, color="0.2",
             fontweight="bold")

ax1.axvline(ANNOUNCE, color="tab:red", lw=2.2, ls="--")
ax1.text(ANNOUNCE - 0.12, 36, "2002 년 7 월\n시험 결과 발표", ha="right",
         va="bottom", fontsize=11.5, color="tab:red", fontweight="bold")
ax1.text(2003.95, 76, "발표 뒤\n38% 감소", ha="left", va="center",
         fontsize=13, color="tab:red", fontweight="bold")

ax1.set_xlim(1994.3, 2005.4)
ax1.set_ylim(30, 108)
ax1.set_xticks([1995, 1997, 1999, 2001, 2003],
               ["1995", "1997", "1999", "2001", "2003"])
ax1.set_xlabel("해", fontsize=11)
ax1.set_ylabel("한 해 처방 건수 (백만 건)", fontsize=11)
ax1.set_title("미국의 호르몬 대체요법 처방", fontsize=13)
ax1.grid(alpha=0.25)
ax1.text(0.015, 0.035, "출처가 적어 둔 해에만 점을 찍었다 — 그 사이를 이은\n"
         "선은 보간이다", transform=ax1.transAxes, ha="left", va="bottom",
         fontsize=9.5, color="0.4")

# ── (우) 약을 끊은 사람 ────────────────────────────────────────────
bx = np.arange(2)
ax2.bar(bx, USERS, width=0.5, color=["0.6", "tab:red"], alpha=0.9)
for x, v in zip(bx, USERS):
    ax2.text(x, v / 2, f"{v:.1f}", ha="center", va="center", fontsize=18,
             fontweight="bold", color="white")
ax2.annotate("", xy=(1, USERS[1] + 0.25), xytext=(0, USERS[0] + 0.25),
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.6,
                             connectionstyle="arc3,rad=-0.25"))
drop = 100 * (1 - USERS[1] / USERS[0])
ax2.text(0.5, USERS[0] + 1.25, f"{drop:.0f}% 가 약을 끊었다", ha="center",
         va="center", fontsize=13, color="tab:red", fontweight="bold")
ax2.set_xticks(bx, ["발표 전", "발표 뒤"], fontsize=13)
ax2.set_ylim(0, 8.2)
ax2.set_ylabel("에스트로겐+프로게스틴을 쓰던 여성 (백만 명)", fontsize=11)
ax2.set_title("시험이 중단한 바로 그 조합을 쓰던 사람", fontsize=13)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("이 일은 논문 안에서 끝나지 않았다", fontsize=15, y=1.0)
plt.tight_layout()
path = OUT + "prescriptions_fell.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print("\n[장 개요] 출처가 적은 수만 그렸다")
for x, y, lab in POINTS:
    print(f"  {x:7.1f}  {y:5.1f} 백만 건   {lab.replace(chr(10), ' ')}")
print(f"  에스트로겐+프로게스틴 사용 여성 {USERS[0]} → {USERS[1]} 백만 명 "
      f"({drop:.1f}% 감소)")
