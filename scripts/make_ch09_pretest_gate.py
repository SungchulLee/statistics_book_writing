"""9.3절 대마초 가격 — 정규성 검정이라는 문은 거꾸로 달려 있다.

왼쪽: 샤피로-윌크 통과율. 로그정규 자료에서 n 이 커지면 0.297 에서 0 으로
      닫힌다. 원자료가 정규인지 묻기 때문이다.
오른쪽: 그 자리에서 t 검정의 실제 1종 오류율. n 과 상관없이 0.05 근처다.
      표본평균이 정규에 가까운지만 필요하고 중심극한정리가 그것을 준다.

문이 가장 세게 닫히는 자리가 정작 t 가 가장 안전한 자리다.
수는 쪽의 모의 출력 그대로다(되풀이 4,000 번).
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

OUT = "docs/ch09/two_sample_tests/img/"
NOMINAL = 0.05
NS = [10, 30, 100, 1000]

# 자료 → (정규성 통과율, 언제나 t 의 실제 1종 오류율)
PASS = {"정규":     [0.908, 0.903, 0.901, 0.899],
        "로그정규": [0.297, 0.002, 0.000, 0.000]}
TERR = {"정규":     [0.050, 0.052, 0.043, 0.048],
        "로그정규": [0.031, 0.048, 0.043, 0.049]}
STYLE = {"정규": ("tab:blue", "o", "-"), "로그정규": ("tab:red", "s", "-")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.1), sharex=True)

for name, (c, m, ls) in STYLE.items():
    ax1.plot(NS, PASS[name], ls, marker=m, color=c, lw=2, ms=8, label=name)
    ax2.plot(NS, TERR[name], ls, marker=m, color=c, lw=2, ms=8, label=name)

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(NS))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{n:,}" for n in NS]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("표본크기 n (로그 눈금)")
    ax.grid(alpha=0.25)

# ── (좌) 문 ──────────────────────────────────────────────────────────
for n, v in zip(NS, PASS["로그정규"]):
    # 축 아래로 내려가 눈금 꼬리표와 겹치므로 점 위에 적는다
    ax1.annotate(f"{v:.3f}", (n, v), textcoords="offset points",
                 xytext=(0, 11), ha="center", fontsize=10, color="tab:red")
ax1.annotate("n 이 커질수록 문이 닫힌다", (30, 0.002),
             textcoords="offset points", xytext=(30, 52), ha="left",
             fontsize=10.5, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3))
ax1.annotate(f'{PASS["정규"][0]:.3f}', (NS[0], PASS["정규"][0]),
             textcoords="offset points", xytext=(4, 9), ha="left",
             fontsize=10, color="tab:blue")
ax1.annotate(f'{PASS["정규"][-1]:.3f}', (NS[-1], PASS["정규"][-1]),
             textcoords="offset points", xytext=(-4, 9), ha="right",
             fontsize=10, color="tab:blue")
ax1.set_ylim(-0.05, 1.12)
ax1.legend(fontsize=10, framealpha=0.95, loc="center right")
ax1.set_ylabel("샤피로–윌크 통과율")
ax1.set_title("정규성 검정이라는 문 — 원자료를 묻는다", fontsize=11.5)

# ── (우) t ───────────────────────────────────────────────────────────
ax2.axhline(NOMINAL, color="black", ls="--", lw=1.3)
ax2.text(NS[0], NOMINAL + 0.0035, "명목 0.05", ha="left", va="bottom",
         fontsize=10)
ax2.text(30, 0.0115, "문이 가장 세게 닫히는 자리에서도\nt 는 명목을 지킨다", ha="left",
         va="bottom", fontsize=10.5, color="0.3")
ax2.set_ylim(0.0, 0.085)
ax2.legend(fontsize=10, framealpha=0.95, loc="lower right")
ax2.set_ylabel("t 검정의 실제 1종 오류율")
ax2.set_title("그 자리에서 t 는 — 표본평균만 묻는다", fontsize=11.5)

plt.tight_layout()
path = OUT + "pretest_gate.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"{'n':>6} {'정규 통과':>9} {'로그정규 통과':>13} {'정규 t':>8} {'로그정규 t':>11}")
for i, n in enumerate(NS):
    print(f"{n:>6} {PASS['정규'][i]:>9.3f} {PASS['로그정규'][i]:>13.3f} "
          f"{TERR['정규'][i]:>8.3f} {TERR['로그정규'][i]:>11.3f}")
