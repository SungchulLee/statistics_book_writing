r"""1.2 「자료 수집을 설계하라」 쪽의 상자에 들어가는 그림을 만든다.

워너(1965)의 무작위응답에서 설계 확률 p 가 정하는 두 가지 — 치러야 하는
표본 배수와, 「예」 한 번이 알려 주는 것 — 이 서로 반대로 움직임을 보인다.

  표본 배수      Var(pi_hat) = Var(Y)/(2p-1)^2 이므로 직접 묻기 대비 1/(2p-1)^2
  드러나는 정도   P(민감 | 예) — 베이즈로 뒤집은 값. p=1 이면 1(전부 드러남),
                 p -> 1/2 이면 참 비율 pi 로 돌아간다(아무것도 드러나지 않음)

만드는 파일:
  ch01/contrast/img/randomized_response.png

실행:  python3 scripts/make_ch01_randomized_response.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE = "#E65100"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch01/contrast/img/"
os.makedirs(OUT, exist_ok=True)

PI = 0.30          # 참 비율. 드러나는 정도를 재려면 하나 정해야 한다.
MARKS = [0.60, 0.70, 0.80, 0.90]

p = np.linspace(0.502, 0.999, 2000)
mult = 1.0 / (2 * p - 1) ** 2                      # 표본 배수
# P(예) = p*pi + (1-p)(1-pi).  베이즈로 P(민감 | 예) 를 뒤집는다.
p_yes = p * PI + (1 - p) * (1 - PI)
reveal = p * PI / p_yes

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.3))

# ---------------------------------------------- 왼쪽: 치러야 하는 표본 배수
axL.plot(p, mult, color=BLUE, lw=2.6, zorder=4)
axL.fill_between(p, 1, mult, color=BLUE_F, zorder=2)
axL.axhline(1.0, color=MUTED, lw=1.3, ls="--", zorder=3)
axL.text(0.525, 1.18, "직접 묻기와 같은 표본", fontsize=9.5, color=INK)
for q in MARKS:
    m = 1.0 / (2 * q - 1) ** 2
    axL.plot([q], [m], "o", color=ORANGE, ms=6, zorder=5)
    axL.annotate(f"$p={q:.1f}$\n{m:.1f}배", (q, m), textcoords="offset points",
                 xytext=(9, 4), fontsize=9, color=INK)
axL.set_yscale("log")
axL.set_xlim(0.5, 1.0)
axL.set_ylim(0.8, 400)
axL.set_xlabel("설계 확률 $p$")
axL.set_ylabel("같은 정밀도에 필요한 표본 배수")
axL.set_title("값은 표본으로 치른다", color=INK, fontsize=12, pad=9)
axL.set_yticks([1, 2, 5, 10, 50, 100])
axL.set_yticklabels(["1배", "2배", "5배", "10배", "50배", "100배"])
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------- 오른쪽: 「예」 가 알려 주는 것
axR.plot(p, reveal, color=RED, lw=2.6, zorder=4)
axR.axhline(PI, color=MUTED, lw=1.3, ls="--", zorder=3)
axR.text(0.515, PI + 0.025, f"아무것도 알려 주지 않는 선 ($\\pi={PI:.2f}$)",
         fontsize=9.5, color=INK)
for q in MARKS:
    py = q * PI + (1 - q) * (1 - PI)
    r = q * PI / py
    axR.plot([q], [r], "o", color=ORANGE, ms=6, zorder=5)
    axR.annotate(f"$p={q:.1f}$\n{r:.2f}", (q, r), textcoords="offset points",
                 xytext=(-30, 2), fontsize=9, color=INK)
axR.set_xlim(0.5, 1.0)
axR.set_ylim(0, 1.05)
axR.set_xlabel("설계 확률 $p$")
axR.set_ylabel("「예」 가 민감 쪽일 확률")
axR.set_title("비밀은 $p$ 를 $1/2$ 쪽으로 밀어 산다", color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "randomized_response.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"참 비율 pi = {PI}")
print(f"{'p':>6} {'표본 배수':>10} {'P(민감|예)':>12}")
for q in MARKS + [0.95, 0.99]:
    m = 1.0 / (2 * q - 1) ** 2
    py = q * PI + (1 - q) * (1 - PI)
    print(f"{q:6.2f} {m:10.2f} {q * PI / py:12.3f}")
print("saved", path)
