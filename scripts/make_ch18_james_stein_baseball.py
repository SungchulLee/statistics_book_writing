r"""에프론과 모리스가 $1975$ 년에 축소추정을 처음으로 실제 자료에 걸어 본 자리.

자료는 논문 표 $1$ 그대로다 — $1970$ 시즌 첫 $45$ 타석을 마친 선수 $18$ 명의 타율과,
그들이 남은 시즌 동안 실제로 친 타율. 아크사인 변환으로 분산을 고르게 만든 뒤
$\bar X$ 쪽으로 수축한다.

① 수축이 $18$ 개의 추정값을 어디로 옮기는가.
② 개인별 손익 — 전체는 좋아지지만 누군가는 손해를 본다.
③ 수축인자를 바꿔 가며 총제곱오차를 그린다. 자료가 고른 값이 신탁에 얼마나 가까운가.

논문의 수(표 $1$·표 $2$, 총제곱오차 $17.56$ 과 $5.01$, 효율 $3.50$, 최대 개인오차
$2.17$ 과 $1.08$)를 모두 다시 계산해 맞춰 본다.

만드는 파일: docs/ch18/ridge/img/james_stein_baseball.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

# === 논문 표 1 ========================================================
NAME = ["Clemente", "F. Robinson", "F. Howard", "Johnstone", "Berry",
        "Spencer", "Kessinger", "L. Alvarado", "Santo", "Swoboda",
        "Unser", "Williams", "Scott", "Petrocelli", "E. Rodriguez",
        "Campaneris", "Munson", "Alvis"]
Y = np.array([.400, .378, .356, .333, .311, .311, .289, .267, .244,
              .244, .222, .222, .222, .222, .222, .200, .178, .156])
P = np.array([.346, .298, .276, .222, .273, .270, .263, .210, .269,
              .230, .264, .256, .303, .264, .226, .285, .316, .200])
N, K = 45, 18

arcs = lambda y: np.sqrt(N) * np.arcsin(2 * y - 1)      # noqa: E731
back = lambda x: (np.sin(x / np.sqrt(N)) + 1) / 2       # noqa: E731

X, TH = arcs(Y), arcs(P)
XB = X.mean()
V = ((X - XB) ** 2).sum()
C_JS = 1 - (K - 3) / V

err = lambda c: (((XB + c * (X - XB)) - TH) ** 2).sum()  # noqa: E731
JS = XB + C_JS * (X - XB)
PHAT = back(JS)

E_MLE, E_JS = err(1.0), err(C_JS)
CS = np.linspace(0.0, 1.0, 4001)
ES = np.array([err(c) for c in CS])
C_ORA = float(CS[ES.argmin()])

print("① 논문 표 1 을 다시 계산한다")
print(f"   (k-3)/V = {(K - 3) / V:.3f} (논문 .791),  "
      f"수축 뒤 남는 몫 = {C_JS:.3f} (논문 .209),  "
      f"tau = {np.sqrt(V / (K - 3) - 1):.3f} (논문 .514)")
print("\n   선수            첫45타석  축소추정  남은시즌   (논문 표2)")
PAPER = [.290, .286, .281, .277, .273, .273, .268, .264, .259,
         .259, .254, .254, .254, .254, .254, .249, .244, .239]
for i in range(K):
    print(f"   {NAME[i]:<14}{Y[i]:>8.3f}{PHAT[i]:>9.3f}{P[i]:>9.3f}"
          f"{PAPER[i]:>11.3f}")

print(f"\n② 총제곱오차 (아크사인 눈금)")
print(f"   표본평균 그대로   {E_MLE:>6.2f}  (논문 17.56)")
print(f"   제임스-스타인     {E_JS:>6.2f}  (논문  5.01)")
print(f"   효율              {E_MLE / E_JS:>6.2f}  (논문  3.50)")
gain = np.abs(JS - TH) < np.abs(X - TH)
print(f"   나아진 선수 {gain.sum()} 명 / {K}  "
      f"(논문: 15 명, 1·10·15 번만 나빠짐 -> 여기서는 "
      f"{list(np.where(~gain)[0] + 1)})")
i_mx = int(np.abs(X - TH).argmax())
print(f"   최대 개인오차  표본평균 {np.abs(X - TH).max():.2f} / "
      f"제임스-스타인 {np.abs(JS - TH).max():.2f}  "
      f"(논문 2.17 / 1.08, 둘 다 {NAME[i_mx]})")

print(f"\n③ 수축인자를 바꿔 가며")
for lab, c in [("표본평균 (c=1)", 1.0), ("제임스-스타인", C_JS),
               ("신탁 최적", C_ORA), ("전부 평균으로 (c=0)", 0.0)]:
    lam = np.inf if c == 0 else 1 / c - 1
    print(f"   {lab:<22} c = {c:.3f}   오차 {err(c):>6.2f}   "
          f"효율 {E_MLE / err(c):>4.2f}   직교설계의 lambda = "
          + ("무한" if np.isinf(lam) else f"{lam:.2f}"))
print(f"   자료가 고른 c 와 신탁의 차이 : 오차 {E_JS:.2f} 대 {ES.min():.2f} "
      f"({100 * (E_JS / ES.min() - 1):.1f}% 손해)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 5.4))
C_OBS, C_SHR, C_TRU = "#1f77b4", "#d62728", "#2ca02c"

# ① 세 칸 사이를 잇는 선
for i in range(K):
    ax1.plot([0, 1, 2], [Y[i], PHAT[i], P[i]], "-", color="#bbbbbb", lw=0.9,
             zorder=1)
ax1.plot(np.zeros(K), Y, "o", color=C_OBS, ms=7, zorder=3)
ax1.plot(np.ones(K), PHAT, "o", color=C_SHR, ms=7, zorder=3)
ax1.plot(2 * np.ones(K), P, "o", color=C_TRU, ms=7, zorder=3)
for i, dx, ha in [(0, -0.8, "right"), (16, -0.8, "right")]:
    ax1.annotate(NAME[i], (0, Y[i]), xytext=(dx, 0),
                 textcoords="offset fontsize", ha=ha, va="center",
                 fontsize=10, color="#333333")
ax1.axhline(back(XB), color="#888888", lw=1.2, ls=":")
ax1.text(2.42, back(XB), f"$\\bar X$ 되돌린 값\n${back(XB):.3f}$",
         fontsize=9.5, color="#555555", va="center")
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(["첫 $45$ 타석", "축소 추정", "남은 시즌\n(참값)"],
                    fontsize=10.5)
ax1.set_xlim(-1.15, 2.95)
ax1.set_ylabel("타율")
ax1.set_title("① 열여덟 개를 가운데로 끌어당긴다", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

# ② 개인별 손익
o = np.argsort(-(np.abs(X - TH) - np.abs(JS - TH)))
yp = np.arange(K)
w = 0.38
ax2.barh(yp - w / 2, np.abs(X - TH)[o], w, color=C_OBS, label="표본평균")
ax2.barh(yp + w / 2, np.abs(JS - TH)[o], w, color=C_SHR,
         label="제임스-스타인")
ax2.set_yticks(yp)
ax2.set_yticklabels([NAME[j] for j in o], fontsize=9)
ax2.invert_yaxis()
ax2.axhline(gain.sum() - 0.5, color="black", lw=1.3, ls="--")
ax2.text(2.22, gain.sum() - 0.72, f"여기까지 {gain.sum()} 명이 이득",
         fontsize=10, ha="right", va="bottom")
ax2.text(2.22, gain.sum() - 0.28, f"아래 {K - gain.sum()} 명은 손해",
         fontsize=10, ha="right", va="top", color="#8c564b")
ax2.set_xlabel("오차의 절댓값 (아크사인 눈금)")   # mathtext 에 \lvert 가 없다
ax2.set_xlim(0, 2.35)
ax2.set_title("② 전체는 좋아지고 누군가는 나빠진다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25, axis="x")

# ③ 수축인자 곡선
ax3.plot(CS, ES, "-", color="#333333", lw=2.4)
pts = [(1.0, E_MLE, C_OBS, "표본평균 그대로", (0.70, 19.2), "center"),
       (0.0, err(0.0), "#8c564b", "전부 평균으로", (0.09, 13.2), "left"),
       (C_JS, E_JS, C_SHR, "자료가 고른 수축", (0.50, 9.0), "left"),
       (C_ORA, ES.min(), "#9467bd", "참값을 아는 신탁", (0.38, 3.0), "left")]
for c, v, col, lab, xy_t, ha in pts:
    ax3.plot([c], [v], "o", color=col, ms=11, zorder=4)
    ax3.annotate(f"{lab}\n$c = {c:.3f}$, 오차 ${v:.2f}$",
                 xy=(c, v), xytext=xy_t, textcoords="data",
                 fontsize=10.5, color=col, ha=ha, va="center",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
ax3.set_xlabel("남겨 두는 몫 $c$  (직교설계의 능형이면 $c = 1/(1+\\lambda)$)")
ax3.set_ylabel("총제곱오차 (아크사인 눈금)")
ax3.set_ylim(1.8, 20.6)
ax3.set_xlim(-0.05, 1.12)
ax3.set_title("③ 자료가 고른 값과 신탁이 거의 같다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/ridge/img/james_stein_baseball.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
