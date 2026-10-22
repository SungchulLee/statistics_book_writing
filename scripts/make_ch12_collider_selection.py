r"""고를 수 없는 충돌자 — **표본에 들어온 것 자체**가 조건부일 때.

노출 $S$ 와 결과 $Y$ 가 있고, 둘 다 **표본에 들어갈 확률** $T$ 를 높인다고 하자.

$$\operatorname{logit}\Pr(T=1 \mid S, Y) = t_0 + b_S S + b_Y Y$$

$T$ 는 충돌자다. 그런데 이 충돌자는 회귀식에 「넣지 않기로」 고를 수가 없다 —
$T = 1$ 인 사람만 자료에 있기 때문이다.

① 참 오즈비가 $1$ 이어도 표본 안에서는 $1$ 보다 작게 보인다.
② 참으로 **해로운** 노출도 보호적으로 뒤집힌다.
③ 편향이 가장 큰 곳은 양 끝이 아니라 **가운데**다.

만드는 파일: docs/ch12/causation/img/collider_selection.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

PS, Y0, T0 = 0.25, -2.5, -3.0        # 노출 비율, 결과의 기저 로그오즈, 선택의 절편
OR_HARM = 1.5                         # ② 에서 쓰는 참 오즈비 (해롭다)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def observed(bs, by, log_or=0.0, t0=T0):
    """선택된 표본 안에서의 오즈비와 선택률 — 정확히 계산한다."""
    tab = np.zeros((2, 2))
    for s in (0, 1):
        ps = PS if s else 1 - PS
        py = sig(Y0 + log_or * s)
        for y in (0, 1):
            tab[s, y] = ps * (py if y else 1 - py) * sig(t0 + bs * s + by * y)
    return (tab[1, 1] * tab[0, 0]) / (tab[1, 0] * tab[0, 1]), tab.sum()


BSS = np.linspace(0, 2.5, 60)
BYS = (1.0, 2.0, 3.0)
print(f"노출 비율 {PS}, 결과 기저 {sig(Y0):.3f}, 선택 절편 {T0}")
print(f"\n① 참 오즈비 = 1 일 때 표본 안의 오즈비")
print(f"{'b_S':>6}" + "".join(f"{f'b_Y={b}':>11}" for b in BYS) + f"{'선택률':>9}")
curves = {b: np.array([observed(x, b)[0] for x in BSS]) for b in BYS}
for x in (0.5, 1.0, 1.5, 2.0):
    row = f"{x:>6.1f}"
    for b in BYS:
        row += f"{observed(x, b)[0]:>11.3f}"
    print(row + f"{observed(x, BYS[1])[1]:>9.4f}")

# === ② 부호가 뒤집히는 영역 ==========================================
GX = np.linspace(0, 2.5, 140)
GY = np.linspace(0, 3.5, 140)
XX, YY = np.meshgrid(GX, GY)
ZZ = np.vectorize(lambda a, b: observed(a, b, np.log(OR_HARM))[0])(XX, YY)
print(f"\n② 참 오즈비 = {OR_HARM} (해롭다) 일 때 표본 안의 오즈비")
for bs, by in [(0.5, 1.0), (1.0, 2.0), (1.5, 2.0), (2.0, 3.0)]:
    print(f"   b_S={bs}, b_Y={by} : {observed(bs, by, np.log(OR_HARM))[0]:.3f}")
flip = (ZZ < 1.0).mean()
print(f"   격자에서 부호가 뒤집힌(관측 OR < 1) 칸의 비율 {100 * flip:.0f}%")

# === ③ 선택률에 따라 =================================================
T0S = np.linspace(-7.0, 5.0, 120)
BS0, BY0 = 1.5, 2.0
rates, ors = [], []
for t in T0S:
    o, r = observed(BS0, BY0, 0.0, t)
    rates.append(r); ors.append(o)
rates, ors = np.array(rates), np.array(ors)
i_worst = int(np.argmin(ors))
print(f"\n③ 선택률을 바꿔 가며 (b_S={BS0}, b_Y={BY0}, 참 오즈비 1)")
for t in (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 2.0, 4.0):
    o, r = observed(BS0, BY0, 0.0, t)
    print(f"   선택률 {100 * r:>5.1f}%  ->  관측 오즈비 {o:.3f}")
print(f"   가장 나쁜 자리: 선택률 {100 * rates[i_worst]:.1f}%, "
      f"오즈비 {ors[i_worst]:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = ["#1f77b4", "#2ca02c", "#d62728"]

for b, c in zip(BYS, COL):
    ax1.plot(BSS, curves[b], color=c, lw=2.8, label=f"$b_Y = {b}$")
ax1.axhline(1.0, color="black", lw=1.4, ls=":")
ax1.text(0.05, 1.015, "참 오즈비 $= 1$", fontsize=10.5)
i15 = int(np.argmin(np.abs(BSS - 1.5)))
ax1.annotate(f"$b_S=1.5,\\ b_Y=2$ 에서\n${curves[2.0][i15]:.3f}$",
             xy=(1.5, curves[2.0][i15]), xytext=(0.28, 0.46), fontsize=10.5,
             color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
ax1.set_xlabel("노출이 표본에 들어갈 확률을 높이는 정도 $b_S$")
ax1.set_ylabel("표본 안에서 관측되는 오즈비")
ax1.set_ylim(0.2, 1.1)
ax1.set_title("① 없던 보호효과가 생긴다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25)

lev = [0.3, 0.5, 0.7, 1.0, 1.25, 1.5]
cf = ax2.contourf(XX, YY, ZZ, levels=lev, cmap="RdYlBu", extend="both")
cs = ax2.contour(XX, YY, ZZ, levels=[1.0], colors="black", linewidths=2.4)
ax2.clabel(cs, fmt={1.0: "관측 $= 1$"}, fontsize=11)
fig.colorbar(cf, ax=ax2, label="표본 안에서 관측되는 오즈비")
ax2.plot(0, 0, "*", color="black", ms=18)
ax2.text(0.06, 0.12, f"선택이 없으면\n참값 ${OR_HARM}$", fontsize=10)
ax2.set_xlabel("$b_S$")
ax2.set_ylabel("$b_Y$")
ax2.set_title(f"② 참 오즈비가 ${OR_HARM}$ (해롭다) 인데", fontsize=12)

ax3.plot(100 * rates, ors, color="#1f77b4", lw=2.8)
ax3.axhline(1.0, color="black", lw=1.4, ls=":")
ax3.text(62, 1.012, "참 오즈비 $= 1$", fontsize=10.5)
ax3.plot(100 * rates[i_worst], ors[i_worst], "o", color="#d62728", ms=12)
ax3.annotate(f"가장 나쁜 자리\n선택률 ${100 * rates[i_worst]:.0f}\\%$, "
             f"오즈비 ${ors[i_worst]:.3f}$",
             xy=(100 * rates[i_worst], ors[i_worst]), xytext=(40, 0.72),
             fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.set_xlabel("표본에 들어간 사람의 비율 ($\\%$)")
ax3.set_ylabel("표본 안에서 관측되는 오즈비")
ax3.set_xlim(0, 100)
ax3.set_ylim(0.45, 1.06)
ax3.set_title("③ 양 끝에서는 사라진다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/causation/img/collider_selection.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
