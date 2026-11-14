"""11.1절 피어슨 상관계수 — 단위가 사라진다는 것이 이 수를 쓰는 까닭이다.

쪽의 보기 1 자료를 그대로 쓴다. y 를 킬로그램에서 파운드로 바꾸고 x 를
100 만큼 옮기면 공분산은 18.405556 에서 40.577256 으로 뛰는데 r 은 소수
열여섯째 자리까지 그대로다.

쪽의 pearson_pitfalls 는 비선형과 이상치에서 r 이 무너지는 자리를 이미
그린다. 여기서는 그 앞의 물음 --- 왜 공분산 대신 r 을 쓰는가 --- 만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

OUT = "docs/ch11/correlation/img/"
x = np.arange(1, 11, dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2])
LB, SHIFT = 2.20462, 37.0
x2, y2 = x + 100, LB * y + SHIFT


def cov(a, b):
    return float(((a - a.mean()) * (b - b.mean())).sum() / (len(a) - 1))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.25]})
for ax, (a, b, xl, yl, name) in zip(axes[:2], [
        (x, y, "x", "y (kg)", "처음 쟀을 때"),
        (x2, y2, "x + 100", "y (lb)", "저울과 영점만 바꾸면")]):
    ax.plot(a, b, "o", ms=9, color="tab:blue")
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(name, fontsize=12.5)
    ax.grid(alpha=0.25)
    ax.text(0.04, 0.95, f"공분산 {cov(a, b):.6f}\n$r$ = {corr(a, b):.6f}",
            transform=ax.transAxes, va="top", fontsize=12, fontweight="bold",
            color="0.15")
axes[0].text(0.5, -0.30, "점의 배치는 두 그림에서 똑같다",
             transform=axes[0].transAxes, ha="center", fontsize=11.5,
             color="0.35")
axes[1].text(0.5, -0.30, "공분산만 두 배 넘게 뛴다",
             transform=axes[1].transAxes, ha="center", fontsize=11.5,
             color="tab:red", fontweight="bold")

g = np.linspace(0.3, 3.2, 300)
covs = [cov(x, t * y) for t in g]
rs = [corr(x, t * y) for t in g]
ax = axes[2]
ax.plot(g, covs, color="tab:red", lw=2.4, label="공분산")
ax.plot(g, rs, color="tab:blue", lw=2.6, label="$r$")
ax.axvline(LB, color="0.4", ls=":", lw=1.6)
ax.plot([1.0, LB], [cov(x, y), cov(x, y2)], "o", ms=9, color="tab:red")
ax.plot([1.0, LB], [corr(x, y), corr(x, y2)], "o", ms=9, color="tab:blue")
ax.annotate(f"kg → lb\n({LB} 배)", xy=(LB, cov(x, y2)),
            xytext=(LB + 0.18, 48), fontsize=11, color="0.25",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1.4))
ax.text(0.36, 4.5, "$r$ 은 수평선이다", fontsize=11.5, color="tab:blue",
        fontweight="bold")
ax.set_xlim(0.3, 3.2)
ax.set_ylim(-2, 62)
ax.set_xlabel("$y$ 의 척도를 몇 배로 바꾸었나", fontsize=11)
ax.set_title("척도를 바꾸면 하나만 따라 움직인다", fontsize=12.5)
ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25)

fig.suptitle("자료는 한 점도 바뀌지 않았다 — 달라진 것은 저울의 눈금뿐이다",
             fontsize=14, y=1.0)
fig.subplots_adjust(bottom=0.22)
path = OUT + "unit_free.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  처음:  공분산 {cov(x, y):.6f}  r {corr(x, y):.16f}")
print(f"  바꾼 뒤: 공분산 {cov(x2, y2):.6f}  r {corr(x2, y2):.16f}")
print(f"  공분산의 비 {cov(x2, y2) / cov(x, y):.6f}  (= {LB})")
