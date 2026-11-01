"""10.2절 집계 편향 — 한 자료 안에 상관이 둘 있다.

쪽의 2 절이 쓰는 설정 그대로다. 개인 1,800 명을 집단 간 상관 0.90,
집단 내 상관 0.15 가 되도록 만들면 개인 수준 상관이 0.33 이고 30 명씩
묶은 평균의 상관이 0.82 다.

aggregation_inflation 은 그 둘의 산점도와 m 곡선을 그린다. 여기서는
**한 그림 안에서** 집단 안의 기울기와 집단 사이의 기울기를 맞댄다 ---
0.33 도 0.82 도 그 둘이 섞인 값이지 어느 한쪽이 아니다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

OUT = "docs/ch10/ecological_correlation/img/"
G, M = 60, 30
RB, RW = 0.90, 0.15
VB = (0.33 - RW) / (RB - RW)          # 전체 분산을 1 로 두었을 때
VW = 1 - VB
rng = np.random.default_rng(1950)


def pair(n, rho, var):
    """표본 상관이 정확히 rho 가 되도록 맞춘 두 벡터.

    표본이 작으면(여기서는 집단 60 개) 뽑은 대로 두었을 때 상관이 목표에서
    쉽게 벗어난다. 쪽이 적어 둔 0.33 과 0.82 를 그대로 재현하려고 직교화해
    표본 상관을 못 박는다.
    """
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    a = (a - a.mean()) / a.std()
    b = b - b.mean() - (b @ a) / (a @ a) * a          # a 에 직교시킨다
    b = b / b.std()
    c = rho * a + np.sqrt(1 - rho ** 2) * b
    return np.sqrt(var) * a, np.sqrt(var) * c


bx, by = pair(G, RB, VB)
wx, wy = pair(G * M, RW, VW)
gid = np.repeat(np.arange(G), M)
X = np.repeat(bx, M) + wx
Y = np.repeat(by, M) + wy
mx = np.array([X[gid == g].mean() for g in range(G)])
my = np.array([Y[gid == g].mean() for g in range(G)])
r_ind = float(np.corrcoef(X, Y)[0, 1])
r_grp = float(np.corrcoef(mx, my)[0, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                               gridspec_kw={"width_ratios": [1.2, 1]})

show = np.arange(8)                        # 여덟 집단만 색으로 가른다
cm = plt.get_cmap("tab20")
ax1.plot(X[gid >= 8], Y[gid >= 8], ".", ms=2.0, color="0.88")
for j, g in enumerate(show):
    m_ = gid == g
    ax1.plot(X[m_], Y[m_], "o", ms=3.6, color=cm(j % 20), alpha=0.85)
    a, b = np.polyfit(X[m_], Y[m_], 1)
    xs = np.linspace(X[m_].min(), X[m_].max(), 2)
    ax1.plot(xs, a * xs + b, "-", lw=1.8, color=cm(j % 20), alpha=0.95)
ax1.plot(mx, my, "D", ms=4.5, color="k", mec="white", mew=0.7,
         zorder=5)
a, b = np.polyfit(mx, my, 1)
xs = np.linspace(mx.min(), mx.max(), 2)
ax1.plot(xs, a * xs + b, "-", lw=3.4, color="k", zorder=6,
         label=f"집단 평균을 잇는 선 ($r$ = {r_grp:.2f})")
ax1.plot([], [], "-", lw=1.8, color="0.5",
         label=f"집단 안의 선 ($r$ = {RW:.2f})")
ax1.set_xlabel("$x$", fontsize=11)
ax1.set_ylabel("$y$", fontsize=11)
ax1.set_title(f"같은 {G*M:,} 명 — 짧은 선과 긴 선의 기울기가 다르다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax1.grid(alpha=0.2)

names = ["집단 안의 상관", f"개인 {G*M:,} 명을\n한 덩어리로",
         f"{M} 명씩 묶은\n평균끼리", "집단 사이의 상관"]
vals = [RW, r_ind, r_grp, RB]
cols = ["0.65", "tab:blue", "tab:orange", "0.65"]
y = np.arange(len(names))[::-1]
ax2.barh(y, vals, height=0.52, color=cols, alpha=0.88)
for yy, v in zip(y, vals):
    ax2.text(v + 0.018, yy, f"{v:.4f}", va="center", fontsize=12.5,
             fontweight="bold", color="0.2")
ax2.set_yticks(y, names, fontsize=11)
ax2.set_xlim(0, 1.08)
ax2.set_xlabel("상관", fontsize=11)
ax2.set_title("가운데 둘은 바깥 둘이 섞인 값이다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.annotate("", xy=(0.5, 2.62), xytext=(0.5, 0.38),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.6))
ax2.text(0.53, 1.5, "묶는 크기가\n이 사이를 정한다", fontsize=11,
         color="0.3", fontweight="bold", va="center")

fig.suptitle("한 자료 안에 상관이 둘 있고, 우리가 보는 수는 그 섞임이다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "within_between.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  설정: 집단 간 {RB}, 집단 내 {RW}, Var_B = {VB:.4f}")
print(f"  개인 수준 상관 {r_ind:.4f} (목표 0.33)")
print(f"  {M} 명씩 묶은 평균의 상관 {r_grp:.4f} (목표 0.82)")
