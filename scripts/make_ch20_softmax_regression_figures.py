r"""ch20 소프트맥스 회귀 절의 개념 그림을 그린다.

만드는 파일:
  ch20/softmax_regression/img/multinomial_shift_invariance.png
      로짓을 통째로 옮겨도 확률이 한 점도 변하지 않는다 (모수 과잉)
  ch20/softmax_regression/img/numerical_stability_overflow.png
      순진한 소프트맥스가 깨지는 지점과 LSE 의 안전 구간
  ch20/softmax_regression/img/softmax_simplex_temperature.png
      확률단체 위의 온도 경로

실행:  python3 scripts/make_ch20_softmax_regression_figures.py   (저장소 최상위에서)
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
OUT = "docs/ch20/softmax_regression/img/"
os.makedirs(OUT, exist_ok=True)

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

COLORS = [BLUE, ORANGE, GREEN]
LABELS = ["범주 1", "범주 2", "범주 3"]


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# === 모수 과잉: 로짓을 통째로 옮겨도 확률은 그대로 =======================
x = np.linspace(-3, 3, 1201)

wA = np.array([-1.5, 0.0, 1.5])
bA = np.array([-0.5, 1.0, -0.5])
dw, db = 2.0, -1.5                       # 세 범주에 공통으로 더하는 양
wB, bB = wA + dw, bA + db

ZA = x[:, None] * wA[None, :] + bA[None, :]
ZB = x[:, None] * wB[None, :] + bB[None, :]
PA = softmax_rows(ZA)
PB = softmax_rows(ZB)
gap = np.abs(PA - PB).max()

i0 = np.argmin(np.abs(x))                # x = 0 근처

fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
ax1, ax2, ax3 = axes

# --- 왼쪽·가운데: 두 모수화의 로짓 ---------------------------------------
for ax, Z, ttl, sub in (
        (ax1, ZA, "모수화 A의 로짓", None),
        (ax2, ZB, "모수화 B의 로짓", None)):
    for k in range(3):
        ax.plot(x, Z[:, k], color=COLORS[k], lw=2.3, label=LABELS[k])
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-6.5, 8.0)
    ax.set_xlabel("특성 x", fontsize=10.5)
    ax.set_title(ttl, fontsize=11.5, color=INK, pad=10)
    ax.grid(alpha=0.25, ls=":")
    ax.set_axisbelow(True)

ax1.set_ylabel("로짓 z", fontsize=10.5)
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

for ax, Z in ((ax1, ZA), (ax2, ZB)):
    for k in range(3):
        ax.plot(0, Z[i0, k], "o", color=COLORS[k], ms=7, zorder=5)
    txt = "  ".join(f"{Z[i0, k]:+.1f}" for k in range(3))
    ax.text(0.05, -6.1, f"x = 0 에서   ({txt})", ha="left", va="bottom",
            fontsize=10, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=MUTED, pad=3.0, alpha=0.95))

ax2.annotate("세 로짓 모두에\n같은 직선을 더했다", xy=(1.6, ZB[np.argmin(np.abs(x - 1.6)), 2]),
             xytext=(-2.75, 6.9), fontsize=9.5, color=PURPLE, fontweight="bold",
             ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))

# --- 오른쪽: 두 모수화의 확률이 겹친다 -----------------------------------
for k in range(3):
    ax3.plot(x, PA[:, k], color=COLORS[k], lw=4.5, alpha=0.35)
for k in range(3):
    ax3.plot(x, PB[:, k], color=COLORS[k], lw=1.6, ls="--")

ax3.axvline(0, color=MUTED, lw=1.0, ls=":")
for k in range(3):
    ax3.plot(0, PA[i0, k], "o", color=COLORS[k], ms=7, zorder=5)
txt = "  ".join(f"{PA[i0, k]:.3f}" for k in range(3))
ax3.text(0.05, 0.035, f"x = 0 에서   ({txt})", ha="left", va="bottom",
         fontsize=10, color=INK, fontweight="bold",
         bbox=dict(facecolor="white", edgecolor=MUTED, pad=3.0, alpha=0.95))

ax3.text(-2.9, 0.995, f"두 곡선의 최대 차이 = {gap:.1e}", ha="left", va="top",
         fontsize=10, color=RED, fontweight="bold")
ax3.text(-2.9, 0.905, "굵고 연한 선 = 모수화 A,  점선 = 모수화 B", ha="left",
         va="top", fontsize=9.5, color=INK,
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))

ax3.set_xlim(-3, 3)
ax3.set_ylim(0, 1.04)
ax3.set_xlabel("특성 x", fontsize=10.5)
ax3.set_ylabel("예측확률", fontsize=10.5)
ax3.set_title("확률은 한 점도 달라지지 않는다", fontsize=11.5, color=INK, pad=10)
ax3.grid(alpha=0.25, ls=":")
ax3.set_axisbelow(True)

for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "multinomial_shift_invariance.png", dpi=170,
            facecolor="white", bbox_inches="tight")
plt.close(fig)

print("A 의 x=0 로짓 =", np.round(ZA[i0], 3))
print("B 의 x=0 로짓 =", np.round(ZB[i0], 3))
print("두 모수화의 x=0 확률 =", np.round(PA[i0], 4), np.round(PB[i0], 4))
print("최대 확률 차이 =", gap)
print("wrote", OUT + "multinomial_shift_invariance.png")


# === 로그-합-지수: 순진한 구현이 깨지는 지점 ============================
LOG_MAX = np.log(np.finfo(np.float64).max)     # 709.78 -- exp 가 inf 로 넘침
LOG_TINY = np.log(np.finfo(np.float64).tiny)   # -708.4 -- 정규수의 하한
EPS = 1e-12

# --- 왼쪽: 로짓을 통째로 키우면 순진한 소프트맥스가 NaN 이 된다 ----------
M = np.linspace(0, 900, 1801)
shape = np.array([-1.0, 0.0, -2.0])            # 본문 보기와 같은 모양
naive = np.empty((M.size, 3))
stable = np.empty((M.size, 3))
with np.errstate(over="ignore", invalid="ignore"):
    for i, m in enumerate(M):
        z = m + shape
        e = np.exp(z)
        naive[i] = e / e.sum()
        s = np.exp(z - z.max())
        stable[i] = s / s.sum()

p_exact = stable[0, 1]
first_nan = M[np.argmax(np.isnan(naive[:, 1]))]

# --- 오른쪽: 참 범주가 최댓값보다 d 만큼 낮을 때의 손실 ------------------
d = np.linspace(0, 900, 1801)
loss_stable = d + np.log1p(np.exp(-d))         # = -log softmax_y, 정확한 값
with np.errstate(under="ignore", divide="ignore"):
    p_naive = np.exp(-d) / (1.0 + np.exp(-d))
    loss_naive = -np.log(p_naive)
    loss_eps = -np.log(p_naive + EPS)
loss_naive[~np.isfinite(loss_naive)] = np.nan
d_break = d[np.argmax(np.isnan(loss_naive))]
eps_cap = -np.log(EPS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

ax1.axvspan(LOG_MAX, 900, color=ORANGE_F, alpha=0.55, zorder=0)
ax1.plot(M, stable[:, 1], color=BLUE, lw=4.5, alpha=0.4,
         label="안정적 계산 (최댓값을 빼고 지수화)")
ax1.plot(M, naive[:, 1], color=RED, lw=1.8, ls="--",
         label="순진한 계산 (그대로 지수화)")
ax1.axvline(LOG_MAX, color=INK, lw=1.3, ls=":")
ax1.plot(first_nan, p_exact, "X", color=RED, ms=12, zorder=5)

ax1.annotate(f"여기서 끊긴다\n로짓 {first_nan:.0f} 부터 NaN",
             xy=(first_nan, p_exact), xytext=(430, 0.40), fontsize=9.5,
             color=RED, fontweight="bold", ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax1.text(LOG_MAX - 16, 0.94, f"exp 의 한계  {LOG_MAX:.1f}", ha="right",
         va="top", fontsize=9.5, color=INK, fontweight="bold")
ax1.text(30, p_exact + 0.045, f"참값 {p_exact:.4f} (로짓 크기와 무관)",
         ha="left", va="bottom", fontsize=9.5, color=BLUE, fontweight="bold")

ax1.set_xlim(0, 900)
ax1.set_ylim(0, 1.0)
ax1.set_xlabel("로짓의 최댓값", fontsize=10.5)
ax1.set_ylabel("가운데 범주의 예측확률", fontsize=10.5)
ax1.set_title("같은 확률을 묻는데 한쪽만 답을 못 한다", fontsize=11.5,
              color=INK, pad=10)
ax1.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

# --- 오른쪽 그림 ---------------------------------------------------------
ax2.axvspan(d_break, 900, color=ORANGE_F, alpha=0.55, zorder=0)
ax2.plot(d, loss_stable, color=BLUE, lw=4.5, alpha=0.4,
         label="안정적 계산")
ax2.plot(d, loss_naive, color=RED, lw=1.8, ls="--",
         label="확률을 먼저 구한 뒤 로그")
ax2.plot(d, loss_eps, color=GREEN, lw=2.0, ls="-.",
         label=r"$\varepsilon = 10^{-12}$ 보정")

ax2.axhline(eps_cap, color=GREEN, lw=1.0, ls=":")
ax2.plot(d_break, d_break, "X", color=RED, ms=12, zorder=5)
ax2.annotate(f"{d_break:.0f} 을 넘으면 확률이 0 으로\n사라져 로그가 무한대가 된다",
             xy=(d_break, d_break), xytext=(880, 330), fontsize=9.5, color=RED,
             fontweight="bold", ha="right", va="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax2.annotate(f"여기서 잘린다  {eps_cap:.1f}", xy=(330, eps_cap),
             xytext=(330, 205), fontsize=9.5, color=GREEN, fontweight="bold",
             ha="center", va="bottom",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

ax2.set_xlim(0, 900)
ax2.set_ylim(0, 950)
ax2.set_xlabel("참 범주 로짓이 최댓값보다 낮은 정도", fontsize=10.5)
ax2.set_ylabel("보고되는 교차엔트로피 손실", fontsize=10.5)
ax2.set_title("확신에 찬 오답을 세 구현이 다르게 본다", fontsize=11.5,
              color=INK, pad=10)
ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

for ax in (ax1, ax2):
    ax.grid(alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "numerical_stability_overflow.png", dpi=170,
            facecolor="white", bbox_inches="tight")
plt.close(fig)

print("exp 오버플로 한계 =", f"{LOG_MAX:.2f}", " NaN 시작 로짓 =", first_nan)
print("정확한 확률 =", np.round(stable[0], 4))
print("로그 발산 지점 d =", d_break, " eps 상한 =", f"{eps_cap:.3f}")
print("wrote", OUT + "numerical_stability_overflow.png")


# === 확률단체와 온도 척도화 ==============================================
V1 = np.array([0.0, 0.0])          # 범주 1 꼭짓점
V2 = np.array([1.0, 0.0])          # 범주 2 꼭짓점
V3 = np.array([0.5, np.sqrt(3) / 2])   # 범주 3 꼭짓점
VERT = np.vstack([V1, V2, V3])


def to_xy(P):
    """확률벡터 (n, 3) 를 정삼각형 좌표로 옮긴다."""
    return P @ VERT


z_ex = np.array([2.0, 1.0, -1.0])      # 본문 연습문제의 로짓
taus = np.array([0.2, 0.4, 0.7, 1.0, 1.6, 3.0, 6.0, 20.0])
P_tau = softmax_rows(z_ex[None, :] / taus[:, None])
tau_dense = np.geomspace(0.12, 30.0, 600)
P_dense = softmax_rows(z_ex[None, :] / tau_dense[:, None])

rng = np.random.default_rng(20)
Zrand = rng.uniform(-5, 5, size=(2200, 3))
Prand = softmax_rows(Zrand)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.3))

# --- 왼쪽: 확률단체 위의 온도 경로 ---------------------------------------
tri = np.vstack([VERT, VERT[0]])
ax1.fill(tri[:, 0], tri[:, 1], color="#F5F7F9", zorder=0)
ax1.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.6, ls="--", zorder=1)

XY = to_xy(Prand)
ax1.scatter(XY[:, 0], XY[:, 1], s=4, color=MUTED, alpha=0.30, zorder=2)

XYd = to_xy(P_dense)
ax1.plot(XYd[:, 0], XYd[:, 1], color=PURPLE, lw=2.4, zorder=3)
XYt = to_xy(P_tau)
ax1.scatter(XYt[:, 0], XYt[:, 1], s=46, color="white", edgecolors=PURPLE,
            linewidths=1.8, zorder=4)

cen = to_xy(np.array([[1 / 3, 1 / 3, 1 / 3]]))[0]
ax1.plot(*cen, "o", color=RED, ms=9, zorder=5)

for name, V, ha, va, dx, dy in (
        ("범주 1", V1, "right", "top", -0.02, -0.015),
        ("범주 2", V2, "left", "top", 0.02, -0.015),
        ("범주 3", V3, "center", "bottom", 0.0, 0.025)):
    ax1.text(V[0] + dx, V[1] + dy, name, ha=ha, va=va, fontsize=10.5,
             color=INK, fontweight="bold")

i1 = int(np.argmin(np.abs(taus - 1.0)))
ax1.annotate(f"온도 1:  ({P_tau[i1,0]:.3f}, {P_tau[i1,1]:.3f}, {P_tau[i1,2]:.3f})",
             xy=XYt[i1], xytext=(0.80, 0.14), fontsize=9.5, color=PURPLE,
             fontweight="bold", ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.92),
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
ax1.annotate("온도 → 0\n꼭짓점으로", xy=XYt[0], xytext=(0.05, 0.30),
             fontsize=9.5, color=PURPLE, fontweight="bold", ha="left",
             va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.92),
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
ax1.annotate("온도 → ∞\n한가운데로", xy=cen, xytext=(0.17, 0.66), fontsize=9.5,
             color=RED, fontweight="bold", ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.92),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax1.text(0.5, -0.115, "회색 점 2200개 — 무작위 로짓의 상(像)은 삼각형 내부를 채우되 변에 닿지 않는다",
         ha="center", va="top", fontsize=9.5, color=INK)

ax1.set_xlim(-0.14, 1.14)
ax1.set_ylim(-0.20, 1.02)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.set_title("확률단체: 소프트맥스의 출력이 사는 곳", fontsize=11.5,
              color=INK, pad=10)

# --- 오른쪽: 온도에 따른 세 확률 -----------------------------------------
for k in range(3):
    ax2.plot(tau_dense, P_dense[:, k], color=COLORS[k], lw=2.4,
             label=LABELS[k])
    ax2.scatter(taus, P_tau[:, k], s=30, color="white", edgecolors=COLORS[k],
                linewidths=1.6, zorder=4)

ax2.axhline(1 / 3, color=RED, lw=1.4, ls="--")
ax2.text(0.135, 1 / 3 + 0.02, "1/3", ha="left", va="bottom", fontsize=9.5,
         color=RED, fontweight="bold")
ax2.axvline(1.0, color=MUTED, lw=1.2, ls=":")
ax2.text(1.12, 0.985, "온도 1", ha="left", va="top", fontsize=9.5, color=INK,
         fontweight="bold")

ax2.set_xscale("log")
ticks = [0.2, 0.5, 1, 2, 5, 10, 30]
ax2.set_xticks(ticks)
ax2.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "30"])
ax2.set_xticks([], minor=True)
ax2.set_xlim(0.12, 30)
ax2.set_ylim(0, 1.03)
ax2.set_xlabel("온도 (가로축은 로그 눈금)", fontsize=10.5)
ax2.set_ylabel("예측확률", fontsize=10.5)
ax2.set_title("온도는 순서를 바꾸지 않고 확신만 조절한다", fontsize=11.5,
              color=INK, pad=10)
ax2.legend(loc="center left", fontsize=9.5, framealpha=0.95)
ax2.grid(alpha=0.25, ls=":")
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.savefig(OUT + "softmax_simplex_temperature.png", dpi=170,
            facecolor="white", bbox_inches="tight")
plt.close(fig)

for t, pv in zip(taus, P_tau):
    print(f"온도 {t:5.2f} -> ({pv[0]:.4f}, {pv[1]:.4f}, {pv[2]:.4f})")
print("무작위 로짓 상의 최소 성분 =", Prand.min())
print("wrote", OUT + "softmax_simplex_temperature.png")
