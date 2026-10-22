r"""평균제곱오차가 고르는 답은 「가장 그럴듯한 것」이 아니다 — 조건부 평균.

이 쪽 정리 1 은 $\mathrm{MSE} = \text{편향}^2 + \text{분산}$ 으로 가른다. 예측 문제에서
MSE 를 가장 작게 만드는 답은 **조건부 평균** $E[Y \mid X]$ 인데, 가능한 답이 여럿일 때
그 평균은 **어느 쪽도 아닌 자리**에 놓인다.

① 가지가 둘인 자료 — MSE 최적 예측은 자료가 하나도 없는 가운데다.
② 그래도 MSE 는 그쪽이 낮다. 둘 가운데 하나를 고르면 두 배다.
③ MNIST 로 — 같은 숫자 여러 장의 평균은 MSE 가 가장 낮은 한 장이지만 숫자가 아니다.

자료: data/MNIST/raw 의 idx 파일 (저장소에 담지 않는다 — .gitignore 의 data/).

만드는 파일: docs/ch05/estimators/img/mse_blurry_mean.png
"""

import os
import struct

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
MDIR = "data/MNIST/raw"

# === ①② 가지가 둘인 자료 ============================================
N = 4000
X = RNG.uniform(0, 1, N)
SGN = RNG.choice([-1.0, 1.0], N)
AMP = 1.0 + 0.5 * np.sin(2 * np.pi * X)
Y = SGN * AMP + RNG.normal(0, 0.10, N)

grid = np.linspace(0, 1, 200)
amp_g = 1.0 + 0.5 * np.sin(2 * np.pi * grid)
mse_pred = np.zeros_like(grid)          # E[Y|X] = 0
up_pred = amp_g                          # 늘 위쪽 가지
print("①② 가지가 둘인 자료 (y = ±(1 + 0.5 sin 2πx) + 잡음)")
mse_mid = float(np.mean((Y - 0.0) ** 2))
mse_up = float(np.mean((Y - (1.0 + 0.5 * np.sin(2 * np.pi * X))) ** 2))
mse_pick = float(np.mean((Y - SGN * AMP) ** 2))
print(f"   가운데(조건부 평균)로 예측  MSE = {mse_mid:.4f}")
print(f"   늘 위쪽 가지로 예측        MSE = {mse_up:.4f}  "
      f"({mse_up / mse_mid:.2f} 배)")
print(f"   맞는 가지를 알고 고르면    MSE = {mse_pick:.4f}")
print(f"   가운데는 자료가 하나도 없는 자리인데 MSE 는 그쪽이 낮다 — "
      f"MSE 가 고르는 것은 **그럴듯함이 아니라 평균**이다.")
near = float(np.mean(np.abs(Y) < 0.3))
print(f"   실제로 |y| < 0.3 인 관측은 전체의 {100 * near:.1f}% 다.")


# === ③ MNIST ========================================================
def load_idx(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            r, c = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)
        return np.frombuffer(f.read(), np.uint8)


HAVE = os.path.exists(f"{MDIR}/train-images-idx3-ubyte")
if HAVE:
    IM = load_idx(f"{MDIR}/train-images-idx3-ubyte").astype(np.float64) / 255
    LB = load_idx(f"{MDIR}/train-labels-idx1-ubyte")
    DIG = 3
    SUB = IM[LB == DIG]
    MEAN_IMG = SUB.mean(axis=0)
    print(f"\n③ MNIST 의 숫자 {DIG} {len(SUB):,} 장")
    d_mean = float(np.mean((SUB - MEAN_IMG) ** 2))
    picks = RNG.choice(len(SUB), 200, replace=False)
    d_one = float(np.mean([np.mean((SUB - SUB[i]) ** 2) for i in picks]))
    print(f"   평균 그림을 답으로 내면       MSE = {d_mean:.5f}")
    print(f"   진짜 한 장을 답으로 내면      MSE = {d_one:.5f}  "
          f"({d_one / d_mean:.2f} 배)")
    frac = float(np.mean((MEAN_IMG > 0.15) & (MEAN_IMG < 0.85)))
    print(f"   그런데 평균 그림은 화소의 {100 * frac:.1f}% 가 회색이다 — "
          f"어떤 손글씨에도 없는 값이고, 이것이 흐릿함의 정체다.")
else:
    print("\n③ data/MNIST/raw 가 없어 건너뛴다.")

# === 그림 ============================================================
fig = plt.figure(figsize=(15.6, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.26)
ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
C_D, C_M, C_U = "#9ecae1", "#d62728", "#2ca02c"

ax1.scatter(X, Y, s=5, color=C_D, alpha=0.5, linewidths=0)
ax1.plot(grid, mse_pred, color=C_M, lw=3.2, label="MSE 가 고르는 답 — $E[Y\\mid X]$")
ax1.plot(grid, up_pred, color=C_U, lw=2.4, ls="--", label="있을 법한 답 하나")
ax1.plot(grid, -up_pred, color=C_U, lw=2.4, ls="--")
ax1.annotate("자료가 하나도 없는 자리", xy=(0.5, 0.0), xytext=(0.42, 1.35),
             fontsize=10.5, color=C_M, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_M, lw=1.2))
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_title("① 가능한 답이 둘일 때", fontsize=12)
ax1.legend(loc="lower center", fontsize=9.5)
ax1.grid(alpha=0.25)

bars = [mse_mid, mse_up, mse_pick]
labs = ["가운데\n($E[Y\\mid X]$)", "늘 위쪽\n가지", "맞는 가지를\n알고 고르면"]
ax2.bar(range(3), bars, 0.56, color=[C_M, C_U, "#555555"])
for i, v in enumerate(bars):
    ax2.text(i, v + 0.06, f"${v:.3f}$", ha="center", fontsize=11)
ax2.set_xticks(range(3))
ax2.set_xticklabels(labs, fontsize=10)
ax2.set_ylabel("평균제곱오차")
ax2.set_ylim(0, max(bars) * 1.25)
ax2.set_title("② 그래도 MSE 는 가운데가 낮다", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

if HAVE:
    ax3.axis("off")
    sub = gs[2].subgridspec(2, 4, wspace=0.08, hspace=0.18)
    for j in range(4):
        a = fig.add_subplot(sub[0, j])
        a.imshow(SUB[j], cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        if j == 0:
            a.set_ylabel("진짜 손글씨", fontsize=10)
    a = fig.add_subplot(sub[1, 1:3])
    a.imshow(MEAN_IMG, cmap="gray_r", vmin=0, vmax=1)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel(f"{len(SUB):,} 장의 평균 — MSE 는 가장 낮고({d_mean:.3f}),\n"
                 f"진짜 한 장은 그 {d_one / d_mean:.1f} 배({d_one:.3f})다",
                 fontsize=10)
    ax3.set_title(f"③ MNIST 의 숫자 {DIG}", fontsize=12)
else:
    ax3.text(0.5, 0.5, "MNIST 자료 없음", ha="center", va="center")
    ax3.axis("off")

out = "docs/ch05/estimators/img/mse_blurry_mean.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
