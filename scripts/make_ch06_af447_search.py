r"""6.4절 「그림으로 보는 통계학」 — 찾지 못했다는 사실도 자료다.

$2009$ 년 대서양에서 사라진 비행기를 두 해 동안 찾지 못한 뒤, 수색 계획을 베이즈로
다시 세웠다. 핵심은 **실패한 수색을 어떻게 읽느냐**다. 수색이 훑고 지나갔는데 아무것도
없었다면 그 칸의 확률은 내려가야 한다 — 다만 **그 수색이 실제로 볼 수 있었을 때만**
그렇다. 입자 하나하나에 대해

    사후 가중치 ∝ (1 - 그 자리에서 탐지됐을 확률) × 사전 가중치

를 곱하는 것이 전부다.

① 사전분포 — 마지막 교신 지점을 둘러싼 원과 활공 가설.
② 비컨이 작동했다고 보면 — 한가운데가 비워진다.
③ 비컨이 고장났다고 보면 — 같은 실패가 아무것도 말해 주지 않는다.

자료는 **직접 만든 것**이다(실제 수색 자료가 아니다).

만드는 파일: docs/ch06/bayesian/img/af447_search.png
실행:  python3 scripts/make_ch06_af447_search.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/bayesian/img/af447_search.png"
RNG = np.random.default_rng(20261022)

NP = 400_000            # 입자 수
R_SEARCH = 20.0         # 음향 수색이 훑은 반지름 (해리)
PD_WORK, PD_FAIL = 0.90, 0.05    # 비컨이 살았을 때와 죽었을 때의 탐지확률
TRUE = np.array([3.5, -2.0])     # 잔해가 실제로 있던 자리 (원의 한가운데 가까이)

# --- 사전분포: 추락 지점 근처 + 활공 가설의 고리 ------------------------
n1 = NP // 2
near = RNG.normal(0, 9.0, (n1, 2))
ang = RNG.uniform(0, 2 * np.pi, NP - n1)
rad = RNG.normal(26, 6.0, NP - n1)
ring = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
pts = np.vstack([near, ring])
w0 = np.ones(NP) / NP

inside = np.hypot(pts[:, 0], pts[:, 1]) <= R_SEARCH
print(f"사전분포에서 반지름 {R_SEARCH:.0f} 해리 안에 든 확률 = {w0[inside].sum():.3f}")


def update(w, pd_inside):
    """찾지 못한 수색 하나를 반영한다 — 식 (4.1)."""
    pd = np.where(inside, pd_inside, 0.0)
    w = (1 - pd) * w
    return w / w.sum()


w_work = update(w0, PD_WORK)
w_fail = update(w0, PD_FAIL)
print(f"실패한 수색 뒤 그 안에 남는 확률:")
print(f"   비컨이 살아 있었다고 보면 {w_work[inside].sum():.3f}")
print(f"   비컨이 죽었다고 보면     {w_fail[inside].sum():.3f}")

# 잔해 자리 둘레의 작은 상자에 각 분포가 얼마를 주는가
box = (np.abs(pts[:, 0] - TRUE[0]) < 5) & (np.abs(pts[:, 1] - TRUE[1]) < 5)
print(f"\n잔해가 있던 10 x 10 해리 상자에 주는 확률")
for nm, w in (("사전", w0), ("비컨 작동 가정", w_work), ("비컨 고장 가정", w_fail)):
    print(f"   {nm:>12s}: {w[box].sum():.4f}")
print(f"   두 가정의 비 = {w_fail[box].sum() / w_work[box].sum():.1f} 배")

# --- 그림 ---------------------------------------------------------------
LIM, BINS = 45, 160
edges = np.linspace(-LIM, LIM, BINS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
titles = ("① 사전분포 — 찾기 전",
          f"② 비컨이 작동했다면 (탐지확률 {PD_WORK:.2f})",
          f"③ 비컨이 고장났다면 (탐지확률 {PD_FAIL:.2f})")
th = np.linspace(0, 2 * np.pi, 400)

for ax, w, t in zip(axes, (w0, w_work, w_fail), titles):
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[edges, edges], weights=w)
    ax.imshow(H.T, origin="lower", extent=[-LIM, LIM, -LIM, LIM],
              cmap="YlOrBr", vmin=0, vmax=np.percentile(H[H > 0], 99.5))
    ax.plot(R_SEARCH * np.cos(th), R_SEARCH * np.sin(th), color="#1f6f8b",
            lw=1.6, ls="--")
    ax.plot([0], [0], "+", color="#1f6f8b", ms=12, mew=2)
    ax.plot(*TRUE, "o", color="#c0532b", ms=9, mew=1.5, mfc="none")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel("동서 (해리)")
    ax.set_title(t, fontsize=11)
    ax.text(0.03, 0.03, f"원 안의 확률 {w[inside].sum():.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92))

axes[0].set_ylabel("남북 (해리)")
axes[0].annotate("마지막 교신 지점", xy=(0, 0), xytext=(-41, 36),
                 color="#1f6f8b", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#1f6f8b"))
axes[2].annotate("잔해가 있던 자리", xy=TRUE, xytext=(10, -34), color="#c0532b",
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="#c0532b"))

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
