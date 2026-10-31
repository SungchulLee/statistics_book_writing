"""9.6절 본페로니와 홀름 — 열 번 견주면 네 번은 헛것이 보인다.

들머리의 광고 시안 보기를 모의로 잰다. 시안 다섯 개를 두 개씩 짝지으면
비교가 열 번이고, 시안들 사이에 **아무 차이가 없어도** 하나하나를
alpha = 0.05 로 판정하면 적어도 하나가 유의하게 나올 확률이 0.4013 이다.

쪽의 bonferroni_vs_holm 은 p 값 네 개에 두 문턱을 대는 그림이고
correction_comparison 은 네 방법을 견준다. 여기서는 보정 이야기에 들어가기
전의 자리 --- 보정이 왜 필요한가 --- 만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from itertools import combinations
from scipy import stats

OUT = "docs/ch09/multiple_testing/img/"
K, N, R, A = 5, 30, 20_000, 0.05
PAIRS = list(combinations(range(K), 2))
rng = np.random.default_rng(405)

x = rng.standard_normal((R, K, N))          # 다섯 시안이 모두 같은 분포다
m = x.mean(axis=2)
v = x.var(axis=2, ddof=1)
cnt = np.zeros(R, int)
first_sig = None
for i, j in PAIRS:
    t = (m[:, i] - m[:, j]) / np.sqrt(v[:, i] / N + v[:, j] / N)
    df = 2 * N - 2
    cnt += (2 * stats.t.sf(np.abs(t), df) < A)
any_rate = (cnt > 0).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.5),
                               gridspec_kw={"width_ratios": [1, 1.1]})

# ── (좌) 다섯을 두 개씩 --- 열 번 ─────────────────────────────────
ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, K, endpoint=False)
px, py = np.cos(ang), np.sin(ang)
for i, j in PAIRS:
    ax1.plot([px[i], px[j]], [py[i], py[j]], "-", color="0.65", lw=2.0)
for i in range(K):
    ax1.plot(px[i], py[i], "o", ms=30, color="tab:blue", alpha=0.9)
    ax1.text(px[i], py[i], f"{i+1}", ha="center", va="center", fontsize=14,
             fontweight="bold", color="white")
ax1.text(0, -1.52, f"시안 {K} 개 → 두 개씩 견주면 {len(PAIRS)} 번",
         ha="center", fontsize=13, fontweight="bold", color="0.2")
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.75, 1.35)
ax1.axis("off")
ax1.set_title("비교의 수는 시안 수보다 빨리 는다", fontsize=12.5)

# ── (우) 유의하게 나온 쌍의 수 ─────────────────────────────────────
vals, counts = np.unique(cnt, return_counts=True)
frac = counts / R
cols = ["0.65"] + ["tab:red"] * (len(vals) - 1)
ax2.bar(vals, frac, width=0.72, color=cols, alpha=0.88)
for v_, f_ in zip(vals[:5], frac[:5]):
    ax2.text(v_, f_ + 0.012, f"{f_:.4f}", ha="center", fontsize=10.5,
             color="0.2", fontweight="bold")
ax2.set_xticks(vals[vals <= 5])
ax2.set_xlim(-0.6, 5.6)
ax2.set_ylim(0, 0.72)
ax2.set_xlabel("「유의하다」로 나온 쌍의 수", fontsize=11)
ax2.set_ylabel("연구의 비율", fontsize=11)
ax2.set_title(f"다섯 시안이 모두 같은데 되풀이 {R:,} 번", fontsize=12.5)
ax2.text(0.97, 0.80,
         f"하나라도 나온 연구 {any_rate:.4f}\n"
         f"(독립이라 치면 {1-(1-A)**len(PAIRS):.4f})",
         transform=ax2.transAxes, ha="right", fontsize=13,
         fontweight="bold", color="tab:red",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="tab:red", lw=1.2))
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("차이가 전혀 없는 다섯 시안인데 열 연구에 셋쯤은 헛것을 본다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "ten_comparisons.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  쌍 {len(PAIRS)} 개 · 되풀이 {R:,}")
print(f"  적어도 하나가 유의: {any_rate:.4f}  (상한 {1-(1-A)**len(PAIRS):.4f})")
print(f"  유의한 쌍의 수별 비율: "
      f"{dict(zip(vals[:6].tolist(), np.round(frac[:6], 4).tolist()))}")
