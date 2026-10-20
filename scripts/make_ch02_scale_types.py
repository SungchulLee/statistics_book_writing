r"""척도를 잘못 잡으면 생기는 일 — 순서척도에 산술을, 명목척도에 순서를.

① 두 집단의 $5$ 점 척도 응답. 「매우 그렇다」를 몇 점으로 칠 것인가.
② 그 점수 하나로 두 집단의 순위가 뒤집힌다.
③ 명목변수를 정수로 바꾸면, **라벨을 붙인 순서**가 설명력을 정한다.

만드는 파일: docs/ch02/data_types/img/scale_types.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from itertools import permutations  # noqa: E402

# === ①② 순서척도에 산술 ============================================
LAB = ["전혀\n아니다", "아니다", "보통", "그렇다", "매우\n그렇다"]
A = np.array([10, 20, 40, 20, 10])  # 가운데로 모인 집단
B = np.array([33, 10, 20, 10, 27])  # 양끝으로 갈린 집단


def mean_with_top(cnt, top):
    sc = np.array([1.0, 2.0, 3.0, 4.0, top])
    return float(np.sum(cnt * sc) / cnt.sum())


tops = np.linspace(4.0, 8.0, 400)
mA = np.array([mean_with_top(A, t) for t in tops])
mB = np.array([mean_with_top(B, t) for t in tops])
# 교차점: sum(A*sc) = sum(B*sc) 를 top 에 대해 푼다
cA, cB = A / A.sum(), B / B.sum()
base = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
flip = (np.dot(cB, base) - np.dot(cA, base)) / (cA[4] - cB[4])

# === ③ 명목척도에 순서 ==============================================
RNG = np.random.default_rng(20261020)
K, NPG, SD = 6, 120, 1.0
EFFECT = np.array([0.0, 3.0, 1.0, 5.0, 2.0, 4.0])  # 범주마다 제멋대로인 참 효과
g = np.repeat(np.arange(K), NPG)
y = EFFECT[g] + RNG.normal(0, SD, K * NPG)


def r2_onehot(g, y):
    """범주별 평균으로 맞추는 것이 원-핫 선형모형의 적합과 같다."""
    yhat = np.array([y[g == j].mean() for j in range(K)])[g]
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def r2_label(g, y, order):
    """order[j] = 범주 j 에 붙인 정수. 그 정수 하나로 단순회귀."""
    x = np.asarray(order, dtype=float)[g]
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    yhat = y.mean() + b * (x - x.mean())
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


R2_OH = r2_onehot(g, y)
r2s = np.array([r2_label(g, y, p) for p in permutations(range(K))])

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

xs = np.arange(5)
w = 0.38
ax1.bar(xs - w / 2, A, w, color="#1f77b4", label="집단 A (가운데로 모임)")
ax1.bar(xs + w / 2, B, w, color="#d62728", label="집단 B (양끝으로 갈림)")
for x, a, b in zip(xs, A, B):
    ax1.text(x - w / 2, a + 1.2, str(a), ha="center", fontsize=9.5)
    ax1.text(x + w / 2, b + 1.2, str(b), ha="center", fontsize=9.5)
ax1.set_xticks(xs)
ax1.set_xticklabels(LAB, fontsize=9.5)
ax1.set_ylabel("응답자 수 (각 $100$ 명)")
ax1.set_title("① 같은 문항, 두 집단", fontsize=12)
ax1.legend(loc="upper center", fontsize=9.5)
ax1.set_ylim(0, 52)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(tops, mA, color="#1f77b4", lw=2.5, label="집단 A 의 평균")
ax2.plot(tops, mB, color="#d62728", lw=2.5, label="집단 B 의 평균")
ax2.axvline(flip, color="#555555", lw=1.2, ls="--")
ax2.plot([flip], [mean_with_top(A, flip)], "o", color="black", ms=8, zorder=5)
ax2.annotate(
    f"여기서 뒤집힌다\n「매우 그렇다」$= {flip:.2f}$",
    xy=(flip, mean_with_top(A, flip)),
    xytext=(6.9, 2.95),
    ha="center",
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1),
)
for t, c in [(5.0, "#2ca02c"), (7.0, "#ff7f0e")]:
    ax2.axvline(t, color=c, lw=1.0, ls=":")
    ax2.text(
        t + 0.06,
        3.62,
        f"${t:.0f}$ 점이면\nA 가 ${mean_with_top(A, t) - mean_with_top(B, t):+.2f}$",
        fontsize=9,
        color=c,
    )
ax2.set_xlabel("「매우 그렇다」에 주는 점수  ($1$–$4$ 는 고정)")
ax2.set_ylabel("평균 점수")
ax2.set_title("② 점수 하나가 순위를 바꾼다", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5)
ax2.set_xlim(4, 8)
ax2.set_ylim(2.75, 3.95)
ax2.grid(alpha=0.25)

ax3.hist(r2s, bins=40, color="#9467bd", alpha=0.85, edgecolor="none")
ax3.axvline(R2_OH, color="#2ca02c", lw=2.4, label=f"원-핫  $R^2 = {R2_OH:.3f}$")
ax3.axvline(r2s.max(), color="#d62728", lw=1.6, ls="--", label=f"가장 운 좋은 순서  ${r2s.max():.3f}$")
ax3.axvline(r2s.min(), color="#1f77b4", lw=1.6, ls="--", label=f"가장 나쁜 순서  ${r2s.min():.3f}$")
ax3.set_xlabel("$R^2$   (범주 $6$ 개에 정수를 붙이는 $720$ 가지)")
ax3.set_ylabel("순서의 수")
ax3.set_title("③ 라벨을 붙인 순서가 설명력을 정한다", fontsize=12)
ax3.legend(loc="upper center", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch02/data_types/img/scale_types.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n①② 순서척도에 산술")
for t in (4.1, 5.0, flip, 6.0, 7.0, 8.0):
    a, b = mean_with_top(A, t), mean_with_top(B, t)
    who = "A" if a > b else ("B" if b > a else "=")
    print(f"  「매우 그렇다」={t:5.2f}:  A {a:.3f}   B {b:.3f}   차 {a - b:+.3f}  → {who}")
print(f"  뒤집히는 점수: {flip:.4f}")
print(f"  곧 「그렇다 → 매우 그렇다」 간격이 다른 간격의 {flip - 4:.2f} 배를 넘으면 순위가 바뀐다")

print("\n③ 명목척도에 순서")
print(f"  참 효과: {list(EFFECT)}")
print(f"  원-핫 R^2            = {R2_OH:.4f}")
print(f"  정수 라벨 R^2 — 최솟값 {r2s.min():.4f}, 중앙값 {np.median(r2s):.4f}, 최댓값 {r2s.max():.4f}")
print(f"  720 가지 가운데 원-핫의 90% 를 넘는 순서: {np.mean(r2s > 0.9 * R2_OH) * 100:.1f}%")
print(f"  알파벳 순(0,1,2,3,4,5)으로 붙이면: {r2_label(g, y, range(K)):.4f}")
