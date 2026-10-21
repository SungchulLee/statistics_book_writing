r"""학습된 임베딩의 코사인은 모형이 정해 주지 않는다.

예측이 $AB^\top$ 에만 달려 있으면, 가역행렬 $M$ 으로 $A \to AM$, $B \to BM^{-\top}$ 를
해도 $AB^\top$ 이 **그대로**다. 예측은 한 치도 안 바뀌는데 $A$ 의 행들이 이루는 각은
바뀐다 — 코사인이 모형의 성질이 아니라 **어쩌다 고른 분해**의 성질이라는 뜻이다.

① 같은 모형, 다른 코사인 — 짝별 코사인의 산점도.
② 코사인 분포 자체가 달라진다.
③ 「가장 비슷한 것 열 개」가 얼마나 바뀌는가.

만드는 파일: docs/ch12/correlation_test/img/cosine_arbitrary.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N_ITEM, N_USER, K = 300, 500, 20
TOPK = 10


def cos_matrix(V):
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    return U @ U.T


# --- 낮은 계수의 참 행렬에서 자료를 만들고 분해한다 -------------------
A0 = RNG.standard_normal((N_ITEM, K))
B0 = RNG.standard_normal((N_USER, K))
X = A0 @ B0.T + 0.3 * RNG.standard_normal((N_ITEM, N_USER))

U, s, Vt = np.linalg.svd(X, full_matrices=False)
A = U[:, :K] * np.sqrt(s[:K])          # 흔히 쓰는 「균형 잡힌」 분해
B = (Vt[:K].T) * np.sqrt(s[:K])

# --- 예측을 바꾸지 않는 변환 -----------------------------------------
# 직교행렬만 쓰면 각이 그대로이므로 코사인도 그대로다. 축마다 눈금을
# 다르게 주는 성분이 있어야 각이 바뀐다.
Q, _ = np.linalg.qr(RNG.standard_normal((K, K)))
scale = np.logspace(-0.7, 0.7, K)      # 축마다 눈금을 달리 준다
M = Q @ np.diag(scale)
A2 = A @ M
B2 = B @ np.linalg.inv(M).T

pred_gap = np.abs(A2 @ B2.T - A @ B.T).max()
print(f"① 예측의 최대 차이 |A'B'^T - AB^T| = {pred_gap:.3e}")
print(f"   (행렬 성분의 크기는 {np.abs(A @ B.T).mean():.2f} 쯤이다)")

C1, C2 = cos_matrix(A), cos_matrix(A2)
iu = np.triu_indices(N_ITEM, 1)
c1, c2 = C1[iu], C2[iu]
print(f"② 코사인 — 원래 평균 {c1.mean():+.3f} 표준편차 {c1.std():.3f}, "
      f"변환 뒤 평균 {c2.mean():+.3f} 표준편차 {c2.std():.3f}")
print(f"   두 코사인의 상관 {np.corrcoef(c1, c2)[0, 1]:.3f}, "
      f"스피어만 {stats.spearmanr(c1, c2).statistic:.3f}")

# 직교변환만 걸면 어떻게 되는가 (대조)
A3 = A @ Q
c3 = cos_matrix(A3)[iu]
print(f"   직교변환만 걸면 코사인 최대 차이 {np.abs(c3 - c1).max():.3e} — 그대로다")


def topk_sets(C, k=TOPK):
    C = C.copy()
    np.fill_diagonal(C, -np.inf)
    return [set(np.argpartition(-row, k)[:k]) for row in C]


T1, T2 = topk_sets(C1), topk_sets(C2)
ov = np.array([len(a & b) for a, b in zip(T1, T2)])
print(f"③ 상위 {TOPK} 이웃의 겹침 — 평균 {ov.mean():.2f}/{TOPK}, "
      f"하나도 안 겹친 항목 {np.sum(ov == 0)} 개, "
      f"절반 이상 겹친 항목 {np.sum(ov >= TOPK / 2)} 개")

# --- 이 쪽의 t 검정을 코사인에 걸면 ----------------------------------
print("\n차원 d 에서 코사인 r 에 이 쪽의 t 검정을 걸면")
for d in (8, 128, 768, 4096):
    for r in (0.05, 0.10):
        t = r * np.sqrt((d - 2) / (1 - r**2))
        p = 2 * stats.t.sf(abs(t), d - 2)
        print(f"   d = {d:>5}, r = {r:.2f}  →  p = {p:.2e}")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

sel = RNG.choice(len(c1), 6000, replace=False)
ax1.scatter(c1[sel], c2[sel], s=5, alpha=0.25, color="#1f77b4",
            edgecolors="none")
lim = 1.02
ax1.plot([-lim, lim], [-lim, lim], color="#d62728", lw=1.6, ls="--",
         label="바뀌지 않았다면 이 선 위")
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
ax1.set_xlabel("원래 분해의 코사인")
ax1.set_ylabel("$M$ 을 건 뒤의 코사인")
ax1.set_title("① 예측은 그대로, 코사인은 다르다", fontsize=12)
# 지수표기를 수식으로 적는다. 문자열 치환으로 만들면 쉽게 깨진다.
_m, _e = f"{pred_gap:.0e}".split("e")
ax1.text(-0.97, 0.92,
         f"예측의 최대 차이\n${_m} \\times 10^{{{int(_e)}}}$",
         fontsize=10, va="top",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(alpha=0.25)

bins = np.linspace(-1, 1, 70)
ax2.hist(c1, bins=bins, density=True, color="#1f77b4", alpha=0.6,
         edgecolor="none", label="원래 분해")
ax2.hist(c2, bins=bins, density=True, color="#d62728", alpha=0.55,
         edgecolor="none", label="$M$ 을 건 뒤")
ax2.set_xlabel("코사인 유사도")
ax2.set_ylabel("밀도")
ax2.set_title("② 분포 자체가 달라진다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.hist(ov, bins=np.arange(-0.5, TOPK + 1.5), color="#9e9e9e",
         edgecolor="white")
ax3.axvline(ov.mean(), color="#d62728", lw=2.0, ls="--")
ax3.text(ov.mean() + 0.25, ax3.get_ylim()[1] * 0.88,
         f"평균 ${ov.mean():.1f}$ / ${TOPK}$", fontsize=11, color="#d62728")
ax3.set_xlabel(f"상위 ${TOPK}$ 이웃 가운데 그대로 남은 개수")
ax3.set_ylabel("항목 수")
ax3.set_title(f"③ 「가장 비슷한 것 ${TOPK}$ 개」가 바뀐다", fontsize=12)
ax3.set_xticks(range(TOPK + 1))
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch12/correlation_test/img/cosine_arbitrary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
