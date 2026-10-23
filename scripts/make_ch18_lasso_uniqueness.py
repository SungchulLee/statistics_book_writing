r"""라쏘의 해가 갈라지는 것은 p > n 이어서가 아니다.

이 쪽 절 2 의 표가 「해의 유일성 — 라쏘: p > n 이면 아님」이라고 적는다. 그런데
라이언 팁시라니(2013)의 보조정리 4 는 **X 의 성분이 연속분포에서 뽑혔으면 n 과 p
의 크기와 무관하게 확률 1 로 해가 유일하다**고 한다. 갈라지게 하는 것은 차원이
아니라 열이 「일반 위치」에 있지 않은 것 — 곧 이산 설명변수, 원핫 더미, 정확한
선형종속이다.

같은 논문의 보조정리 1 은 해가 여럿일 때에도 적합값 X*beta 와 L1 노름은 모든 해에서
같다고 한다. 그러니 갈리는 것은 **보고하는 변수 목록뿐**이다.

① lambda 를 움직이며 서로 다른 지지집합의 개수 — 연속 X 대 이산 X.
② 갈라진 해들을 나란히 — 어느 변수가 들고 나는가.
③ 보조정리 1 의 검산 — 계수는 다른데 적합값과 L1 노름은 같다.

만드는 파일: docs/ch18/overview/img/lasso_uniqueness.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.linear_model import Lasso                 # noqa: E402

NPERM = 60


def fit_perm(X, y, lam, nperm=NPERM, seed=0):
    """열 순서를 바꿔 가며 같은 문제를 푼다. 답이 하나면 늘 같아야 한다."""
    r = np.random.default_rng(seed)
    p = X.shape[1]
    sols, sups = [], []
    for _ in range(nperm):
        pi = r.permutation(p)
        m = Lasso(alpha=lam, fit_intercept=False, max_iter=200_000, tol=1e-12)
        m.fit(X[:, pi], y)
        b = np.zeros(p)
        b[pi] = m.coef_
        sols.append(b)
        sups.append(tuple(np.flatnonzero(np.abs(b) > 1e-8)))
    return np.array(sols), sups


# === 연속 X — p 가 n 의 다섯 배 ======================================
rng = np.random.default_rng(0)
NC, PC = 40, 200
XC = rng.normal(size=(NC, PC))
XC /= np.linalg.norm(XC, axis=0) / np.sqrt(NC)
YC = XC[:, :5] @ np.array([2.0, -1.5, 1.0, 0.8, -0.6]) + rng.normal(size=NC)

# === 이산 X — 범주형 더미에 정확한 선형종속이 섞인다 =================
ND = 60
A = rng.integers(0, 3, ND)
B = rng.integers(0, 4, ND)
D = np.column_stack([(A == k).astype(float) for k in range(3)]
                    + [(B == k).astype(float) for k in range(4)])
XD = np.column_stack([D, D[:, :3].sum(1)])          # 여덟째 열 = 앞 셋의 합
XD = np.column_stack([XD, rng.integers(0, 2, (ND, 6)).astype(float)])
YD = XD[:, 0] * 1.5 + XD[:, 3] * (-1.0) + rng.normal(size=ND) * 0.5

LAMS = np.logspace(np.log10(0.012), np.log10(0.6), 18)
NSUP_C, NSUP_D = [], []
for lam in LAMS:
    _, sc = fit_perm(XC, YC, lam)
    _, sd = fit_perm(XD, YD, lam)
    NSUP_C.append(len(set(sc)))
    NSUP_D.append(len(set(sd)))
NSUP_C, NSUP_D = np.array(NSUP_C), np.array(NSUP_D)

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"열 순서를 {NPERM} 가지로 바꿔 가며 같은 라쏘 문제를 푼다.")
print(f"연속 X: n = {NC}, p = {PC} (p 가 n 의 {PC//NC} 배)")
print(f"이산 X: n = {ND}, p = {XD.shape[1]} — 범주형 더미에 "
      "정확한 선형종속(여덟째 열 = 앞 세 열의 합)")
print("\n① 서로 다른 지지집합의 개수")
print("   lambda    연속 X   이산 X")
for i in (0, 3, 5, 7, 9, 12, 17):
    print(f"   {LAMS[i]:>6.3f}   {NSUP_C[i]:>6d}   {NSUP_D[i]:>6d}")
print(f"   연속 X 는 {len(LAMS)} 개 lambda 어디서도 {NSUP_C.max()} 개다 — "
      "**p 가 n 의 다섯 배인데도 해가 하나다.**")
print(f"   이산 X 는 최대 {NSUP_D.max()} 개까지 갈라진다.")

# === ② 갈라진 해들을 들여다본다 =====================================
i_sel = int(np.argmax(NSUP_D))
LAM_D = LAMS[i_sel]
SOLS, SUPS = fit_perm(XD, YD, LAM_D)
uniq = []
for s in SUPS:
    if s not in [u[0] for u in uniq]:
        uniq.append((s, SOLS[SUPS.index(s)]))
print(f"\n② lambda = {LAM_D:.3f} 에서 나온 서로 다른 해 {len(uniq)} 개")
for s, b in uniq:
    nz = ", ".join(f"x{j+1}" for j in s)
    print(f"   지지집합 ({len(s)} 개): {nz}")
U = set().union(*[set(t) for t, _ in uniq])
I = set.intersection(*[set(t) for t, _ in uniq])
dif = U - I
print(f"   목록이 갈리는 변수: "
      f"{', '.join('x%d' % (j+1) for j in sorted(dif))}")

# === ③ 보조정리 1 의 검산 ===========================================
B = np.array([b for _, b in uniq])
FIT = B @ XD.T
L1 = np.abs(B).sum(1)
d_beta = float(np.max(B.max(0) - B.min(0)))
d_fit = float(np.max(FIT.max(0) - FIT.min(0)))
d_l1 = float(L1.max() - L1.min())
print("\n③ 보조정리 1 의 검산 — 해는 다른데 무엇이 같은가")
print(f"   계수의 최대 차이          {d_beta:.3e}")
print(f"   적합값 X*beta 의 최대 차이  {d_fit:.3e}")
print(f"   L1 노름의 최대 차이        {d_l1:.3e}")
print("   **계수는 눈에 띄게 다르고 적합값과 L1 노름은 기계정밀도로 같다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_C, C_D = "#2ca02c", "#d62728"

ax1.semilogx(LAMS, NSUP_C, "o-", color=C_C, lw=2.8, ms=9,
             label=f"연속 $X$ ($n={NC},\\ p={PC}$)")
ax1.semilogx(LAMS, NSUP_D, "s-", color=C_D, lw=2.8, ms=9,
             label=f"이산 $X$ ($n={ND},\\ p={XD.shape[1]}$)")
ax1.axhline(1.0, color="#333333", lw=2.0, ls="--")
ax1.text(0.014, 1.09, "해가 하나", fontsize=10.5, color="#333333")
# 로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다 —
# 눈금을 십진수로 직접 적는다
_t = [0.02, 0.05, 0.1, 0.2, 0.5]
ax1.set_xticks(_t)
ax1.set_xticklabels([f"${v:g}$" for v in _t])
ax1.minorticks_off()
ax1.set_xlabel("$\\lambda$")
ax1.set_ylabel(f"열 순서 {NPERM} 가지에서 나온\n서로 다른 지지집합의 개수")
ax1.set_ylim(0.5, max(NSUP_D.max(), 2) + 0.6)
ax1.set_yticks(range(1, max(NSUP_D.max(), 2) + 1))
ax1.set_title("① 갈라지는 것은 $p > n$ 때문이 아니다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25)

w = 0.8 / len(uniq)
idx = np.arange(XD.shape[1])
for k, (s, b) in enumerate(uniq):
    ax2.bar(idx + (k - (len(uniq) - 1) / 2) * w, b, width=w,
            label=f"해 {k+1} (변수 {len(s)} 개)")
ax2.axhline(0, color="#333333", lw=1.4)
for j in sorted(dif):
    ax2.axvspan(j - 0.5, j + 0.5, color="#d62728", alpha=0.10)
ax2.set_xticks(idx)
ax2.set_xticklabels([f"$x_{{{j+1}}}$" for j in idx], fontsize=9)
ax2.set_ylabel("$\\hat\\beta_j$")
ax2.set_title(f"② $\\lambda = {LAM_D:.3f}$ 에서 나온 해들", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

vals = [d_beta, d_fit, d_l1]
# matplotlib mathtext 는 \lVert 를 모른다 — \| 를 쓴다
labs = ["계수\n$\\hat\\beta_j$", "적합값\n$X\\hat\\beta$",
        "$\\|\\hat\\beta\\|_1$"]
cols = [C_D, "#1f77b4", "#1f77b4"]
ax3.bar(range(3), np.maximum(vals, 1e-16), color=cols, width=0.6)
ax3.set_yscale("log")
ax3.set_xticks(range(3))
ax3.set_xticklabels(labs, fontsize=11)
ax3.set_ylabel("해들 사이의 최대 차이")
def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


for i, v in enumerate(vals):
    ax3.text(i, max(v, 1e-16) * 1.8, sci(v), ha="center", fontsize=11)
ax3.set_ylim(1e-16, 10)
ax3.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
ax3.set_yticklabels(["$10^{-15}$", "$10^{-12}$", "$10^{-9}$",
                     "$10^{-6}$", "$10^{-3}$", "$10^{0}$"])
ax3.minorticks_off()
ax3.set_title("③ 다른 것은 계수뿐이다", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch18/overview/img/lasso_uniqueness.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
