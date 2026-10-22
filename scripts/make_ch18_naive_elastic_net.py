r"""조우와 헤이스티의 보기 4 를 그대로 재현한다.

그들의 $2005$ 년 논문은 자기네 첫 판에 **「순진한」**이라는 이름을 붙였다.
두 벌점을 그냥 더하면 계수가 **두 번 수축**되어, 능형이나 라쏘에 아주
가깝지 않은 한 쓸 만한 성능이 나오지 않았기 때문이다. 고치는 방법은
곱하기 한 번 — $\hat\beta(\text{EN}) = (1+\lambda_2)\,\hat\beta(\text{순진한})$ 이다.

보기 4 (집단 변수): $50/50/400$, $p = 40$, $\beta = (3 \times 15,\ 0 \times 25)$,
$\sigma = 15$. $x_i = Z_g + \varepsilon_i$ 로 다섯씩 세 집단을 이루고
나머지 $25$ 개는 순수 잡음이다.

① 집단마다 몇 명을 뽑는가 — 「큰 물고기를 다 잡는 그물」이 맞는가.
② 논문 표 $2$ 의 네 수를 다시 계산한다.
③ $\lambda_2$ 에 따라 보정이 얼마나 벌어지는가.

만드는 파일: docs/ch18/elastic_net/img/naive_elastic_net.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso, Ridge   # noqa: E402

RNG = np.random.default_rng(1)
P, SIG, NTR, NVA, REPS = 40, 15.0, 50, 50, 50
BETA = np.r_[np.full(15, 3.0), np.zeros(25)]
L1S = np.logspace(-2, 2.6, 26)
L2S = np.r_[0.0, np.logspace(-3, 2.2, 16)]
PAPER = {"라쏘": 46.6, "엘라스틱넷": 34.5, "능형": 64.5, "순진한 EN": 45.9}

SIGMA = np.zeros((P, P))                       # 참 공분산
for g in range(3):
    s = slice(g * 5, (g + 1) * 5)
    SIGMA[s, s] = 1.0
np.fill_diagonal(SIGMA, np.r_[np.full(15, 1.01), np.ones(25)])


def gen(m):
    Z = RNG.normal(0, 1, (m, 3))
    X = np.empty((m, P))
    for g in range(3):
        X[:, g * 5:(g + 1) * 5] = Z[:, [g]] + RNG.normal(0, 0.1, (m, 5))
    X[:, 15:] = RNG.normal(0, 1, (m, 25))
    return X, X @ BETA + RNG.normal(0, SIG, m)


def en(X, y, l1, l2):
    """논문 식 (3) 의 순진한 EN 과 식 (11) 의 보정 EN 을 함께 돌려준다."""
    p = X.shape[1]
    s = np.sqrt(1 + l2)
    Xa = np.vstack([X, np.sqrt(l2) * np.eye(p)]) / s
    ya = np.r_[y, np.zeros(p)]
    bs = Lasso(alpha=(l1 / s) / (2 * len(ya)), fit_intercept=False,
               max_iter=100000).fit(Xa, ya).coef_
    return bs / s, s * bs


vmse = lambda b, X, y: float(np.mean((X @ b - y) ** 2))   # noqa: E731


def ME(b_unit, nr):
    d = b_unit / nr - BETA
    return float(d @ SIGMA @ d)


KEYS = ["라쏘", "엘라스틱넷", "능형", "순진한 EN"]
R = {k: [] for k in KEYS}
GRP = {"라쏘": [], "엘라스틱넷": []}
FALSE = {"라쏘": [], "엘라스틱넷": []}
CURVE_N = np.zeros((REPS, len(L2S)))
CURVE_E = np.zeros((REPS, len(L2S)))

for rep in range(REPS):
    Xtr, ytr = gen(NTR)
    Xva, yva = gen(NVA)
    nr = np.linalg.norm(Xtr, axis=0)           # 논문의 단위노름 표준화
    Xtr, Xva = Xtr / nr, Xva / nr
    mu = ytr.mean()
    ytr, yva = ytr - mu, yva - mu
    best = {k: (1e18, None) for k in KEYS}
    for a, l2 in enumerate(L2S):
        bn_best = be_best = (1e18, None)
        for l1 in L1S:
            bn, be = en(Xtr, ytr, l1, l2)
            vn, ve = vmse(bn, Xva, yva), vmse(be, Xva, yva)
            if vn < bn_best[0]:
                bn_best = (vn, bn)
            if ve < be_best[0]:
                be_best = (ve, be)
            if l2 == 0.0 and vn < best["라쏘"][0]:
                best["라쏘"] = (vn, bn)
            if vn < best["순진한 EN"][0]:
                best["순진한 EN"] = (vn, bn)
            if ve < best["엘라스틱넷"][0]:
                best["엘라스틱넷"] = (ve, be)
        CURVE_N[rep, a] = ME(bn_best[1], nr)
        CURVE_E[rep, a] = ME(be_best[1], nr)
    for al in np.logspace(-6, 3, 40):
        b = Ridge(alpha=al, fit_intercept=False).fit(Xtr, ytr).coef_
        v = vmse(b, Xva, yva)
        if v < best["능형"][0]:
            best["능형"] = (v, b)
    for k in KEYS:
        R[k].append(ME(best[k][1], nr))
    for k in GRP:
        nz = best[k][1] != 0
        GRP[k].append([int(nz[g * 5:(g + 1) * 5].sum()) for g in range(3)])
        FALSE[k].append(int(nz[15:].sum()))

print(f"보기 4 — {NTR}/{NVA}/400, p = {P}, sigma = {SIG}, "
      f"자료 {REPS} 벌")
print(f"\n② 모형오차의 중앙값")
print(f"{'방법':<12}{'내 계산':>9}{'논문':>8}")
for k in KEYS:
    print(f"{k:<12}{np.median(R[k]):>9.1f}{PAPER[k]:>8.1f}")
print(f"   라쏘 대비 엘라스틱넷의 감소 "
      f"{100 * (1 - np.median(R['엘라스틱넷']) / np.median(R['라쏘'])):.0f}% "
      f"(논문 27%)")
print(f"   순진한 EN 은 라쏘보다 "
      f"{'나쁘다' if np.median(R['순진한 EN']) > np.median(R['라쏘']) else '낫다'} "
      f"({np.median(R['순진한 EN']):.1f} 대 {np.median(R['라쏘']):.1f}; "
      f"논문 {PAPER['순진한 EN']} 대 {PAPER['라쏘']})")

print(f"\n① 참 집단(다섯 명)마다 몇 명을 뽑았나 · 가짜 {25} 개 중 몇 개를 집었나")
for k in GRP:
    g = np.array(GRP[k])
    print(f"   {k:<10} 집단당 평균 {g.mean():.2f} 명 / 5,  "
          f"가짜 평균 {np.mean(FALSE[k]):.1f} 개,  "
          f"0 이 아닌 계수 중앙값 {np.median(g.sum(axis=1) + np.array(FALSE[k])):.0f}")
print("   (논문 표 3 의 0 이 아닌 계수 중앙값 — 라쏘 11, 엘라스틱넷 16)")

mn, me_ = np.median(CURVE_N, axis=0), np.median(CURVE_E, axis=0)
print(f"\n③ lambda_2 에 따라 (각 lambda_2 에서 lambda_1 은 검증집합으로 고름)")
print(f"{'lambda_2':>10}{'순진한 EN':>11}{'보정 EN':>10}")
for a in (0, 6, 10, 13, 16):
    print(f"{L2S[a]:>10.4g}{mn[a]:>11.1f}{me_[a]:>10.1f}")
jn, je = int(mn.argmin()), int(me_.argmin())
print(f"   순진한 EN 이 가장 좋은 자리 : lambda_2 = {L2S[jn]:.4g}, "
      f"{mn[jn]:.1f}  (라쏘 {mn[0]:.1f} 보다 "
      f"{'낫다' if mn[jn] < mn[0] else '낫지 않다'})")
print(f"   보정 EN 이 가장 좋은 자리   : lambda_2 = {L2S[je]:.4g}, "
      f"{me_[je]:.1f}  (라쏘 대비 "
      f"{100 * (1 - me_[je] / mn[0]):.0f}% 감소)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C_L, C_E, C_N, C_R = "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"

w = 0.36
xs = np.arange(3)
for k, c, off in [("라쏘", C_L, -w / 2), ("엘라스틱넷", C_E, w / 2)]:
    g = np.array(GRP[k]).mean(axis=0)
    ax1.bar(xs + off, g, w, color=c, label=k)
    for x, v in zip(xs, g):
        ax1.text(x + off, v + 0.1, f"${v:.1f}$", ha="center", fontsize=10)
ax1.axhline(5, color="black", lw=1.5, ls="--")
ax1.text(2.42, 5.12, "집단의 크기 $5$", fontsize=10, ha="right")
ax1.set_xticks(xs)
ax1.set_xticklabels([f"{g + 1} 집단" for g in range(3)], fontsize=10.5)
ax1.set_ylabel("뽑힌 참 변수의 수 (평균)")
ax1.set_ylim(0, 6.0)
ax1.set_title("① 큰 물고기를 다 잡는가", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

order = ["엘라스틱넷", "라쏘", "순진한 EN", "능형"]
bp = ax2.boxplot([R[k] for k in order], widths=0.55, showfliers=False,
                 patch_artist=True, medianprops=dict(color="black", lw=2))
for patch, c in zip(bp["boxes"], [C_E, C_L, C_N, C_R]):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for i, k in enumerate(order):
    ax2.plot([i + 1], [PAPER[k]], "*", color="black", ms=16, zorder=5)
    ax2.text(i + 1, np.percentile(R[k], 95) + 6,
             f"중앙값 ${np.median(R[k]):.0f}$", ha="center", fontsize=10)
ax2.plot([], [], "*", color="black", ms=13, label="논문 표 $2$ 의 값")
ax2.set_xticklabels(order, fontsize=10.5)
ax2.set_ylabel("모형오차")
ax2.set_ylim(0, 170)
ax2.set_title("② 논문 표 $2$ 를 다시 계산하면", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

msk = L2S > 0
ax3.semilogx(L2S[msk], mn[msk], "s-", color=C_N, lw=2.6, ms=6,
             label="순진한 EN")
ax3.semilogx(L2S[msk], me_[msk], "o-", color=C_E, lw=2.6, ms=6,
             label="보정 EN  $(1+\\lambda_2)\\,\\hat\\beta$")
ax3.set_yscale("log")
ax3.axhline(mn[0], color=C_L, lw=1.8, ls="--")
ax3.text(1.1e-3, mn[0] * 0.80, "라쏘 ($\\lambda_2 = 0$)", fontsize=10,
         color=C_L)
ax3.set_yticks([30, 50, 100, 200, 400, 800])
ax3.set_yticklabels(["$30$", "$50$", "$100$", "$200$", "$400$", "$800$"])
ax3.set_ylim(25, 1100)
ax3.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
ax3.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$", "$10$",
                     "$100$"], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("$\\lambda_2$ (능형 쪽 벌점)")
ax3.set_ylabel("모형오차의 중앙값")
ax3.set_title("③ 곱하기 한 번이 가르는 것", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/elastic_net/img/naive_elastic_net.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
