r"""상관행렬을 보고 특징을 고르는 일이 왜 체계적으로 틀리는가.

흔한 전처리 두 가지를 수로 본다.

- 「$y$ 와의 상관이 작은 특징은 버린다」
- 「서로 상관이 높은 쌍에서는 하나만 남긴다」

① 억제변수 — $y$ 와의 상관이 **$0$** 인데 $R^2$ 의 대부분을 들고 있는 특징이 있다.
② 신호가 두 특징의 **차이**에 있으면 하나를 버리는 순간 사라진다. 손실은 상관이
   클수록 커지지 않고 **가운데**에서 가장 크다.
③ 무작위로 만든 문제들에서 $\lvert r\rvert$ 와 실제 기여의 순위상관이 $0.2$ 밖에 안 된다.

만드는 파일: docs/ch10/correlation/img/corr_feature_filter.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

N = 4000
SIG = 0.3
RHO1 = 0.9                       # ① 에서 두 특징의 상관
RHOS = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
PROBS, PFEAT, THR = 4000, 6, 0.1   # ③ 의 설정


def gauss2(rho, n=N):
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    return RNG.standard_normal((n, 2)) @ L.T


def r2(X, y):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - X @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# === ① 억제변수 =====================================================
X = gauss2(RHO1)
y = X[:, 0] - RHO1 * X[:, 1] + RNG.normal(0, SIG, N)
r_marg = [float(np.corrcoef(y, X[:, j])[0, 1]) for j in (0, 1)]
r2_one = [r2(X[:, [j]], y) for j in (0, 1)]
r2_both = r2(X, y)
print(f"① 두 특징의 상관 {RHO1}, 참 관계 y = x1 - {RHO1} x2")
print(f"   y 와의 상관 : x1 {r_marg[0]:+.4f},  x2 {r_marg[1]:+.4f}")
print(f"   R^2         : x1 만 {r2_one[0]:.4f}, x2 만 {r2_one[1]:.4f}, "
      f"둘 다 {r2_both:.4f}")
print(f"   -> 상관이 사실상 0 인 x2 를 버리면 {r2_both:.3f} 가 "
      f"{r2_one[0]:.3f} 로 내려간다")

# === ② 상관이 높은 쌍에서 하나를 버리면 ==============================
print(f"\n② 신호가 차이에 있을 때 (y = x1 - x2)")
print(f"{'rho':>6}{'둘 다':>10}{'하나만':>10}{'손실':>10}")
loss, both_l, one_l = [], [], []
for rho in RHOS:
    Z = gauss2(rho)
    yy = Z[:, 0] - Z[:, 1] + RNG.normal(0, SIG, N)
    b, o = r2(Z, yy), r2(Z[:, [0]], yy)
    both_l.append(b); one_l.append(o); loss.append(b - o)
    print(f"{rho:>6.2f}{b:>10.4f}{o:>10.4f}{loss[-1]:>10.4f}")
loss = np.array(loss)
i_worst = int(np.argmax(loss))
print(f"   손실이 가장 큰 자리: rho = {RHOS[i_worst]}, 손실 {loss[i_worst]:.4f}")

# === ③ 주변 상관이 기여를 얼마나 말해 주는가 =========================
rs, ds = [], []
for _ in range(PROBS):
    A = RNG.standard_normal((PFEAT, PFEAT)) * 0.6 + np.eye(PFEAT)
    Z = RNG.standard_normal((N // 2, PFEAT)) @ A.T
    Z = (Z - Z.mean(0)) / Z.std(0)
    b = RNG.standard_normal(PFEAT)
    yy = Z @ b + RNG.normal(0, 0.5, N // 2)
    yy = (yy - yy.mean()) / yy.std()
    full = r2(Z, yy)
    rs.append(np.abs([np.corrcoef(Z[:, j], yy)[0, 1] for j in range(PFEAT)]))
    ds.append([full - r2(np.delete(Z, j, axis=1), yy) for j in range(PFEAT)])
rs, ds = np.concatenate(rs), np.concatenate(ds)
low = rs < THR
rho_rank = float(stats.spearmanr(rs, ds).statistic)
print(f"\n③ 무작위 문제 {PROBS:,} 개 x 특징 {PFEAT} 개 = {len(rs):,} 개")
print(f"   |r| < {THR} 인 특징이 {100 * low.mean():.1f}%")
for t in (0.02, 0.05, 0.10):
    print(f"     그중 빼면 R^2 가 {t} 넘게 떨어지는 것 "
          f"{100 * (ds[low] > t).mean():.1f}%")
print(f"   그 특징들의 최대 기여 {ds[low].max():.4f}")
print(f"   |r| 와 기여의 순위상관 {rho_rank:.3f}")

# === 실제 자료에서는 어떤가 (정직하게 적기 위한 확인) ================
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer          # noqa: E402
from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.model_selection import cross_val_score      # noqa: E402
from sklearn.pipeline import make_pipeline               # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402

Xb, yb = load_breast_cancer(return_X_y=True)
rb = np.abs([np.corrcoef(Xb[:, j], yb)[0, 1] for j in range(Xb.shape[1])])
cand = np.where(rb >= THR)[0]              # 규칙 1
keep = []
for j in cand:                             # 규칙 2
    if all(abs(np.corrcoef(Xb[:, j], Xb[:, k])[0, 1]) < 0.9 for k in keep):
        keep.append(j)
keep = np.array(keep)
est = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
auc_all = cross_val_score(est(), Xb, yb, cv=5, scoring="roc_auc").mean()
auc_cut = cross_val_score(est(), Xb[:, keep], yb, cv=5, scoring="roc_auc").mean()
print(f"\n[실제 자료] 유방암, 특징 {Xb.shape[1]} 개 -> 두 규칙 뒤 {len(keep)} 개")
print(f"   교차검증 AUC {auc_all:.4f} -> {auc_cut:.4f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

vals = [r2_one[0], r2_one[1], r2_both]
names = [f"$x_1$ 만\n($r = {r_marg[0]:+.2f}$)",
         f"$x_2$ 만\n($r = {r_marg[1]:+.2f}$)", "둘 다"]
cols = ["#1f77b4", "#d62728", "#2ca02c"]
ax1.bar(np.arange(3), vals, 0.6, color=cols)
for i, v in enumerate(vals):
    ax1.text(i, v + 0.018, f"${v:.3f}$", ha="center", fontsize=12.5)
ax1.set_xticks(np.arange(3))
ax1.set_xticklabels(names, fontsize=10.5)
ax1.set_ylim(0, max(vals) * 1.26)
ax1.set_ylabel("$R^2$")
ax1.annotate("", xy=(0.08, r2_one[0] + 0.02), xytext=(1.92, r2_both - 0.012),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8, ls=":"))
ax1.text(1.0, r2_both * 0.70,
         f"$y$ 와 상관이 ${r_marg[1]:+.2f}$ 인 $x_2$ 를\n버리면 이만큼 잃는다",
         ha="center", fontsize=10.5, color="#d62728")
ax1.set_title("① 상관 $0$ 인 특징이 들고 있던 것", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(RHOS, both_l, "o-", color="#2ca02c", lw=2.6, ms=8, label="둘 다 쓸 때")
ax2.plot(RHOS, one_l, "s--", color="#d62728", lw=2.4, ms=7, label="하나만 쓸 때")
ax2.fill_between(RHOS, one_l, both_l, color="#d62728", alpha=0.15,
                 label="버려서 잃는 몫")
ax2.annotate(f"손실이 가장 큰 자리\n$\\varrho = {RHOS[i_worst]}$, "
             f"${loss[i_worst]:.2f}$",
             xy=(RHOS[i_worst], (both_l[i_worst] + one_l[i_worst]) / 2),
             xytext=(0.3, 0.42), fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xlabel("두 특징 사이의 상관 $\\varrho$")
ax2.set_ylabel("$R^2$")
ax2.set_ylim(0, 1.02)
ax2.set_title("② 가운데가 가장 위험하다", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

sub = RNG.choice(len(rs), 4000, replace=False)
ax3.scatter(rs[sub], ds[sub], s=9, alpha=0.3, color="#1f77b4",
            edgecolors="none")
ax3.axvline(THR, color="#d62728", lw=2.0, ls="--")
ax3.text(0.15, 0.93, f"이 선 왼쪽을 버린다\n(전체의 ${100 * low.mean():.0f}\\%$)",
         transform=ax3.transAxes, fontsize=10.5, color="#d62728", va="top")
ax3.text(0.97, 0.52,
         f"그중 ${100 * (ds[low] > 0.05).mean():.0f}\\%$ 는 빼면 $R^2$ 가\n"
         f"$0.05$ 넘게 떨어진다 (최대 ${ds[low].max():.2f}$)\n"
         f"$| r|$ 와 기여의 순위상관 ${rho_rank:.2f}$",
         transform=ax3.transAxes, ha="right", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax3.set_xlabel("$y$ 와의 주변 상관 $| r|$")
ax3.set_ylabel("그 특징을 빼면 잃는 $R^2$")
ax3.set_xlim(0, 0.9)
ax3.set_title("③ 기준이 재려는 것을 못 본다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch10/correlation/img/corr_feature_filter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
