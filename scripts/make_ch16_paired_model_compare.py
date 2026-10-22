r"""같은 시험집합에서 두 모형을 견줄 때 짝을 버리면 어떻게 되는가.

모형 둘을 **같은** 시험표본에 걸면 표본마다 두 손실이 나온다. 짝지은 자료다.
그런데 "모형 A 의 점수 분포"와 "모형 B 의 점수 분포"를 독립 두 표본처럼
견주는 일이 흔하다. 그 대가를 잰다.

① 짝 안의 상관에 따른 검정력 — 대응 순열검정 대 독립 순열검정.
   상관이 높으면 독립 검정은 검정력이 $\alpha$ **아래로** 떨어진다.
② 실제 두 모형의 표본별 손실 상관은 얼마나 되는가.
③ 같은 결론에 이르려면 시험집합이 몇 배 필요한가.

만드는 파일: docs/ch16/paired_sample_nonparametric/img/paired_model_compare.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from scipy import stats                                   # noqa: E402
from sklearn.datasets import make_classification          # noqa: E402
from sklearn.linear_model import LogisticRegression       # noqa: E402

RNG = np.random.default_rng(1)
M, B, REPS, DELTA, ALPHA = 400, 999, 400, 0.02, 0.05
RHOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.998]


def p_paired(d, B, r):
    """대응 순열검정 — 짝마다 부호를 뒤집는다."""
    s = np.abs(d) * np.where(r.random((B, len(d))) < 0.5, -1, 1)
    return (np.sum(np.abs(s.mean(axis=1)) >= abs(d.mean())) + 1) / (B + 1)


def p_unpaired(a, b, B, r):
    """독립 두 표본 순열검정 — 두 무더기를 합쳐 다시 가른다."""
    z = np.r_[a, b]
    n = len(a)
    obs = abs(a.mean() - b.mean())
    zz = z[np.argsort(r.random((B, len(z))), axis=1)]
    st = np.abs(zz[:, :n].mean(axis=1) - zz[:, n:].mean(axis=1))
    return (np.sum(st >= obs) + 1) / (B + 1)


# === ① 검정력 ========================================================
print(f"① 시험표본 {M} 개, 참 차이 {DELTA}, {REPS} 번 (명목 {ALPHA})")
print(f"{'짝 안의 상관':>11}{'대응 순열':>11}{'독립 순열':>11}")
POW_P, POW_U = [], []
for rho in RHOS:
    pa = pb = 0
    for _ in range(REPS):
        a = RNG.normal(0, 1, M)
        b = rho * a + np.sqrt(1 - rho ** 2) * RNG.normal(0, 1, M) - DELTA
        pa += p_paired(a - b, B, RNG) < ALPHA
        pb += p_unpaired(a, b, B, RNG) < ALPHA
    POW_P.append(pa / REPS)
    POW_U.append(pb / REPS)
    print(f"{rho:>11.3f}{POW_P[-1]:>11.3f}{POW_U[-1]:>11.3f}")
print(f"   상관이 0 이면 둘이 같다. 상관이 커지면 대응 쪽만 올라가고,")
print(f"   독립 쪽은 {ALPHA} 아래로 내려간다 — 합쳐서 섞는 순간 귀무분포가")
print(f"   실제 차이의 흩어짐보다 넓어져 지나치게 보수적이 되기 때문이다.")

# === ② 실제 두 모형 ==================================================
X, y = make_classification(n_samples=4000, n_features=20, n_informative=6,
                           flip_y=0.12, random_state=0)
Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]


def fit(seed, C=1.0, frac=0.8):
    i = np.random.default_rng(seed).choice(len(ytr), int(frac * len(ytr)),
                                           replace=False)
    m = LogisticRegression(C=C, max_iter=2000).fit(Xtr[i], ytr[i])
    p = np.clip(m.predict_proba(Xte)[:, 1], 1e-9, 1 - 1e-9)
    loss = -(yte * np.log(p) + (1 - yte) * np.log(1 - p))
    return loss, (m.predict(Xte) == yte).astype(float)


LA, CA = fit(1)
LB, CB = fit(2, C=0.05)
RHO_REAL = float(np.corrcoef(LA, LB)[0, 1])
AGREE = float(np.mean(CA == CB))
print(f"\n② 같은 자료로 학습한 두 로지스틱 모형 (시험표본 {len(yte)} 개)")
print(f"   표본별 손실의 상관 {RHO_REAL:.4f}")
print(f"   정오가 일치하는 표본의 비율 {AGREE:.4f} "
      f"— 엇갈리는 표본은 {100 * (1 - AGREE):.1f}% 뿐이다")
print(f"   정확도 {CA.mean():.4f} 대 {CB.mean():.4f}")
d = LA - LB
print(f"   손실 차이의 대응 순열검정 p = {p_paired(d, 9999, RNG):.4f}, "
      f"독립 순열검정 p = {p_unpaired(LA, LB, 9999, RNG):.4f}")

# === ③ 필요한 시험집합 ===============================================
Z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)
print(f"\n③ 검정력 0.8 에 필요한 시험표본 수 (참 차이 {DELTA})")
print(f"{'상관':>7}{'대응':>10}{'독립':>12}{'배수':>9}")
MP, MU = [], []
for rho in RHOS:
    mp = 2 * (1 - rho) * (Z / DELTA) ** 2
    mu = 2 * (Z / DELTA) ** 2                 # 짝을 버리면 상관이 도움이 안 된다
    MP.append(mp)
    MU.append(mu)
    print(f"{rho:>7.3f}{mp:>10.0f}{mu:>12.0f}{mu / mp:>9.1f}")
j = RHOS.index(0.998)
print(f"   상관 {RHOS[j]} 에서 {MU[j] / MP[j]:.0f} 배다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C_P, C_U = "#2ca02c", "#d62728"

ax1.plot(RHOS, POW_P, "o-", color=C_P, lw=2.8, ms=7, label="대응 순열검정")
ax1.plot(RHOS, POW_U, "s-", color=C_U, lw=2.8, ms=7, label="독립 순열검정")
ax1.axhline(ALPHA, color="black", lw=1.4, ls="--")
ax1.text(0.02, ALPHA + 0.03, f"명목 ${ALPHA}$", fontsize=10.5)
ax1.axvline(RHO_REAL, color="#555555", lw=1.3, ls=":")
ax1.annotate(f"실제 두 모형의 상관\n${RHO_REAL:.3f}$ (②)",
             xy=(RHO_REAL, 0.55), xytext=(0.42, 0.62), textcoords="data",
             fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax1.set_xlabel("짝 안의 상관")
ax1.set_ylabel("참 차이를 잡아낼 확률")
ax1.set_ylim(-0.03, 1.05)
ax1.set_title(f"① 짝을 버리면 ($n = {M}$)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.plot(LA, LB, ".", color="#1f77b4", ms=3, alpha=0.5)
lim = [0, np.percentile(np.r_[LA, LB], 99.5)]
ax2.plot(lim, lim, "-", color="#555555", lw=1.4)
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.text(0.04, 0.96,
         f"표본별 손실의 상관 ${RHO_REAL:.4f}$\n"
         f"정오가 엇갈리는 표본 ${100 * (1 - AGREE):.1f}\\%$\n"
         f"정확도 ${CA.mean():.4f}$ 대 ${CB.mean():.4f}$",
         transform=ax2.transAxes, va="top", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax2.set_xlabel("모형 $A$ 의 표본별 손실")
ax2.set_ylabel("모형 $B$ 의 표본별 손실")
ax2.set_title("② 두 모형은 같은 표본에서 틀린다", fontsize=12)
ax2.grid(alpha=0.25)

ax3.semilogy(RHOS, MP, "o-", color=C_P, lw=2.8, ms=7, label="대응")
ax3.semilogy(RHOS, MU, "s--", color=C_U, lw=2.8, ms=7, label="독립")
ax3.axvline(RHO_REAL, color="#555555", lw=1.3, ls=":")
ax3.annotate(f"상관 ${RHO_REAL:.3f}$ 에서\n${MU[j] / MP[j]:.0f}$ 배",
             xy=(RHO_REAL, np.sqrt(MP[j] * MU[j])), xytext=(0.30, 2e3),
             textcoords="data", fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax3.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
ax3.set_yticklabels(["$10$", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"])
ax3.minorticks_off()
ax3.set_xlabel("짝 안의 상관")
ax3.set_ylabel("검정력 $0.8$ 에 필요한 시험표본 수")
ax3.set_title(f"③ 같은 결론을 내는 값 (차이 ${DELTA}$)", fontsize=12)
ax3.legend(loc="center left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch16/paired_sample_nonparametric/img/paired_model_compare.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
