r"""「평균이 답이다」가 정규분포를 고른다 — 가우스가 거꾸로 세운 논리.

오차의 밀도를 $f(x-\mu)$ 라 할 때 $\mu$ 의 최대가능도추정은
$\sum_i \psi(x_i - \hat\mu) = 0$ 으로 정해진다. 여기서 $\psi = -(\log f)'$ 다.
이 식의 해가 **어떤 표본에서도** $\bar x$ 가 되려면 $\psi$ 가 원점을 지나는
직선이어야 하고, 그것은 $f$ 가 정규라는 말과 같다.

① 다섯 밀도의 $\psi$ — 직선인 것은 정규 하나뿐이다.
② 그래서 최대가능도추정이 정규에서는 평균, 라플라스에서는 중앙값이 된다.
③ 오차가 정규가 아닌데 평균을 쓰면 얼마를 잃는가.

만드는 파일: docs/ch06/mle/img/gauss_mean_characterization.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import optimize   # noqa: E402

RNG = np.random.default_rng(0)
N_SMALL, R_CHECK = 9, 300
N_EFF, R_EFF = 20, 2000

# psi = -(log f)' — 최대가능도 방정식 sum psi(x_i - mu) = 0 의 가중함수
PSI = {
    "정규": lambda z: z,
    "라플라스": lambda z: np.sign(z),
    "로지스틱": lambda z: np.tanh(z / 2),
    "코시": lambda z: 2 * z / (1 + z ** 2),
    "$t(3)$": lambda z: 4 * z / (3 + z ** 2),
}
LL = {
    "정규": lambda z: -0.5 * z ** 2,
    "라플라스": lambda z: -np.abs(z),
    "로지스틱": lambda z: -z - 2 * np.log1p(np.exp(-z)),
    "코시": lambda z: -np.log1p(z ** 2),
    "$t(3)$": lambda z: -2 * np.log1p(z ** 2 / 3),
}
NAMES = list(PSI)


def mle_mu(x, ll):
    f = lambda m: -np.sum(ll(x - m))          # noqa: E731
    return float(optimize.minimize_scalar(
        f, bounds=(x.min() - 2, x.max() + 2), method="bounded",
        options={"xatol": 1e-11}).x)


# === ② 어느 통계량으로 가는가 ========================================
print(f"② 최대가능도추정이 무엇과 같아지는가 (n = {N_SMALL}, {R_CHECK} 번)")
print(f"{'오차분포':>10}{'|mu_hat - 평균| 최대':>21}{'|mu_hat - 중앙값| 최대':>23}")
DA, DM = {}, {}
for nm in NAMES:
    da = dm = 0.0
    for _ in range(R_CHECK):
        x = RNG.normal(0, 1, N_SMALL) * RNG.choice([1.0, 2.0]) + RNG.normal(0, 3)
        m = mle_mu(x, LL[nm])
        da = max(da, abs(m - x.mean()))
        dm = max(dm, abs(m - np.median(x)))
    DA[nm], DM[nm] = da, dm
    print(f"{nm:>10}{da:>21.2e}{dm:>23.2e}")
print("   정규는 평균과, 라플라스는 중앙값과 기계오차까지 같다. "
      "나머지 셋은 어느 쪽도 아니다.")

# === ③ 평균을 고집하면 =============================================
DRAW = {
    "정규": lambda m: RNG.normal(0, 1, (m, N_EFF)),
    "라플라스": lambda m: RNG.laplace(0, 1 / np.sqrt(2), (m, N_EFF)),
    "로지스틱": lambda m: RNG.logistic(0, np.sqrt(3) / np.pi, (m, N_EFF)),
    "$t(3)$": lambda m: RNG.standard_t(3, (m, N_EFF)) / np.sqrt(3),
}
LL_S = {                                    # 분산을 1 로 맞춘 꼴
    "정규": LL["정규"],
    "라플라스": lambda z: -np.abs(z) * np.sqrt(2),
    "로지스틱": lambda z: LL["로지스틱"](z * np.pi / np.sqrt(3)),
    "$t(3)$": lambda z: -2 * np.log1p(z ** 2),
}
print(f"\n③ 오차의 분산을 1 로 맞추고 n = {N_EFF}, {R_EFF} 번")
print(f"{'오차분포':>10}{'평균의 MSE':>12}{'MLE 의 MSE':>12}{'평균의 상대효율':>17}")
EFF = {}
for nm in DRAW:
    Xm = DRAW[nm](R_EFF)
    a = Xm.mean(axis=1)
    b = np.array([mle_mu(Xm[i], LL_S[nm]) for i in range(R_EFF)])
    ma, mb = float(np.mean(a ** 2)), float(np.mean(b ** 2))
    EFF[nm] = mb / ma
    print(f"{nm:>10}{ma:>12.4f}{mb:>12.4f}{EFF[nm]:>17.3f}")
print(f"   꼬리가 두꺼울수록 평균이 버리는 몫이 커진다 "
      f"($t(3)$ 에서 {100 * (1 - EFF['$t(3)$']):.0f}%).")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
COL = {"정규": "#2ca02c", "라플라스": "#d62728", "로지스틱": "#ff7f0e",
       "코시": "#9467bd", "$t(3)$": "#1f77b4"}

g = np.linspace(-4, 4, 600)
for nm in NAMES:
    ax1.plot(g, PSI[nm](g), color=COL[nm], lw=2.8 if nm == "정규" else 2.0,
             label=nm, zorder=3 if nm == "정규" else 2)
ax1.axhline(0, color="#888888", lw=1.0)
ax1.axvline(0, color="#888888", lw=1.0)
ax1.annotate("원점을 지나는 직선은\n정규 하나뿐이다", xy=(2.4, 2.4),
             xytext=(-3.7, 2.9), textcoords="data", fontsize=10.5,
             color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1))
ax1.set_xlabel("$z = x - \\mu$")
ax1.set_ylabel("$\\psi(z) = -(\\log f)'(z)$")
ax1.set_ylim(-4.2, 4.2)
ax1.set_title("① 최대가능도 방정식의 가중함수", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25)

xs = np.arange(len(NAMES))
w = 0.36
FLOOR = 1e-9
ax2.bar(xs - w / 2, [max(DA[n], FLOOR) for n in NAMES], w, color="#1f77b4",
        label="표본평균과의 차")
ax2.bar(xs + w / 2, [max(DM[n], FLOOR) for n in NAMES], w, color="#ff7f0e",
        label="표본중앙값과의 차")
ax2.set_yscale("log")
ax2.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0])
ax2.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$",
                     "$1$"])
ax2.minorticks_off()
ax2.set_xticks(xs)
ax2.set_xticklabels(NAMES, fontsize=10)
ax2.set_ylim(1e-9, 1e2)
# mathtext 에 \lvert 가 없다 — 세로막대를 그대로 쓴다
ax2.set_ylabel("$|\\hat\\mu - \\cdot|$ 의 최댓값")
ax2.set_title(f"② 정규만 평균을 고른다 ($n = {N_SMALL}$)", fontsize=12)
ax2.legend(loc="upper center", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

ns = list(DRAW)
ax3.bar(np.arange(len(ns)), [EFF[n] for n in ns], 0.55,
        color=[COL[n] for n in ns])
for k, n in enumerate(ns):
    ax3.text(k, EFF[n] + 0.02, f"${EFF[n]:.3f}$", ha="center", fontsize=11)
ax3.axhline(1.0, color="black", lw=1.4, ls="--")
ax3.set_xticks(range(len(ns)))
ax3.set_xticklabels(ns, fontsize=10.5)
ax3.set_ylabel("표본평균의 상대효율 (MLE 대비)")
ax3.set_ylim(0, 1.15)
ax3.set_title(f"③ 평균을 고집하면 ($n = {N_EFF}$)", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch06/mle/img/gauss_mean_characterization.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
