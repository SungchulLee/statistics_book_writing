"""9.3절 두 분산의 F 검정 — 두 고장을 한 그림에.

들머리의 두 수를 그린다. 왼쪽은 눈에 보이는 차이를 못 잡는 쪽이다.
n1 = 15, n2 = 12 에서 참 표준편차비가 1.444(분산으로는 2.09 배)여도 기각할
확률이 낮다. 오른쪽은 반대쪽 고장이다 --- 두 분산이 **정확히 같은**
모집단인데 분포가 정규가 아니면 명목 5% 검정이 훨씬 자주 기각한다.

쪽의 f_two_sided_asymmetry 는 F 분포의 비대칭과 양측 기각역을,
fisher_z_vs_f 는 F 눈금과 z 눈금을, var_ratio_skew 는 등분산일 때 분산비의
표집분포를 그린다. 여기서는 검정이 실제로 내는 **기각률**만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/two_sample_tests/img/"
N1, N2, A, R = 15, 12, 0.05, 40_000
D1, D2 = N1 - 1, N2 - 1
LO, HI = stats.f.ppf(A / 2, D1, D2), stats.f.ppf(1 - A / 2, D1, D2)
rng = np.random.default_rng(1934)

# ── (좌) 참 비가 1 이 아닐 때 기각할 확률 ─────────────────────────
ratios = np.linspace(1.0, 3.0, 120)
power = stats.f.sf(HI / ratios ** 2, D1, D2) + \
    stats.f.cdf(LO / ratios ** 2, D1, D2)
R0 = 1.3 / 0.9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax1.plot(ratios, 100 * power, color="tab:purple", lw=2.4)
ax1.axhline(100 * A, color="0.45", ls="--", lw=1.5)
ax1.text(2.98, 100 * A + 1.5, "명목 5%", ha="right", fontsize=10.5,
         color="0.45")
pw = stats.f.sf(HI / R0 ** 2, D1, D2) + stats.f.cdf(LO / R0 ** 2, D1, D2)
ax1.plot([R0], [100 * pw], "o", ms=10, color="tab:red")
ax1.annotate(f"표준편차 $1.3$ 대 $0.9$\n(비 {R0:.3f}, 분산으로 "
             f"{R0**2:.2f} 배)\n기각할 확률 {100*pw:.1f}%",
             xy=(R0, 100 * pw), xytext=(1.62, 20), fontsize=11,
             color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
for r in (2.0, 2.5):
    p_ = stats.f.sf(HI / r ** 2, D1, D2) + stats.f.cdf(LO / r ** 2, D1, D2)
    ax1.plot([r], [100 * p_], "o", ms=7, color="0.35")
    ax1.text(r, 100 * p_ + 4, f"{100*p_:.0f}%", ha="center", fontsize=10.5,
             color="0.3")
ax1.set_xlim(1.0, 3.0)
ax1.set_ylim(0, 100)
ax1.set_xlabel("참 표준편차의 비", fontsize=11)
ax1.set_ylabel("기각할 확률 (%)", fontsize=11)
ax1.set_title(f"고장 하나 — 보이는 차이를 못 잡는다 ($n_1$ = {N1}, "
              f"$n_2$ = {N2})", fontsize=12)
ax1.grid(alpha=0.25)

# ── (우) 등분산인데 분포를 바꾸면 ─────────────────────────────────
# 쪽의 모의와 같은 설정으로 맞춘다 --- n1 = n2 = 20
NB = 20
LO_B = stats.f.ppf(A / 2, NB - 1, NB - 1)
HI_B = stats.f.ppf(1 - A / 2, NB - 1, NB - 1)


def size(draw):
    a = draw((R, NB)); b = draw((R, NB))
    f = a.var(axis=1, ddof=1) / b.var(axis=1, ddof=1)
    return float(((f < LO_B) | (f > HI_B)).mean())


DISTS = [("정규", lambda s: rng.standard_normal(s)),
         ("균등", lambda s: rng.uniform(-1, 1, s)),
         ("$t(5)$", lambda s: rng.standard_t(5, s)),
         ("지수", lambda s: rng.exponential(1, s)),
         ("로그정규", lambda s: rng.lognormal(0, 1, s))]
sizes = [(n, size(f)) for n, f in DISTS]
names = [n for n, _ in sizes]
vals = [100 * v for _, v in sizes]
y = np.arange(len(names))[::-1]
ax2.barh(y, vals, height=0.56,
         color=["0.6", "0.6", "tab:orange", "tab:orange", "tab:red"],
         alpha=0.88)
for yy, v in zip(y, vals):
    ax2.text(v + 1.0, yy, f"{v:.2f}%", va="center", fontsize=12,
             fontweight="bold", color="0.2")
ax2.axvline(100 * A, color="0.2", ls="--", lw=1.8)
ax2.text(100 * A + 1.0, len(names) - 1.42, "약속한 5%", ha="left",
         fontsize=11, color="0.2", fontweight="bold")
ax2.set_yticks(y, names, fontsize=12)
ax2.set_xlim(0, 62)
ax2.set_xlabel("실제로 기각한 비율 (%)", fontsize=11)
ax2.set_title(f"고장 둘 — 두 분산이 정확히 같은데도 "
              f"($n_1 = n_2 = {NB}$)", fontsize=12)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("가정을 지켜 주어야 할 검정이 자기 가정에는 더 약하다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "f_test_two_failures.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  임계값 {LO:.4f} ~ {HI:.4f} (자유도 {D1}, {D2})")
print(f"  표준편차비 {R0:.4f} 에서 기각할 확률 {pw:.4f}")
for n, v in sizes:
    print(f"  {n}: 등분산인데 기각한 비율 {v:.4f}")
