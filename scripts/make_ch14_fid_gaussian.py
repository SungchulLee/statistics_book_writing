r"""생성모형 평가지표 FID 안에 숨어 있는 정규성 가정을 수로 본다.

FID 는 두 특징 집합에 **가우스를 하나씩 맞추고** 그 둘 사이의 프레셰 거리를 잰다.

$$\mathrm{FID} = \lVert\mu_1-\mu_2\rVert^2
  + \operatorname{tr}\!\left(\Sigma_1+\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2}\right)$$

그러므로 평균과 공분산이 같으면 **분포가 아무리 달라도 $0$** 이다.

① 가우스와 이봉분포를 평균·공분산이 같게 만들면 FID 가 $0$ 근처다. 같은 분포에서
   뽑은 두 표본 사이의 FID 보다도 작다.
② 그런데 실제 특징은 가우스가 아니다 — ReLU 를 지난 값의 절반이 정확히 $0$ 이다.
③ 게다가 FID 는 표본 수에 편향되어 있어 $n$ 이 다르면 견줄 수 없다.

만드는 파일: docs/ch14/limitations/img/fid_gaussian.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from scipy import linalg, stats  # noqa: E402

RNG = np.random.default_rng(20261022)

D, N = 16, 20000
A_MIX = 0.9                       # 이봉분포의 봉우리 위치
NS = np.array([500, 1000, 2000, 5000, 10000, 20000])
NF, DIN, H, LAYERS = 5000, 64, 128, 3


def fid(X, Y):
    """두 표본에 가우스를 맞추고 프레셰 거리를 잰다."""
    m1, m2 = X.mean(0), Y.mean(0)
    S1, S2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    c, _ = linalg.sqrtm(S1 @ S2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(S1 + S2 - 2 * np.real(c)))


# === ① 평균과 공분산만 맞추면 ========================================
A = RNG.standard_normal((N, D))
s = np.sqrt(1 - A_MIX ** 2)
B = RNG.standard_normal((N, D))
B[:, 0] = A_MIX * RNG.choice([-1.0, 1.0], N) + s * RNG.standard_normal(N)
A2 = RNG.standard_normal((N, D))

f_ab, f_aa = fid(A, B), fid(A, A2)
ks = stats.ks_2samp(A[:, 0], B[:, 0])
print(f"차원 {D}, 표본 {N:,}")
print(f"  첫 좌표 평균 : A {A[:, 0].mean():+.4f}, B {B[:, 0].mean():+.4f}")
print(f"  첫 좌표 분산 : A {A[:, 0].var():.4f}, B {B[:, 0].var():.4f}")
print(f"\n① FID(가우스, 이봉)       = {f_ab:.5f}")
print(f"   FID(가우스, 가우스')     = {f_aa:.5f}   <- 같은 분포인데 더 크다")
print(f"   첫 좌표의 KS 통계량 {ks.statistic:.4f}, p = {ks.pvalue:.1e}")

# === ② 특징은 가우스인가 =============================================
h = RNG.standard_normal((NF, DIN))
for _ in range(LAYERS):
    W = RNG.standard_normal((h.shape[1], H)) / np.sqrt(h.shape[1])
    h = np.maximum(h @ W, 0.0)
zero = (h == 0).mean(0)
p_sw = np.array([stats.shapiro(h[:4000, j]).pvalue for j in range(h.shape[1])])
skew = np.array([stats.skew(h[:, j]) for j in range(h.shape[1])])
print(f"\n② ReLU 층 {LAYERS} 개를 지난 특징 {h.shape[1]} 차원")
print(f"   정확히 0 인 값의 비율 평균 {zero.mean():.3f} "
      f"(범위 {zero.min():.3f}–{zero.max():.3f})")
print(f"   샤피로–윌크 통과 {100 * (p_sw > 0.05).mean():.1f}%, "
      f"p < 1e-10 인 차원 {100 * (p_sw < 1e-10).mean():.0f}%")
print(f"   왜도 중앙값 {np.median(skew):+.3f}")

# === ③ 표본 수 편향 ==================================================
print(f"\n③ 같은 분포에서 뽑은 두 표본의 FID (0 이어야 한다)")
bias = []
for n in NS:
    bias.append(float(np.mean([fid(RNG.standard_normal((int(n), D)),
                                   RNG.standard_normal((int(n), D)))
                               for _ in range(5)])))
    print(f"   n = {n:>6} : {bias[-1]:.4f}")
bias = np.array(bias)
sl = np.polyfit(np.log(NS), np.log(bias), 1)[0]
print(f"   로그-로그 기울기 {sl:.2f} (1/n 이면 -1)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

bins = np.linspace(-4, 4, 70)
ax1.hist(A[:, 0], bins=bins, density=True, alpha=0.55, color="#1f77b4",
         label="가우스")
ax1.hist(B[:, 0], bins=bins, density=True, alpha=0.55, color="#d62728",
         label="이봉")
ax1.text(0.03, 0.97,
         f"평균 ${A[:, 0].mean():+.3f}$ 대 ${B[:, 0].mean():+.3f}$\n"
         f"분산 ${A[:, 0].var():.3f}$ 대 ${B[:, 0].var():.3f}$\n"
         f"$\\mathrm{{FID}} = {f_ab:.4f}$\n"
         f"같은 분포끼리는 ${f_aa:.4f}$",
         transform=ax1.transAxes, va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax1.set_xlabel("첫 좌표의 값")
ax1.set_ylabel("밀도")
ax1.set_title("① 평균과 공분산이 같으면 $0$ 이다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25)

ax2.hist(zero, bins=np.linspace(0, 1, 26), color="#2ca02c", alpha=0.75)
ax2.axvline(zero.mean(), color="#d62728", lw=2.2, ls="--")
ax2.text(zero.mean() + 0.02, ax2.get_ylim()[1] * 0.92,
         f"평균 ${zero.mean():.3f}$", fontsize=11, color="#d62728")
ax2.text(0.03, 0.70,
         f"샤피로–윌크 통과\n${100 * (p_sw > 0.05).mean():.1f}\\%$ "
         f"(차원 {h.shape[1]} 개)\n"
         f"$p < 10^{{-10}}$ 인 차원 ${100 * (p_sw < 1e-10).mean():.0f}\\%$\n"
         f"왜도 중앙값 ${np.median(skew):+.2f}$",
         transform=ax2.transAxes, va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax2.set_xlabel("그 차원에서 값이 정확히 $0$ 인 비율")
ax2.set_ylabel("차원 수")
ax2.set_title("② 맞추는 대상이 가우스가 아니다", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.loglog(NS, bias, "o-", color="#1f77b4", lw=2.8, ms=9)
ax3.plot(NS, bias[0] * (NS / NS[0]) ** -1.0, color="#999999", lw=1.6, ls=":",
         label="$1/n$ 기울기")
ax3.axhline(f_ab, color="#d62728", lw=2.0, ls="--",
            label=f"① 의 두 분포 사이 FID (${f_ab:.4f}$)")
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax3.set_yticks([0.01, 0.03, 0.1, 0.3])
ax3.set_yticklabels(["$0.01$", "$0.03$", "$0.1$", "$0.3$"])
ax3.minorticks_off()
ax3.set_xlabel("표본 수 $n$")
ax3.set_ylabel("같은 분포에서 뽑은 두 표본의 FID")
ax3.set_title("③ 표본 수가 다르면 견줄 수 없다", fontsize=12)
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch14/limitations/img/fid_gaussian.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
