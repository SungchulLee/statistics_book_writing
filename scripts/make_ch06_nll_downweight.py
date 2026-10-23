r"""MSE 는 분산이 일정하다고 믿는 정규 MLE 다 — 그 믿음을 거두면.

이 쪽은 mu 와 sigma^2 을 함께 최대가능도로 풀고, 절 3 에서 그것이 최소제곱과 만나는
자리를 짚는다. 그 만남에는 조건이 하나 붙어 있다 — **sigma^2 이 모든 관측에서 같다**는
것이다. 조건을 거두면 로그가능도의 기울기에 1/sigma^2 이 붙는다.

    d(-logL)/d mu_hat = (mu_hat - y) / sigma_hat^2

곧 **시끄러운 곳의 자료는 덜, 조용한 곳의 자료는 더** 끌어당긴다. 가중최소제곱이다.

① 같은 바닥함수, 같은 자료. 분산을 함께 추정한 쪽과 MSE 로 맞춘 쪽.
② 가능도가 실제로 거는 가중 1/sigma_hat^2.
③ 분산을 함께 추정하면 잡음 구조 자체가 나온다.

만드는 파일: docs/ch06/mle/img/nll_downweight.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
# 한글 글꼴에는 mathtext 의 U+2212 가 없다
plt.rcParams["mathtext.fontset"] = "dejavusans"

RNG = np.random.default_rng(0)
N, K = 3000, 14
STEPS, LR = 4000, 0.05
SIG_LO, SIG_HI = 0.10, 0.70     # 왼쪽 절반은 조용하고 오른쪽 절반은 시끄럽다


def mu_true(x):
    return np.sin(2 * np.pi * x)


def sig_true(x):
    return np.where(x < 0.5, SIG_LO, SIG_HI)


x = RNG.uniform(0, 1, N)
y = mu_true(x) + RNG.normal(0, sig_true(x))

# 공통 바닥함수 (RBF) — 두 손실이 똑같은 표현력을 쓰게 한다
C = np.linspace(0, 1, K)
H = 1.0 / (K - 1)
PHI = np.exp(-0.5 * ((x[:, None] - C) / H) ** 2)
PHI = np.hstack([PHI, np.ones((N, 1))])
XG = np.linspace(0, 1, 400)
PHI_G = np.exp(-0.5 * ((XG[:, None] - C) / H) ** 2)
PHI_G = np.hstack([PHI_G, np.ones((len(XG), 1))])
HI = x >= 0.5                    # 시끄러운 구간


def fit(beta, track=False):
    """beta = 0 이면 보통 NLL, beta = 1 이면 평균 쪽 가중이 사라진다."""
    a = np.zeros(PHI.shape[1])          # mu 의 계수
    b = np.zeros(PHI.shape[1])          # log sigma^2 의 계수
    hist = []
    for t in range(STEPS):
        mu = PHI @ a
        lv = PHI @ b
        var = np.exp(np.clip(lv, -12, 6))
        w = var ** beta                 # beta-NLL 의 가중 (상수로 취급)
        r = mu - y
        ga = PHI.T @ (w * r / var) / N
        gb = PHI.T @ (w * 0.5 * (1 - r ** 2 / var)) / N
        a -= LR * ga
        b -= LR * gb
        if track and (t % 20 == 0 or t == STEPS - 1):
            hist.append(np.sqrt(np.mean((PHI[HI] @ a - mu_true(x[HI])) ** 2)))
    return a, b, np.array(hist)


a0, b0, h0 = fit(0.0, track=True)       # 보통 NLL
a5, b5, h5 = fit(0.5, track=True)       # beta = 0.5
a1, b1, h1 = fit(1.0, track=True)       # beta = 1 (평균은 MSE 와 같아진다)

print(f"n = {N}, 바닥함수 {K} 개, 걸음 {STEPS}, "
      f"sigma = {SIG_LO} (왼쪽) / {SIG_HI} (오른쪽)")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n① 참 평균과의 제곱근평균오차")
print("   손실          조용한 구간   시끄러운 구간")
for nm, aa in (("NLL (beta=0)", a0), ("beta-NLL (0.5)", a5),
               ("beta=1 (MSE)", a1)):
    lo = np.sqrt(np.mean((PHI[~HI] @ aa - mu_true(x[~HI])) ** 2))
    hi = np.sqrt(np.mean((PHI[HI] @ aa - mu_true(x[HI])) ** 2))
    print(f"   {nm:<14} {lo:>11.4f}   {hi:>13.4f}")
rmse_hi0 = np.sqrt(np.mean((PHI[HI] @ a0 - mu_true(x[HI])) ** 2))
rmse_hi1 = np.sqrt(np.mean((PHI[HI] @ a1 - mu_true(x[HI])) ** 2))
print(f"   시끄러운 구간에서 NLL 이 beta=1 보다 {rmse_hi0 / rmse_hi1:.2f} 배 "
      f"더 틀린다.")

print("\n② 평균을 끌어당기는 기울기에 걸리는 가중 1/sigma_hat^2")
v0 = np.exp(PHI @ b0)
print(f"   조용한 구간 평균 1/sigma^2   = {np.mean(1 / v0[~HI]):>8.2f}")
print(f"   시끄러운 구간 평균 1/sigma^2 = {np.mean(1 / v0[HI]):>8.2f}")
print(f"   비 = {np.mean(1 / v0[~HI]) / np.mean(1 / v0[HI]):.1f} 배. "
      f"참값으로는 ({SIG_HI}/{SIG_LO})^2 = {(SIG_HI / SIG_LO) ** 2:.0f} 배다.")

print("\n③ 가능도가 찾아낸 잡음 구조")
sd_hat = np.sqrt(np.exp(PHI @ b0))
print("   구간          참 sigma   추정 sigma")
print(f"   조용한 쪽      {SIG_LO:>8.2f}   {np.mean(sd_hat[~HI]):>10.3f}")
print(f"   시끄러운 쪽    {SIG_HI:>8.2f}   {np.mean(sd_hat[HI]):>10.3f}")
print("   MSE 로 맞추면 이 구조가 아예 나오지 않는다 — 분산을 상수로 두었으므로")
print("   어디가 믿을 만한 예측인지 말해 줄 수가 없다.")

print("\n[덧] mu 와 sigma^2 을 **동시에** 경사하강으로 푸는 일 자체가 까다롭다.")
print("     잡음 비를 바꿔 가며 여섯 번 돌려 보니 그 가운데 둘이 발산했다.")
print("     닫힌 꼴에서는 없던 어려움이고, 뒤에 적은 논문이 다루는 것이 이 자리다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_N, C_B, C_T = "#d62728", "#1f77b4", "#2ca02c"

sub = RNG.choice(N, 700, replace=False)
ax1.plot(x[sub], y[sub], ".", ms=3, color="#cccccc", label="자료")
ax1.plot(XG, mu_true(XG), lw=3.0, color="#333333", ls="--", label="참 평균")
ax1.plot(XG, PHI_G @ a0, lw=2.8, color=C_N, label="MLE (분산도 함께)")
ax1.plot(XG, PHI_G @ a1, lw=2.8, color=C_B, label="MSE (분산을 상수로)")
ax1.axvline(0.5, color="#888888", lw=1.4, ls=":")
ax1.text(0.52, 2.15, "여기서부터 시끄럽다", fontsize=10, color="#555555")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_ylim(-2.6, 2.6)
ax1.set_title("① 같은 자료, 같은 바닥함수", fontsize=12)
ax1.legend(loc="lower left", fontsize=9.5)
ax1.grid(alpha=0.25)

ax2.semilogy(XG, 1 / np.exp(PHI_G @ b0), lw=2.8, color=C_N,
             label="가능도가 거는 가중 $1/\\hat\\sigma^2$")
ax2.axvline(0.5, color="#888888", lw=1.4, ls=":")
ax2.set_xlabel("$x$")
ax2.set_ylabel("기울기에 걸리는 가중")
ax2.set_title("② 자료 한 점의 무게가 자리마다 다르다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.plot(XG, sig_true(XG), lw=3.0, color="#333333", ls="--", label="참 $\\sigma(x)$")
ax3.plot(XG, np.sqrt(np.exp(PHI_G @ b0)), lw=2.8, color=C_N,
         label="MLE 가 찾아낸 $\\hat\\sigma(x)$")
ax3.axhline(np.sqrt(np.mean((PHI @ a1 - y) ** 2)), lw=2.6, color=C_B,
            ls="-.", label="MSE 가 쓰는 상수 $\\sigma$")
ax3.axvline(0.5, color="#888888", lw=1.4, ls=":")
ax3.set_xlabel("$x$")
ax3.set_ylabel("$\\sigma$")
ax3.set_ylim(0, 0.95)
ax3.set_title("③ 분산을 모형에 넣으면 잡음 구조가 나온다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/mle/img/nll_downweight.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
