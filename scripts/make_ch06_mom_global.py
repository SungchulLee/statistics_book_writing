r"""어디서 출발하느냐가 답을 바꾼다 — 적률법이 MLE 의 출발값으로 쓰이는 까닭.

이 쪽은 「적률법 추정값이 반복적인 MLE 알고리즘의 출발값으로 자주 쓰인다」고 적는다.
그 한 줄이 무엇을 막아 주는지를 잰다.

성분마다 분산이 자유로운 정규혼합에서 가능도는 비볼록이고, EM 이 닿는 곳은 출발점이
정한다. 분산을 작게 주고 출발하면 한 성분이 좁은 덩어리에 붙어 **더 낮은** 로그가능도
에서 멈춘다. 적률법에는 넣을 출발점이 없으므로 이 갈림 자체가 없다.

① 같은 자료에 출발만 달리한 두 EM 의 로그가능도 자취.
② 출발 표준편차를 키우면 나쁜 쪽으로 가는 비율이 어떻게 줄어드는가.
③ 가능도가 유계가 되도록 분산을 묶으면, 본문 말대로 EM 이 더 정확하다.

덧붙임: 분산이 자유로우면 가능도가 위로 유계가 아니라는 것은 알려진 사실이지만,
아래 모의에서 EM 이 닿은 곳은 그 발산점이 아니라 분산이 작은 국소최적이다.

만드는 파일: docs/ch06/estimation_methods/img/mom_global.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
W3, MU3, SD3 = [1 / 3] * 3, [-3.0, 0.0, 3.0], [1.0, 1.0, 1.0]
FLOOR = 1e-12          # 분산 바닥을 사실상 치운다
COLLAPSE = 0.05        # 표준편차가 이보다 작아지면 무너진 것으로 본다


def draw(n):
    c = RNG.choice(3, n, p=W3)
    return RNG.normal(np.array(MU3)[c], np.array(SD3)[c])


def loglik(x, w, mu, sd):
    c = np.array([wi * stats.norm.pdf(x, m, s)
                  for wi, m, s in zip(w, mu, sd)])
    return float(np.log(np.maximum(c.sum(0), 1e-300)).sum())


def em(x, mu0, sd0, iters=400, trace=False):
    """분산 바닥 없는 EM. 닿은 로그가능도와 가장 작은 표준편차를 돌려준다."""
    mu = np.array(mu0, float)
    sd = np.array(sd0, float)
    w = np.full(3, 1 / 3)
    hist = []
    for _ in range(iters):
        r = np.array([wi * stats.norm.pdf(x, m, s)
                      for wi, m, s in zip(w, mu, sd)])
        tot = np.maximum(r.sum(0), 1e-300)
        r = r / tot
        nk = r.sum(1) + 1e-12
        w = nk / len(x)
        mu = (r * x).sum(1) / nk
        sd = np.sqrt(np.maximum((r * (x - mu[:, None]) ** 2).sum(1) / nk, FLOOR))
        if trace:
            hist.append(loglik(x, w, mu, sd))
        if sd.min() < 1e-8:
            break
    if trace:
        return loglik(x, w, mu, sd), float(sd.min()), hist
    return loglik(x, w, mu, sd), float(sd.min())


# === ① 그 자리로 가는 길은 좁지만 분명히 있다 ======================
N1 = 300
x1 = draw(N1)
ll_true = loglik(x1, W3, MU3, SD3)
# 성분 하나를 관측값 하나에 붙이고 표준편차를 작게 주고 출발한다
mu_bad = [x1[0], float(np.median(x1)), float(x1.max())]
_, s_bad, H_BAD = em(x1, mu_bad, [0.01, x1.std(), x1.std()], trace=True)
# 견줄 멀쩡한 출발
_, s_ok, H_OK = em(x1, RNG.choice(x1, 3, replace=False),
                   [x1.std()] * 3, trace=True)
print(f"① n = {N1}. 성분 하나를 관측값에 붙이고 표준편차 0.01 로 출발하면")
print(f"   가장 작은 표준편차가 {s_bad:.3e} 까지 내려가 거기서 멈춘다.")
print(f"   그 해의 로그가능도   = {H_BAD[-1]:>9.2f}")
print(f"   예사로운 출발의 해   = {H_OK[-1]:>9.2f}")
print(f"   참값에서의 로그가능도 = {ll_true:>9.2f}")
print(f"   차이가 {H_OK[-1] - H_BAD[-1]:.1f} 다. **같은 자료, 같은 알고리즘인데 "
      f"출발점만으로 갈린다.**")

# 길의 폭 — 출발 표준편차를 바꿔 가며
SD0 = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PER0 = 120
print(f"\n   출발 표준편차를 바꿔 가며 (초기값 {PER0} 개씩)")
print("   출발 sd   무너진 비율")
NARROW = []
for s0 in SD0:
    cnt = 0
    for _ in range(PER0):
        mu0 = RNG.choice(x1, 3, replace=False)
        _, sm = em(x1, mu0, [s0, x1.std(), x1.std()])
        cnt += sm < COLLAPSE
    NARROW.append(cnt / PER0)
    print(f"   {s0:>8.3f}   {NARROW[-1]:>11.3f}")
print("   표준편차를 넉넉히 주고 출발하면 거의 가지 않는다.")

# === ② 적률법에는 이 고장이 없다 =====================================
# 가중 1/2, 등분산 sigma 를 안다고 두면 평균 둘이 이차방정식의 두 근이다.
#   m1 = (mu1+mu2)/2,  m2 - sigma^2 = (mu1^2+mu2^2)/2
#   (mu1-mu2)^2 = 2(mu1^2+mu2^2) - (mu1+mu2)^2 = 4*half - 4*m1^2
MU_T, SD_T = (-2.0, 2.0), 1.0
N3, DATASETS = 400, 600


def mom_two(x, s):
    m1 = x.mean()
    half = (x ** 2).mean() - s ** 2
    d = np.sqrt(max(4 * half - 4 * m1 ** 2, 0.0))
    return m1 - d / 2, m1 + d / 2


print(f"\n③ 성분 둘·등분산 (참 평균 {MU_T}), 자료 {DATASETS} 벌")
em_err, mom_err = [], []
for _ in range(DATASETS):
    c = RNG.integers(0, 2, N3)
    x = RNG.normal(np.array(MU_T)[c], SD_T)
    a, b = mom_two(x, SD_T)
    mom_err.append(max(abs(a - MU_T[0]), abs(b - MU_T[1])))
    mu = np.sort(RNG.choice(x, 2, replace=False))
    w = np.array([0.5, 0.5])
    for _ in range(300):
        rr = np.array([wi * stats.norm.pdf(x, m, SD_T)
                       for wi, m in zip(w, mu)])
        rr = rr / np.maximum(rr.sum(0), 1e-300)
        nk = rr.sum(1) + 1e-12
        w = nk / N3
        mu = (rr * x).sum(1) / nk
    lo, hi = sorted(mu)
    em_err.append(max(abs(lo - MU_T[0]), abs(hi - MU_T[1])))
em_err, mom_err = np.array(em_err), np.array(mom_err)
print("            중앙 오차   90 분위")
print(f"   적률법    {np.median(mom_err):>8.4f} {np.percentile(mom_err, 90):>9.4f}")
print(f"   EM        {np.median(em_err):>8.4f} {np.percentile(em_err, 90):>9.4f}")
print(f"   분산을 묶어 두면 가능도가 유계가 되고, 그때는 EM 이 "
      f"{np.median(mom_err) / np.median(em_err):.2f} 배 정확하다.")
print("   **이 쪽 본문의 말이 맞는 자리가 바로 거기다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_M, C_E, C_G = "#1f77b4", "#d62728", "#2ca02c"

ax1.plot(np.arange(1, len(H_BAD) + 1), H_BAD, lw=2.8, color=C_E,
         label="성분 하나를 관측값에 붙이고 출발")
ax1.plot(np.arange(1, len(H_OK) + 1), H_OK, lw=2.8, color=C_G,
         label="예사로운 출발")
ax1.axhline(ll_true, color="#333333", lw=2.0, ls="--")
# 축 밖에 두면 bbox_inches="tight" 가 그림을 찌그러뜨린다 — 축 좌표로 붙인다
# 점선 바로 위에 붙인다. ylim 밖에 두면 bbox_inches="tight" 가 그림을 찌그러뜨린다
ax1.text(len(H_OK) * 0.62, ll_true + 0.6, "참값에서의 로그가능도",
         fontsize=10.5, color="#333333")
ax1.set_xlabel("EM 되풀이 횟수")
ax1.set_ylabel("로그가능도")
ax1.set_title("① 같은 자료, 다른 출발 → 다른 답", fontsize=12)
ax1.legend(loc="center right", fontsize=10)
ax1.grid(alpha=0.25)

ax2.semilogx(SD0, NARROW, "o-", color=C_E, lw=2.8, ms=11)
ax2.set_xlabel("출발할 때 준 표준편차")
ax2.set_ylabel("무너진 초기값의 비율")
ax2.set_xticks([0.01, 0.1, 1.0])
ax2.set_xticklabels(["$0.01$", "$0.1$", "$1$"])
ax2.minorticks_off()
ax2.set_ylim(-0.03, 1.03)
ax2.set_title("② 작게 주고 출발할수록 나쁜 쪽으로", fontsize=12)
ax2.grid(alpha=0.25)

bins = np.linspace(0, 0.6, 45)
ax3.hist(mom_err, bins=bins, color=C_M, alpha=0.75, label="적률법 (닫힌 형태)")
ax3.hist(em_err, bins=bins, color=C_E, alpha=0.60, label="EM (분산을 묶음)")
ax3.set_xlabel("평균 추정의 최대 오차")
ax3.set_ylabel("자료 벌 수")
ax3.set_title("③ 가능도가 유계인 자리에선 본문 말이 맞는다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/estimation_methods/img/mom_global.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
