r"""교환가능성만으로 **구간**을 주는 길 — 분할 등각예측.

순열검정의 보장은 「검정통계량의 순위가 귀무가설 아래에서 균등하다」는 한 줄에서 온다.
등각예측은 **같은 한 줄**로 예측구간을 만든다. 보정표본의 비적합 점수들 사이에서
새 점의 점수가 차지하는 순위가 균등하므로, 그 순위가 위에서 $\alpha$ 안에 들 확률이
정확히 $\alpha$ 이하다.

① 같은 적합, 두 가지 $90\%$ 예측구간 — 정규 가정과 등각.
② 보정표본 크기에 따른 포함률 — 등각은 $n$ 이 작아도 보장을 지킨다.
③ 새 점의 순위 분포 — 이 균등성이 보장의 전부다.

만드는 파일: docs/ch17/comparison/img/conformal.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
ALPHA = 0.10
N_TRAIN, N_TEST = 300, 4000
NCALS = (20, 50, 100, 300, 1000)
REPS = 600


def truth(x):
    return 1.0 + 0.8 * x


def errors(n, rng):
    """$90\\%$ 는 조용하고 $10\\%$ 는 크게 튄다 — 정규가 아니다.
    어느 성분에서 나왔는지도 함께 돌려준다 (조건부 포함률을 보려고)."""
    big = rng.random(n) < 0.10
    e = np.where(big, 3.0 * rng.standard_normal(n), 0.3 * rng.standard_normal(n))
    return e, big


def sample(n, rng, with_tag=False):
    x = rng.uniform(-3, 3, n)
    e, big = errors(n, rng)
    return (x, truth(x) + e, big) if with_tag else (x, truth(x) + e)


def fit(x, y):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = r @ r / (len(x) - 2)
    return b, np.sqrt(s2)


def predict(b, x):
    return b[0] + b[1] * x


def conformal_q(scores, alpha=ALPHA):
    """분할 등각의 유한표본 보정 — n 개 점수의
    ceil((n+1)(1-alpha)) 번째로 작은 값을 쓴다."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf                      # n 이 너무 작으면 보장이 자명해진다
    return np.sort(scores)[k - 1]


# === ① 한 판 =======================================================
xtr, ytr = sample(N_TRAIN, RNG)
b, sd = fit(xtr, ytr)
xcal, ycal = sample(300, RNG)
scal = np.abs(ycal - predict(b, xcal))
q_conf = conformal_q(scal)
q_norm = stats.norm.isf(ALPHA / 2) * sd

# === ②③ 여러 판 ====================================================
cov_conf = {n: [] for n in NCALS}
cov_norm = []
wid_conf = {n: [] for n in NCALS}
ranks = []

for _ in range(REPS):
    xa, ya = sample(N_TRAIN, RNG)
    bb, ss = fit(xa, ya)
    xt, yt = sample(N_TEST, RNG)
    st = np.abs(yt - predict(bb, xt))
    cov_norm.append(np.mean(st <= stats.norm.isf(ALPHA / 2) * ss))
    for n in NCALS:
        xc, yc = sample(n, RNG)
        sc = np.abs(yc - predict(bb, xc))
        q = conformal_q(sc)
        cov_conf[n].append(np.mean(st <= q))
        wid_conf[n].append(2 * q)

cov_norm = np.array(cov_norm)

# --- ③ 순위 분포 -----------------------------------------------------
# 적합을 하나 고정해 두면 보정점 100 개와 새 점 1 개의 점수 101 개가
# 교환가능하므로, 새 점의 순위는 1..101 에 균등해야 한다. 그것만 센다.
M_RANK, NC = 200_000, 100
xr, yr = sample(N_TRAIN, RNG)
br, _ = fit(xr, yr)
xs_, ys_ = sample(M_RANK * (NC + 1), RNG)
S = np.abs(ys_ - predict(br, xs_)).reshape(M_RANK, NC + 1)
ranks = 1 + np.sum(S[:, :NC] < S[:, [NC]], axis=1)

# 조건부 포함률 — 작은 오차와 큰 오차를 따로 보면
xa, ya = sample(N_TRAIN, RNG)
bb, ss = fit(xa, ya)
xc, yc = sample(1000, RNG)
q = conformal_q(np.abs(yc - predict(bb, xc)))
xt, yt, is_big = sample(200_000, RNG, with_tag=True)
inside = np.abs(yt - predict(bb, xt)) <= q
cov_small, cov_big = inside[~is_big].mean(), inside[is_big].mean()
cov_marg = inside.mean()

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

t = np.linspace(-3, 3, 200)
ax1.scatter(xtr, ytr, s=11, color="#bdbdbd", alpha=0.75, edgecolors="none")
ax1.plot(t, predict(b, t), color="black", lw=1.8, label="적합한 직선")
ax1.fill_between(t, predict(b, t) - q_norm, predict(b, t) + q_norm,
                 color="#d62728", alpha=0.14, lw=0)
ax1.plot(t, predict(b, t) + q_norm, color="#d62728", lw=2.0,
         label=f"정규 가정 ($\\pm {q_norm:.2f}$)")
ax1.plot(t, predict(b, t) - q_norm, color="#d62728", lw=2.0)
ax1.plot(t, predict(b, t) + q_conf, color="#1f77b4", lw=2.0,
         label=f"등각 ($\\pm {q_conf:.2f}$)")
ax1.plot(t, predict(b, t) - q_conf, color="#1f77b4", lw=2.0)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_title(f"① 같은 적합, 두 가지 ${int((1 - ALPHA) * 100)}\\%$ 예측구간",
              fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25)

pos = np.arange(len(NCALS))
m_conf = np.array([np.mean(cov_conf[n]) for n in NCALS]) * 100
lo = np.array([np.quantile(cov_conf[n], 0.05) for n in NCALS]) * 100
hi = np.array([np.quantile(cov_conf[n], 0.95) for n in NCALS]) * 100
ax2.errorbar(pos, m_conf, yerr=[m_conf - lo, hi - m_conf], fmt="o-",
             color="#1f77b4", lw=2.2, ms=6, capsize=5, label="등각")
ax2.axhline(cov_norm.mean() * 100, color="#d62728", lw=2.2, ls="-",
            label="정규 가정")
ax2.axhline((1 - ALPHA) * 100, color="black", lw=1.5, ls="--")
ax2.text(len(NCALS) - 0.55, (1 - ALPHA) * 100 + 0.5,
         f"명목 ${int((1 - ALPHA) * 100)}\\%$", ha="right", fontsize=10)
for i, n in enumerate(NCALS):
    ax2.text(i, lo[i] - 1.2, f"폭 ${np.mean(wid_conf[n]):.2f}$", ha="center",
             fontsize=9, color="#1f77b4")
ax2.text(len(NCALS) - 0.55, cov_norm.mean() * 100 + 0.5,
         f"폭 ${2 * q_norm:.2f}$", ha="right", fontsize=9, color="#d62728")
ax2.set_xticks(pos)
ax2.set_xticklabels([f"${n}$" for n in NCALS])
ax2.set_xlabel("보정표본 크기 $n$")
ax2.set_ylabel("실제 포함률 ($\\%$)")
ax2.set_title("② 등각은 어느 $n$ 에서도 보장을 지킨다", fontsize=12)
ax2.legend(loc="lower right", fontsize=9.5)
ax2.set_ylim(min(lo) - 3.0, max(hi) + 1.2)
ax2.grid(alpha=0.25, axis="y")

nb = NC + 1
ax3.hist(ranks, bins=np.arange(0.5, nb + 1.5), density=True, color="#9e9e9e",
         edgecolor="none")
ax3.axhline(1 / nb, color="#d62728", lw=2.2, ls="--",
            label=f"균등분포 $1/{nb}$")
k = int(np.ceil((NC + 1) * (1 - ALPHA)))
# 보정점수 k 개보다 큰 순위, 곧 k+1 부터가 구간 밖이다
ax3.axvline(k + 0.5, color="#1f77b4", lw=2.0)
ax3.set_ylim(0, 1 / nb * 1.7)
ax3.text(k - 3, 1 / nb * 1.58,
         f"여기 오른쪽이 구간 밖\n${nb - k}/{nb} = "
         f"{(nb - k) / nb:.3f}$ — 명목 ${ALPHA}$ 이하", fontsize=9.5,
         color="#1f77b4", va="top", ha="right")
ax3.set_xlabel(f"보정점수 ${NC}$ 개 사이에서 새 점이 차지한 순위")
ax3.set_ylabel("밀도")
ax3.set_title("③ 보장의 전부는 이 균등성이다", fontsize=12)
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch17/comparison/img/conformal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① 한 판 — 정규 반폭 {q_norm:.3f}, 등각 반폭 {q_conf:.3f} "
      f"({q_norm / q_conf:.2f} 배)")
print(f"   적합한 잔차 표준편차 {sd:.3f}")

print(f"\n② 포함률 (명목 {(1 - ALPHA) * 100:.0f}%, 반복 {REPS} 회)")
print(f"{'n':>6}{'등각 평균':>11}{'가운데 90%':>20}{'등각 폭':>10}")
for i, n in enumerate(NCALS):
    print(f"{n:>6}{m_conf[i]:>10.2f}%{f'[{lo[i]:.1f}, {hi[i]:.1f}]':>20}"
          f"{np.mean(wid_conf[n]):>10.3f}")
print(f"{'정규':>6}{cov_norm.mean() * 100:>10.2f}%"
      f"{f'[{np.quantile(cov_norm, 0.05) * 100:.1f}, {np.quantile(cov_norm, 0.95) * 100:.1f}]':>20}"
      f"{2 * q_norm:>10.3f}")

print(f"\n③ 순위 (보정 {NC} 개, 반복 {len(ranks):,} 회)")
print(f"   평균 {ranks.mean():.1f} (균등이면 {(nb + 1) / 2:.1f}), "
      f"카이제곱 균등성 p = "
      f"{stats.chisquare(np.bincount(ranks, minlength=nb + 1)[1:nb + 1]).pvalue:.3f}")
print(f"   구간 밖이 되는 순위 {k + 1}–{nb} → 확률 {(nb - k) / nb:.4f} "
      f"(명목 {ALPHA} 이하)")
print(f"   실제로 관측된 비율 {np.mean(ranks > k):.4f}")

print(f"\n조건부 포함률 — 등각의 보장은 **주변** 보장이다")
print(f"   주변 {cov_marg * 100:.1f}% — 조용한 성분 {cov_small * 100:.1f}%, "
      f"크게 튄 성분 {cov_big * 100:.1f}%")
