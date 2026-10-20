r"""붓스트랩 표본이 고유 관측의 $63\%$ 만 담는다는 사실은, 자리에 따라
**기능**이 되기도 하고 **비용**이 되기도 한다.

통계학에서 그 $37\%$ 는 공짜 검증집합(OOB)이다. 그런데 신경망 앙상블에서는
구성원 하나하나가 자료를 덜 본다는 뜻이고, 그쪽이 더 아프다. 그래서
Lakshminarayanan 등(2017)은 자료를 재표집하지 않고 **초기값과 섞는 순서만**
바꾸어 앙상블을 만들었다.

① 같은 자료, 두 앙상블의 평균과 $\pm 2\sigma$ 띠 ($M = 5$).
② 구성원 하나의 정확도와 앙상블의 정확도.
③ 구성원 수 $M$ 에 따른 시험집합 음의 로그가능도.

만드는 파일: docs/ch17/bootstrap/img/deep_ensemble.png
"""

import warnings

import numpy as np
import matplotlib

warnings.filterwarnings("ignore")  # 반복 수천 회의 수렴 경고
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.neural_network import MLPRegressor  # noqa: E402

N_TR, N_CAL, N_TE = 200, 100, 3000
M_MAX, REPS = 12, 8
M_SHOW = 5  # 논문의 기본값
SD = 0.3


def truth(x):
    return np.sin(1.5 * x) + 0.3 * x


def sample(n, rng):
    x = rng.uniform(-3, 3, n)
    return x, truth(x) + SD * rng.standard_normal(n)


def net(seed):
    return MLPRegressor(
        (64, 64), activation="tanh", max_iter=4000,
        tol=1e-6, n_iter_no_change=50, random_state=seed,
    )


def fit_members(xtr, ytr, xcal, ycal, method, rep, rng):
    """M_MAX 개의 구성원을 적합한다.

    붓스트랩 앙상블 — 구성원마다 복원추출한 표본으로 학습한다.
    초기값 앙상블  — 구성원마다 전체 자료로 학습하되 초기값과 섞는 순서만 바꾼다.

    두 방식 모두 잡음 분산 sigma_m^2 은 **같은 보정집합**에서 잰다. 학습에
    쓰지 않은 자료라야 구성원이 덜 본 벌을 제대로 치른다.
    """
    mus, s2s, uniq = [], [], []
    for m in range(M_MAX):
        seed = 10_000 * rep + 100 * (method == "boot") + m
        if method == "boot":
            idx = rng.integers(0, len(xtr), len(xtr))
            uniq.append(len(np.unique(idx)) / len(xtr))
            xs, ys = xtr[idx], ytr[idx]
        else:
            xs, ys = xtr, ytr
        f = net(seed).fit(xs[:, None], ys)
        mus.append(f.predict(xcal[:, None]))
        s2s.append(max(np.mean((ycal - mus[-1]) ** 2), 1e-4))
        mus[-1] = f  # 예측기 자체를 보관한다
    return mus, np.array(s2s), uniq


def predict(members, s2s, x, M):
    """구성원 M 개를 가우스 혼합으로 묶고 적률을 맞춘다 (논문 식 그대로).

    mu   = mean(mu_m)
    s^2  = mean(s_m^2 + mu_m^2) - mu^2 = mean(s_m^2) + var(mu_m)
    """
    P = np.array([f.predict(x[:, None]) for f in members[:M]])
    mu = P.mean(axis=0)
    s2 = s2s[:M].mean() + P.var(axis=0)
    return mu, s2, P


def nll(y, mu, s2):
    return 0.5 * np.mean(np.log(2 * np.pi * s2) + (y - mu) ** 2 / s2)


# === 반복 ===========================================================
RNG = np.random.default_rng(20261020)
curves = {"boot": np.zeros((REPS, M_MAX)), "init": np.zeros((REPS, M_MAX))}
rmse_one = {"boot": [], "init": []}
rmse_ens = {"boot": [], "init": []}
band = {"boot": [], "init": []}  # (예측 표준편차, 구성원 사이 흩어짐)
uniq_all = []
shot = {}

for rep in range(REPS):
    xtr, ytr = sample(N_TR, RNG)
    xcal, ycal = sample(N_CAL, RNG)
    xte, yte = sample(N_TE, RNG)
    for method in ("boot", "init"):
        mem, s2s, uq = fit_members(xtr, ytr, xcal, ycal, method, rep, RNG)
        uniq_all += uq
        for M in range(1, M_MAX + 1):
            mu, s2, P = predict(mem, s2s, xte, M)
            curves[method][rep, M - 1] = nll(yte, mu, s2)
            if M == M_SHOW:
                # 참 함수를 알고 있으므로 잡음을 뺀 **추정오차**를 바로 잴 수 있다.
                # RMSE 는 어차피 잡음 SD 에 눌려 차이가 안 보인다.
                f0 = truth(xte)
                rmse_ens[method].append(np.mean((mu - f0) ** 2))
                rmse_one[method].append(np.mean([(p - f0) ** 2 for p in P]))
                band[method].append((np.sqrt(s2).mean(), P.std(axis=0).mean()))
        if rep == 0:
            shot[method] = (mem, s2s)
    print(f"  반복 {rep + 1}/{REPS}")

# ① 에는 마지막 반복이 아니라 **첫 반복**의 학습자료가 필요하다. 같은 씨앗으로
# 난수열을 다시 열면 첫 draw 가 그대로 나온다.
x0, y0 = sample(N_TR, np.random.default_rng(20261020))

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

t = np.linspace(-3, 3, 400)
ax1.axhline(0, color="black", lw=1.5, ls="--", zorder=5)
ax1.text(-2.9, 0.035, "참 함수", fontsize=10)
for method, c, lab in (
    ("boot", "#d62728", "붓스트랩 구성원"),
    ("init", "#1f77b4", "초기값만 바꾼 구성원"),
):
    mem, _ = shot[method]
    for j, f in enumerate(mem[:M_SHOW]):
        ax1.plot(t, f.predict(t[:, None]) - truth(t), color=c, lw=1.4, alpha=0.85,
                 label=lab if j == 0 else None)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$\\hat f(x) - f(x)$")
ax1.set_title(f"① 구성원 {M_SHOW} 개씩, 참 함수로부터의 어긋남", fontsize=12)
ax1.legend(loc="lower left", fontsize=9.5)
ax1.grid(alpha=0.25)

w = 0.34
xs_ = np.arange(2)
one = [np.mean(rmse_one["boot"]), np.mean(rmse_one["init"])]
ens = [np.mean(rmse_ens["boot"]), np.mean(rmse_ens["init"])]
ax2.bar(xs_ - w / 2, one, w, color="#9e9e9e", label="구성원 하나")
ax2.bar(xs_ + w / 2, ens, w, color=["#d62728", "#1f77b4"],
        label=f"앙상블 ($M = {M_SHOW}$)")
for i, (a, b) in enumerate(zip(one, ens)):
    ax2.text(i - w / 2, a + 0.0003, f"{a:.4f}", ha="center", fontsize=10)
    ax2.text(i + w / 2, b + 0.0003, f"{b:.4f}", ha="center", fontsize=10)
ax2.annotate(
    f"붓스트랩은 {M_SHOW} 개를 모으고도\n초기값 쪽 **하나**보다 나쁘다"
    .replace("**", ""),
    xy=(0 + w / 2, ens[0]), xytext=(0.72, ens[0] * 1.18),
    fontsize=10, ha="left",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)
ax2.set_xticks(xs_)
ax2.set_xticklabels(["붓스트랩", "초기값만"])
ax2.set_ylabel("$\\mathbb{E}[(\\hat f(x) - f(x))^2]$  (잡음을 뺀 오차)")
ax2.set_title("② 덜 본 구성원은 앙상블해도 못 따라잡는다", fontsize=12)
ax2.set_ylim(0, max(one) * 1.32)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

Ms = np.arange(1, M_MAX + 1)
for method, c, lab in (
    ("boot", "#d62728", "붓스트랩 앙상블"),
    ("init", "#1f77b4", "초기값만 바꾼 앙상블"),
):
    m_ = curves[method].mean(axis=0)
    se = curves[method].std(axis=0, ddof=1) / np.sqrt(REPS)
    ax3.plot(Ms, m_, "o-", color=c, lw=2.3, ms=5, label=lab)
    ax3.fill_between(Ms, m_ - se, m_ + se, color=c, alpha=0.18, lw=0)
ax3.axvline(M_SHOW, color="#555555", lw=1.3, ls=":")
ax3.text(M_SHOW + 0.25, ax3.get_ylim()[1] - 0.02,
         f"논문의 기본값 $M = {M_SHOW}$", fontsize=10, color="#555555", va="top")
ax3.set_xlabel("앙상블 구성원 수 $M$")
ax3.set_ylabel("시험집합 음의 로그가능도 (낮을수록 좋다)")
ax3.set_title("③ 구성원을 늘려도 간격이 닫히지 않는다", fontsize=12)
ax3.set_xticks([1, 3, 5, 8, 12])
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch17/bootstrap/img/deep_ensemble.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n붓스트랩 표본이 담은 고유 관측의 비율: 실측 {np.mean(uniq_all) * 100:.1f}% "
      f"(이론 {(1 - (1 - 1 / N_TR) ** N_TR) * 100:.1f}%)")
print(f"\n② 잡음을 뺀 추정오차 E[(f_hat - f)^2] (반복 {REPS} 회 평균)")
print(f"{'':>10}{'구성원 하나':>14}{'앙상블':>12}{'앙상블의 이득':>14}")
for k, lab in (("boot", "붓스트랩"), ("init", "초기값만")):
    a, b = np.mean(rmse_one[k]), np.mean(rmse_ens[k])
    print(f"{lab:>10}{a:>14.5f}{b:>12.5f}{a / b:>12.2f} 배")
print(f"  붓스트랩 앙상블 {np.mean(rmse_ens['boot']):.5f} 대 "
      f"초기값 구성원 하나 {np.mean(rmse_one['init']):.5f} — "
      f"{np.mean(rmse_ens['boot']) / np.mean(rmse_one['init']):.2f} 배 나쁘다")
print(f"\n① 구성원 사이의 흩어짐 (M = {M_SHOW})")
print(f"{'':>10}{'예측 표준편차':>16}{'구성원 사이 흩어짐':>20}")
for k, lab in (("boot", "붓스트랩"), ("init", "초기값만")):
    a = np.array(band[k]).mean(axis=0)
    print(f"{lab:>10}{a[0]:>16.4f}{a[1]:>20.4f}")

print(f"\n③ 시험집합 NLL")
print(f"{'M':>4}{'붓스트랩':>12}{'초기값만':>12}{'차이':>10}")
for i, M in enumerate(Ms):
    b, n_ = curves["boot"].mean(axis=0)[i], curves["init"].mean(axis=0)[i]
    print(f"{M:>4}{b:>12.4f}{n_:>12.4f}{b - n_:>10.4f}")
print(f"\n  M={M_SHOW} 에서 붓스트랩 쪽이 NLL 을 "
      f"{curves['boot'].mean(axis=0)[M_SHOW - 1] - curves['init'].mean(axis=0)[M_SHOW - 1]:.3f} "
      f"더 치른다")
print(f"  초기값 앙상블은 M=1 에서 M={M_SHOW} 로 가며 NLL 이 "
      f"{curves['init'].mean(axis=0)[0]:.3f} -> {curves['init'].mean(axis=0)[M_SHOW - 1]:.3f}")
print(f"  붓스트랩 앙상블은 M={M_MAX} 에서도 "
      f"{curves['boot'].mean(axis=0)[-1]:.3f} 로 초기값 쪽 M=1 "
      f"({curves['init'].mean(axis=0)[0]:.3f}) 과 견준다")
