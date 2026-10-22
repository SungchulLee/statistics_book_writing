r"""시드 몇 개로 「우리 방법이 더 안정적이다」를 말할 수 있는가.

두 방법의 시드별 성능이 각각 $k$ 개 있을 때 "분산이 더 작다"는 주장은
분산 동일성 검정이고, 이 쪽의 $F$ 검정이 바로 그것이다. 검정력 쪽은 이
쪽 $\S 3$ 이 이미 계산해 두었다(SD 비 $2$ 에 $19$, $1.5$ 에 $50$,
$1.25$ 에 $160$). 여기서는 **크기** 쪽을 잰다.

① 시드별 지표가 어떻게 생겼는가 — 첨도 $\beta_2$ 와 함께.
② 귀무가설 아래의 실제 기각률. $14$ 장이 한 표본 카이제곱 분산검정에
   대해 유도한 식
   $\alpha_\infty = 2[1 - \Phi(z_{0.975}\sqrt{2/(\beta_2-1)})]$
   이 두 표본 $F$ 검정의 극한 크기도 그대로 맞히는지 본다.
③ 르빈과 브라운–포사이드는 어떤가.

만드는 파일: docs/ch15/f_test/img/seed_variance_power.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(11)
ALPHA = 0.05
KS = [5, 10, 20, 30, 50, 200, 1000]
K_SHOW = 30
LN_S, BAD_P, BAD_S = 0.8, 0.07, 3.5
KINDS = ["정규", "치우침", "유계", "가끔 나쁜 시드"]
Z = stats.norm.ppf(1 - ALPHA / 2)

# === 첨도는 해석적으로 ===============================================
s2 = LN_S ** 2
B2 = {
    "정규": 3.0,
    "치우침": float(np.exp(4 * s2) + 2 * np.exp(3 * s2)
                  + 3 * np.exp(2 * s2) - 3),
    "유계": float(stats.beta(2, 18).stats(moments="k")) + 3.0,
    "가끔 나쁜 시드": float(
        3 * ((1 - BAD_P) + BAD_P * BAD_S ** 4)
        / ((1 - BAD_P) + BAD_P * BAD_S ** 2) ** 2),
}
PRED = {k: float(2 * (1 - stats.norm.cdf(Z * np.sqrt(2 / (v - 1)))))
        for k, v in B2.items()}


def draw(kind, k, r):
    """귀무가설이 참인 (= 두 쪽의 분산이 같은) 시드별 성능 r 벌."""
    if kind == "정규":
        return RNG.normal(0, 1, (r, k))
    if kind == "치우침":
        return RNG.lognormal(0, LN_S, (r, k))
    if kind == "유계":
        return RNG.beta(2, 18, (r, k))          # 정확도처럼 [0,1] 에 갇힌다
    x = RNG.normal(0, 1, (r, k))                # 열넷에 하나쯤 터지는 시드
    bad = RNG.random((r, k)) < BAD_P
    return np.where(bad, RNG.normal(0, BAD_S, (r, k)), x)


def p_F(a, b):
    d = a.shape[1] - 1
    c = stats.f.cdf(a.var(axis=1, ddof=1) / b.var(axis=1, ddof=1), d, d)
    return 2 * np.minimum(c, 1 - c)


def p_levene(a, b, center="mean"):
    """두 집단 르빈 검정 — |x - 중심| 에 대한 등분산 t 검정과 같다."""
    f = np.mean if center == "mean" else np.median
    za = np.abs(a - f(a, axis=1, keepdims=True))
    zb = np.abs(b - f(b, axis=1, keepdims=True))
    k = a.shape[1]
    sp = np.sqrt((za.var(axis=1, ddof=1) + zb.var(axis=1, ddof=1)) / 2)
    t = (za.mean(axis=1) - zb.mean(axis=1)) / (sp * np.sqrt(2.0 / k))
    return 2 * stats.t.sf(np.abs(t), 2 * k - 2)


_a, _b = draw("정규", 8, 3), draw("정규", 8, 3)
for ctr in ("mean", "median"):
    assert np.allclose(p_levene(_a, _b, ctr),
                       [stats.levene(_a[i], _b[i], center=ctr).pvalue
                        for i in range(3)], atol=1e-10)
print("르빈·브라운–포사이드 구현이 scipy 와 일치함을 확인했다.")

print(f"\n시드별 지표의 첨도와 예측되는 극한 크기 (명목 {ALPHA})")
print(f"{'분포':>14}{'beta2':>9}{'예측 극한':>11}")
for kd in KINDS:
    print(f"{kd:>14}{B2[kd]:>9.2f}{PRED[kd]:>11.3f}")


def rates(kind, k):
    """메모리를 아끼려 덩어리로 돌린다."""
    reps = 20000 if k <= 50 else 6000
    chunk = max(1, int(2.0e6 // k))
    nf = nl = nb = done = 0
    while done < reps:
        m = min(chunk, reps - done)
        A, B = draw(kind, k, m), draw(kind, k, m)
        nf += int((p_F(A, B) < ALPHA).sum())
        nl += int((p_levene(A, B, "mean") < ALPHA).sum())
        nb += int((p_levene(A, B, "median") < ALPHA).sum())
        done += m
    return nf / reps, nl / reps, nb / reps


print(f"\n귀무가설 아래의 실제 기각률")
print(f"{'시드':>5}{'분포':>14}{'F 검정':>9}{'르빈':>8}{'브라운-포사이드':>15}")
SIZE = {k: {} for k in KS}
for k in KS:
    for kd in KINDS:
        SIZE[k][kd] = rates(kd, k)
        f_, l_, b_ = SIZE[k][kd]
        print(f"{k:>5}{kd:>14}{f_:>9.3f}{l_:>8.3f}{b_:>15.3f}")

print("\n예측과 측정 (F 검정)")
print(f"{'분포':>14}{'예측 극한':>11}{'k=1000 측정':>13}")
for kd in KINDS:
    print(f"{kd:>14}{PRED[kd]:>11.3f}{SIZE[1000][kd][0]:>13.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
CK = {"정규": "#2ca02c", "치우침": "#d62728", "유계": "#1f77b4",
      "가끔 나쁜 시드": "#ff7f0e"}

for kd in KINDS:
    x = draw(kd, 200000, 1).ravel()
    x = (x - x.mean()) / x.std()                # 분산을 맞춰 모양만 견준다
    g = np.linspace(-3.0, 4.5, 400)
    ax1.plot(g, stats.gaussian_kde(x[:60000])(g), color=CK[kd], lw=2.4,
             label=f"{kd}  ($\\beta_2 = {B2[kd]:.1f}$)")
ax1.set_xlabel("시드별 성능 (분산을 맞춘 뒤)")
ax1.set_ylabel("밀도")
ax1.set_xlim(-3.0, 4.5)
ax1.set_title("① 시드별 지표는 정규가 아니다", fontsize=12)
ax1.legend(loc="upper right", fontsize=9.5)
ax1.grid(alpha=0.25)

for kd in KINDS:
    ax2.plot(KS, [SIZE[k][kd][0] for k in KS], "o-", color=CK[kd], lw=2.6,
             ms=7, label=kd)
    ax2.axhline(PRED[kd], color=CK[kd], lw=1.3, ls=":")
    ax2.text(1150, PRED[kd], f"${PRED[kd]:.3f}$", fontsize=9.5,
             color=CK[kd], va="center")
ax2.axhline(ALPHA, color="black", lw=1.4, ls="--")
ax2.set_xscale("log")
ax2.set_xticks(KS)
ax2.set_xticklabels([f"${v}$" for v in KS], fontsize=9)
ax2.minorticks_off()
ax2.set_xlim(4.3, 2400)
ax2.set_xlabel("시드 수 $k$")
ax2.set_ylabel("귀무가설 아래의 실제 기각률")
ax2.set_ylim(0, 0.72)
ax2.set_title("② 점선은 $14$ 장의 식이 예측한 극한", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5)
ax2.grid(alpha=0.25)

xs = np.arange(len(KINDS))
w = 0.26
for j, (nm, c) in enumerate([("$F$ 검정", "#999999"), ("르빈", "#9467bd"),
                             ("브라운–포사이드", "#2ca02c")]):
    v = [SIZE[K_SHOW][kd][j] for kd in KINDS]
    ax3.bar(xs + (j - 1) * w, v, w, color=c, label=nm)
    for x, y in zip(xs, v):
        ax3.text(x + (j - 1) * w, y + 0.009, f"${y:.3f}$", ha="center",
                 fontsize=8, rotation=90 if y < 0.08 else 0, va="bottom")
ax3.axhline(ALPHA, color="black", lw=1.4, ls="--")
ax3.set_xticks(xs)
ax3.set_xticklabels(KINDS, fontsize=10)
ax3.set_ylabel("귀무가설 아래의 실제 기각률")
ax3.set_ylim(0, 0.56)
ax3.set_title(f"③ 시드 ${K_SHOW}$ 개에서 세 검정", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch15/f_test/img/seed_variance_power.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
