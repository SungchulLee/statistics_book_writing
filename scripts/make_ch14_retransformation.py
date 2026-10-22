r"""변환하고 되돌리면 **추정하던 양이 바뀐다.**

로그 척도에서 적합한 뒤 $\exp$ 로 되돌리면 나오는 것은 조건부 **평균**이 아니라
조건부 **중앙값**(정확히는 기하평균)이다. 로그정규라면

$$E[Y \mid x] = e^{\mu(x) + \sigma^2/2}, \qquad e^{\mu(x)} = \text{중앙값}$$

이므로 되돌린 값은 $e^{\sigma^2/2}$ 배만큼 작다.

① $\sigma$ 가 커질수록 과소예측이 커진다 — $\sigma = 1$ 이면 $39\%$ 다.
② 실제 회귀에서 총합 예측이 그만큼 어긋난다.
③ 고치는 두 길 — $\hat\sigma^2/2$ 보정과 두안의 **스미어링**. 로그잔차가 정규가
   아니면 앞의 것이 틀리고 뒤의 것만 버틴다.

만드는 파일: docs/ch14/non_normal_data/img/retransformation.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261022)

N = 20000
SIGS = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
SIG0 = 0.8                      # ②③ 에서 쓰는 로그 척도의 잔차 크기


def truth(x):
    """로그 척도의 평균 구조."""
    return 1.0 + 1.2 * x


def make(sig, kind="normal", n=N):
    """로그잔차의 모양을 바꿔 가며 자료를 만든다 (평균 0, 표준편차 sig)."""
    x = RNG.uniform(0, 2, n)
    if kind == "normal":
        e = RNG.normal(0, sig, n)
    else:                        # 치우친 잔차 — 평균과 분산을 맞춰 둔다
        k = 1.5
        g = RNG.gamma(k, 1.0, n)
        e = (g - k) / np.sqrt(k) * sig
    return x, truth(x) + e, e


def fit_log(x, logy):
    """로그 척도에서 최소제곱."""
    D = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(D, logy, rcond=None)[0]
    res = logy - D @ b
    return b, res


def back(x, b, res, how):
    mu = b[0] + b[1] * x
    if how == "naive":
        return np.exp(mu)
    if how == "sigma":                       # 정규를 가정한 보정
        return np.exp(mu + res.var(ddof=2) / 2)
    return np.exp(mu) * np.mean(np.exp(res))  # 두안의 스미어링


# === ① 이론적인 크기 =================================================
print("① 로그잔차 표준편차에 따른 과소예측 (이론값)")
print(f"{'sigma':>7}{'E[Y]/중앙값':>14}{'과소예측':>10}")
ratio = np.exp(SIGS ** 2 / 2)
for s, r in zip(SIGS, ratio):
    print(f"{s:>7.1f}{r:>14.3f}{1 - 1 / r:>9.1%}")

# === ② 실제 회귀에서 =================================================
print(f"\n② 로그 척도에서 적합하고 되돌릴 때 (sigma = {SIG0}, n = {N:,})")
res_tbl = {}
for kind in ("normal", "skewed"):
    x, logy, e = make(SIG0, kind)
    y = np.exp(logy)
    b, r = fit_log(x, logy)
    true_mean = float(np.mean(np.exp(truth(x)) * np.mean(np.exp(e))))
    row = {}
    for how in ("naive", "sigma", "smear"):
        pred = back(x, b, r, how)
        row[how] = float(pred.mean() / true_mean - 1)
    res_tbl[kind] = row
    name = "정규 잔차" if kind == "normal" else "치우친 잔차"
    print(f"   {name}: " + ",  ".join(
        f"{k} {100 * v:+.1f}%" for k, v in row.items()))

# === ③ sigma 를 바꿔 가며 세 방법 =====================================
print(f"\n③ 잔차 모양과 크기에 따른 상대오차 (%)")
curves = {}
for kind in ("normal", "skewed"):
    curves[kind] = {h: [] for h in ("naive", "sigma", "smear")}
    for s in SIGS:
        x, logy, e = make(s, kind)
        b, r = fit_log(x, logy)
        tm = float(np.mean(np.exp(truth(x)) * np.mean(np.exp(e))))
        for h in curves[kind]:
            curves[kind][h].append(100 * (back(x, b, r, h).mean() / tm - 1))
    print(f"   [{kind}]")
    for h in curves[kind]:
        print(f"     {h:>6}: " + " ".join(f"{v:>7.1f}" for v in curves[kind][h]))

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = {"naive": "#d62728", "sigma": "#ff7f0e", "smear": "#1f77b4"}
LBL = {"naive": "그냥 $\\exp$ 로 되돌리기",
       "sigma": "$+\\hat\\sigma^2/2$ 보정 (정규 가정)",
       "smear": "스미어링 (가정 없음)"}

ax1.plot(SIGS, 100 * (1 - 1 / ratio), "o-", color="#d62728", lw=2.8, ms=8)
for s in (0.5, 1.0):
    i = int(np.argmin(np.abs(SIGS - s)))
    ax1.plot(SIGS[i], 100 * (1 - 1 / ratio[i]), "*", color="black", ms=16,
             zorder=5)
    ax1.annotate(f"$\\sigma = {SIGS[i]}$ 이면\n${100 * (1 - 1 / ratio[i]):.0f}\\%$",
                 xy=(SIGS[i], 100 * (1 - 1 / ratio[i])),
                 xytext=(SIGS[i] - 0.05, 100 * (1 - 1 / ratio[i]) + 13),
                 fontsize=10.5, ha="right",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax1.set_xlabel("로그 척도에서 잔차의 표준편차 $\\sigma$")
ax1.set_ylabel("평균을 얼마나 적게 예측하는가 ($\\%$)")
ax1.set_ylim(0, 75)
ax1.set_title("① 되돌리면 중앙값이 나온다", fontsize=12)
ax1.grid(alpha=0.25)

names = list(LBL)
xpos = np.arange(2)
w = 0.26
for i, h in enumerate(names):
    vals = [100 * res_tbl[k][h] for k in ("normal", "skewed")]
    ax2.bar(xpos + (i - 1) * w, vals, w, color=COL[h], label=LBL[h])
    for xx, v in zip(xpos + (i - 1) * w, vals):
        ax2.text(xx, v + (2.0 if v >= 0 else -4.5), f"${v:+.1f}$",
                 ha="center", fontsize=10.5)
ax2.axhline(0, color="black", lw=1.4)
ax2.set_xticks(xpos)
ax2.set_xticklabels(["정규 잔차", "치우친 잔차"], fontsize=11)
ax2.set_ylabel("평균 예측의 상대오차 ($\\%$)")
ax2.set_ylim(-60, 34)
ax2.set_title(f"② 세 가지 되돌리기 ($\\sigma = {SIG0}$)", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

for h in names:
    ax3.plot(SIGS, curves["skewed"][h], "o-", color=COL[h], lw=2.6, ms=7,
             label=LBL[h])
ax3.axhline(0, color="black", lw=1.4)
ax3.set_xlabel("로그 척도에서 잔차의 표준편차 $\\sigma$")
ax3.set_ylabel("평균 예측의 상대오차 ($\\%$)")
ax3.set_title("③ 로그잔차가 치우쳐 있을 때", fontsize=12)
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch14/non_normal_data/img/retransformation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
