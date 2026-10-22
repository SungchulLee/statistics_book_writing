r"""이름이 같다고 같은 검정이 아니다 — `sklearn` 의 `chi2` 특징 점수.

`sklearn.feature_selection.chi2` 는 독립성 검정의 식을 닮았지만 다른 물건이다.
각 class 에서 특징값을 **더한 것**을 관측도수로, 전체 합을 class 비율로 나눈 것을
기대도수로 놓는다. 그래서 특징이 **도수**일 때만 뜻이 맞고, 자가 바뀌면 점수가
그 배로 바뀐다.

① 붓꽃 자료에서 한 열만 cm 에서 mm 로 바꾸면 순위가 뒤집힌다.
② 귀무가설이 참일 때 통계량의 기댓값이 $(k-1)\,\sigma^2/\mu$ 다 — $(k-1)$ 이 아니다.
③ 그래서 유의수준이 지켜지지 않는다. 분산/평균이 $1$ 일 때만 $0.05$ 다.

만드는 파일: docs/ch10/practice/img/sklearn_chi2_scale.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.datasets import load_iris          # noqa: E402
from sklearn.feature_selection import chi2      # noqa: E402

RNG = np.random.default_rng(0)

# === ① 자를 바꾸면 순위가 바뀐다 ======================================
IRIS = load_iris()
X, Y = IRIS.data, IRIS.target
NAMES = ["꽃받침 길이", "꽃받침 너비", "꽃잎 길이", "꽃잎 너비"]
S_CM = chi2(X, Y)[0]
XM = X.copy()
XM[:, 0] *= 10                                   # 첫 열만 mm 로 적는다
S_MM = chi2(XM, Y)[0]
R_CM = np.argsort(np.argsort(-S_CM)) + 1
R_MM = np.argsort(np.argsort(-S_MM)) + 1
print("① 붓꽃 — 꽃받침 길이만 cm 에서 mm 로 바꾼다")
print(f"{'특징':>12}{'cm 점수':>11}{'순위':>6}{'mm 점수':>11}{'순위':>6}")
for i, nm in enumerate(NAMES):
    print(f"{nm:>12}{S_CM[i]:>11.3f}{R_CM[i]:>6}{S_MM[i]:>11.3f}{R_MM[i]:>6}")
print(f"   바뀐 열의 점수가 정확히 {S_MM[0] / S_CM[0]:.1f} 배가 되고, "
      f"순위가 {R_CM[0]} 위에서 {R_MM[0]} 위로 올라간다.")
print("   자료는 그대로다. 센티미터를 밀리미터로 적었을 뿐이다.")

# === ②③ 귀무가설이 참일 때 ===========================================
K, N, REP = 3, 300, 3000
YNULL = np.repeat(np.arange(K), N // K)
THETAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
SHAPE = 2.0            # 감마(a, theta): 분산/평균 = theta 로 shape 과 무관하다


def null_run(draw, rep=REP):
    st = np.empty(rep)
    pv = np.empty(rep)
    for i in range(rep):
        s, p = chi2(draw(N)[:, None], YNULL)
        st[i] = s[0]
        pv[i] = p[0]
    return st.mean(), float((pv < 0.05).mean())


print(f"\n②③ 특징이 y 와 **무관**할 때 (class {K} 개, n = {N}, {REP:,} 번)")
print(f"{'분산/평균':>10}{'E[통계량]':>11}{'(k-1)*분산/평균':>16}{'기각률 a=.05':>13}")
EST, REJ = [], []
for th in THETAS:
    m, r = null_run(lambda n, t=th: RNG.gamma(SHAPE, t, n))
    EST.append(m)
    REJ.append(r)
    print(f"{th:>10.2f}{m:>11.3f}{(K - 1) * th:>16.3f}{r:>13.3f}")
print(f"   기댓값이 (k-1) x 분산/평균 과 "
      f"{max(abs(a / ((K - 1) * t) - 1) for a, t in zip(EST, THETAS)) * 100:.1f}% "
      f"안에서 맞는다.")
print(f"   기각률이 {min(REJ):.3f} 에서 {max(REJ):.3f} 까지 움직인다 — "
      f"**어느 경우에도 특징은 y 와 무관하다.**")

NAMED = {
    "균등 $(9, 11)$": lambda n: RNG.uniform(9, 11, n),
    "푸아송 $(5)$": lambda n: RNG.poisson(5.0, n).astype(float),
    "지수 (평균 $1$)": lambda n: RNG.exponential(1.0, n),
    "로그정규 $(0, 1)$": lambda n: RNG.lognormal(0, 1, n),
}
print(f"\n   이름 있는 분포 몇 개")
print(f"{'분포':>16}{'분산/평균':>11}{'E[통계량]':>11}{'기각률':>10}")
PTS = []
for nm, f in NAMED.items():
    big = f(1_000_000)
    d = float(big.var() / big.mean())
    m, r = null_run(f, rep=1500)
    PTS.append((nm, d, m, r))
    print(f"{nm:>16}{d:>11.3f}{m:>11.3f}{r:>10.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_R = "#1f77b4", "#ff7f0e", "#d62728"

xs = np.arange(4)
w = 0.38
ax1.bar(xs - w / 2, S_CM, w, color=C_A, label="모두 $\\mathrm{cm}$ 로 적을 때")
ax1.bar(xs + w / 2, S_MM, w, color=C_B,
        label="꽃받침 길이만 $\\mathrm{mm}$ 로 적을 때")
for i in range(4):
    ax1.text(i - w / 2, S_CM[i] + 3, f"${R_CM[i]}$위", ha="center",
             fontsize=10, color=C_A)
    ax1.text(i + w / 2, S_MM[i] + 3, f"${R_MM[i]}$위", ha="center",
             fontsize=10, color=C_B)
ax1.text(0 + w / 2, S_MM[0] * 0.45, "$\\times 10$", fontsize=13,
         color="white", ha="center", va="center")
ax1.set_xticks(xs)
ax1.set_xticklabels(NAMES, fontsize=10)
ax1.set_ylabel("$\\mathtt{sklearn}$ 의 $\\mathtt{chi2}$ 점수")
ax1.set_ylim(0, 142)
ax1.set_title("① 자를 바꾸면 순위가 바뀐다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

ax2.loglog(THETAS, EST, "o", color=C_A, ms=10, label="모의한 기댓값 (감마)")
ax2.loglog(THETAS, (K - 1) * THETAS, "-", color="#666666", lw=2.0,
           label=f"$(k-1)\\,\\sigma^2/\\mu$")
OFF = [(0.5, 0.4), (0.6, -1.5), (0.6, 0.6), (0.5, 0.4)]
for (nm, d, m, r), off in zip(PTS, OFF):
    ax2.loglog([d], [m], "s", color=C_R, ms=8)
    ax2.annotate(nm, (d, m), xytext=off, textcoords="offset fontsize",
                 fontsize=9, color=C_R)
ax2.axhline(K - 1, color=C_B, lw=1.5, ls="--")
ax2.text(0.035, (K - 1) * 1.25, "$\\chi^2(k-1)$ 이라면 여기여야 한다",
         color=C_B, fontsize=10.5)
ax2.set_xlabel("특징의 분산/평균 ($\\sigma^2/\\mu$)")
ax2.set_ylabel("통계량의 기댓값")
ax2.set_xticks([0.1, 1, 10])
ax2.set_xticklabels(["$0.1$", "$1$", "$10$"])
ax2.set_yticks([0.1, 1, 10])
ax2.set_yticklabels(["$0.1$", "$1$", "$10$"])
ax2.minorticks_off()
ax2.set_title("② 분포가 틀렸다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.semilogx(THETAS, REJ, "o-", color=C_A, lw=2.6, ms=9, label="감마")
for nm, d, m, r in PTS:
    ax3.semilogx([d], [r], "s", color=C_R, ms=8)
ax3.axhline(0.05, color=C_B, lw=1.6, ls="--")
ax3.axvline(1.0, color="#888888", lw=1.3, ls=":")
ax3.text(1.25, 0.72, "분산/평균 $= 1$\n곧 도수 자료일 때만\n$0.05$ 가 된다",
         fontsize=10.5, color="#555555")
ax3.text(0.035, 0.09, "$\\alpha = 0.05$", color=C_B, fontsize=10.5)
ax3.set_xlabel("특징의 분산/평균 ($\\sigma^2/\\mu$)")
ax3.set_ylabel("$y$ 와 무관한 특징을 뽑아낸 비율")
ax3.set_xticks([0.1, 1, 10])
ax3.set_xticklabels(["$0.1$", "$1$", "$10$"])
ax3.minorticks_off()
ax3.set_ylim(-0.03, 1.0)
ax3.set_title("③ 유의수준이 지켜지지 않는다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch10/practice/img/sklearn_chi2_scale.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
