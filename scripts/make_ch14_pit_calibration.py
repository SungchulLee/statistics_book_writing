r"""예측분포가 맞는지는 정규성 검정으로 볼 수 없다 — 원리적으로.

이 장의 검정들은 **위치와 척도에 불변**이다. 표준화 잔차
$z = (y-\hat\mu)/\hat\sigma$ 의 샤피로–윌크 통계량은 $\hat\mu$ 와 $\hat\sigma$ 를
어떻게 잡든 똑같은 값이 나온다. 그러니 「중심이 맞는가, 폭이 맞는가」는 그 검정이
답할 수 있는 물음이 아니다.

예측분포를 통째로 보려면 **확률적분변환**을 쓴다 — $u_i = \hat F_i(y_i)$ 가
$U(0,1)$ 이어야 한다. 이것이 PIT 히스토그램이다.

① 네 모형의 RMSE 와 샤피로–윌크 $p$ 값이 사실상 같다.
② PIT 히스토그램은 넷을 모두 구별한다 — 균등·U자·언덕·기울기.
③ 기각률로 보면 정규성 검정 $0.05$ 대 PIT 검정 $1.000$ 이다.

만드는 파일: docs/ch14/graphical_methods/img/pit_calibration.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

N, SIG, REPS = 2000, 1.0, 500
NSW = 1500                       # 샤피로–윌크에 쓰는 표본 (scipy 권장 범위)
MODELS = {                       # (중심의 치우침, 폭의 배율)
    "잘 보정": (0.0, 1.0),
    "과신 $(\\hat\\sigma = 0.6\\sigma)$": (0.0, 0.6),
    "과소신뢰 $(1.6\\sigma)$": (0.0, 1.6),
    "치우침 $(+0.5\\sigma)$": (0.5, 1.0),
}

y = RNG.normal(0.0, SIG, N)
print(f"참 예측분포 $N(0, {SIG})$, 표본 {N:,}")
print(f"{'모형':>26}{'RMSE':>9}{'z 의 SW p':>12}{'PIT KS p':>12}{'로그점수':>10}")
rows = {}
for name, (mu, sc) in MODELS.items():
    z = (y - mu) / (sc * SIG)
    pit = stats.norm.cdf(z)
    rows[name] = dict(
        rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
        sw=float(stats.shapiro(z[:NSW]).pvalue),
        ks=float(stats.kstest(pit, "uniform").pvalue),
        ls=float(np.mean(-stats.norm.logpdf(y, mu, sc * SIG))),
        pit=pit, mu=mu, sc=sc)
    r = rows[name]
    print(f"{name.replace('$',''):>26}{r['rmse']:>9.4f}{r['sw']:>12.3g}"
          f"{r['ks']:>12.3g}{r['ls']:>10.4f}")
print("\n  샤피로–윌크 p 값이 넷 다 같은 것은 우연이 아니다 — 그 통계량이")
print("  아핀변환에 불변이라 z 를 어떻게 만들든 값이 바뀌지 않는다.")

# === ③ 기각률 =======================================================
print(f"\n③ 기각률 ({REPS} 회, n = {N:,})")
print(f"{'모형':>26}{'샤피로–윌크':>13}{'PIT–KS':>10}")
rej = {}
for name, (mu, sc) in MODELS.items():
    a = b = 0
    for _ in range(REPS):
        yy = RNG.normal(0.0, SIG, N)
        z = (yy - mu) / (sc * SIG)
        a += int(stats.shapiro(z[:NSW]).pvalue < 0.05)
        b += int(stats.kstest(stats.norm.cdf(z), "uniform").pvalue < 0.05)
    rej[name] = (a / REPS, b / REPS)
    print(f"{name.replace('$',''):>26}{rej[name][0]:>13.3f}{rej[name][1]:>10.3f}")

# === 그림 ============================================================
fig = plt.figure(figsize=(15.4, 5.0))
gs = fig.add_gridspec(1, 3, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
sub = gs[0, 1].subgridspec(2, 2, hspace=0.42, wspace=0.26)
ax3 = fig.add_subplot(gs[0, 2])
COL = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
names = list(MODELS)

xs = np.linspace(-4, 4, 400)
ax1.hist(y, bins=40, density=True, color="#cccccc", alpha=0.8, label="자료")
for name, c in zip(names, COL):
    mu, sc = MODELS[name]
    ax1.plot(xs, stats.norm.pdf(xs, mu, sc * SIG), color=c, lw=2.4, label=name)
ax1.text(0.02, 0.76,
         f"RMSE : 앞의 셋이 모두 ${rows[names[0]]['rmse']:.3f}$\n"
         f"샤피로–윌크 $p$ : 넷 다 ${rows[names[0]]['sw']:.3f}$",
         transform=ax1.transAxes, va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax1.set_xlabel("$y$")
ax1.set_ylabel("밀도")
ax1.set_ylim(0, 1.02)
ax1.set_title("① 네 예측분포, 같은 진단값", fontsize=12)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax1.grid(alpha=0.25)

for k, (name, c) in enumerate(zip(names, COL)):
    ax = fig.add_subplot(sub[k // 2, k % 2])
    ax.hist(rows[name]["pit"], bins=np.linspace(0, 1, 21), density=True,
            color=c, alpha=0.8)
    ax.axhline(1.0, color="black", lw=1.2, ls=":")
    ax.set_ylim(0, 3.4)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["$0$", "", "$1$"],
                                                   fontsize=8.5)
    ax.set_yticks([])
    ax.set_title(name, fontsize=9.5, pad=3)
fig.text(0.515, 0.945, "② PIT 히스토그램은 넷을 구별한다", ha="center",
         fontsize=12)
fig.text(0.515, 0.045, "점선이 균등분포의 높이", ha="center", fontsize=10,
         color="#555555")

xpos = np.arange(len(names))
w = 0.36
ax3.bar(xpos - w / 2, [rej[n][0] for n in names], w, color="#999999",
        label="표준화 잔차의 샤피로–윌크")
ax3.bar(xpos + w / 2, [rej[n][1] for n in names], w, color="#1f77b4",
        label="PIT 의 균등성 검정")
for i, n in enumerate(names):
    ax3.text(i - w / 2, rej[n][0] + 0.02, f"${rej[n][0]:.3f}$", ha="center",
             fontsize=9.5)
    ax3.text(i + w / 2, rej[n][1] + 0.02, f"${rej[n][1]:.3f}$", ha="center",
             fontsize=9.5)
ax3.axhline(0.05, color="black", lw=1.3, ls=":")
ax3.set_xticks(xpos)
ax3.set_xticklabels(["잘 보정", "과신", "과소신뢰", "치우침"], fontsize=10)
ax3.set_ylim(0, 1.16)
ax3.set_ylabel("기각률")
ax3.set_title(f"③ 무엇을 잡아내는가 ($n = {N:,}$)".replace(",", "{,}"),
              fontsize=12)
ax3.legend(loc="center left", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

out = "docs/ch14/graphical_methods/img/pit_calibration.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
