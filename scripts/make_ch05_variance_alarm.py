r"""분산을 지켜보는 장치가 거짓경보를 쏟는 까닭 — 꼬리와 $\operatorname{Var}(S^2)$.

이 쪽은 지수모집단에서 $S^2$ 의 흔들림이 카이제곱 예측보다 **크다**는 것을 보인다.
그 사실이 값을 치르는 자리가 있다 — 분산이 커졌는지 지켜보다 넘으면 알리는 장치다.
문턱을 카이제곱으로 잡으면, 꼬리가 두꺼운 자료에서 **아무 일도 없는데 울린다.**

① 이름값 $1\%$ 로 잡은 문턱의 실제 거짓경보율.
② 까닭은 한 줄이다 — $\operatorname{Var}(S^2)$ 가 첨도를 탄다.
③ 고치는 법 — 문턱을 자료에서 뽑으면 돌아온다.

만드는 파일: docs/ch05/applications/img/variance_alarm.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.01
REP = 200_000

POPS = {
    "균등": (lambda sh: RNG.uniform(-np.sqrt(3), np.sqrt(3), sh), -1.2),
    "정규": (lambda sh: RNG.standard_normal(sh), 0.0),
    "지수": (lambda sh: RNG.exponential(1.0, sh) - 1.0, 6.0),
    "로그정규": (lambda sh: (RNG.lognormal(0, 1, sh) - np.exp(0.5))
                 / np.sqrt((np.e - 1) * np.e),
                 float(np.exp(4 * 1.0) + 2 * np.exp(3 * 1.0)
                       + 3 * np.exp(2 * 1.0) - 6)),
}
NS = [10, 20, 50, 100, 200, 500]

# === ① 거짓경보율 ====================================================
print(f"① '분산이 커졌다' 경보 — 문턱을 카이제곱으로 잡았을 때 (이름값 "
      f"{100 * ALPHA:.0f}%)")
print(f"{'n':>6}" + "".join(f"{k:>12}" for k in POPS))
FA = {k: [] for k in POPS}
for n in NS:
    crit = stats.chi2.ppf(1 - ALPHA, n - 1) / (n - 1)
    for k, (draw, _) in POPS.items():
        x = draw((REP, n))
        s2 = x.var(axis=1, ddof=1)
        FA[k].append(float(np.mean(s2 > crit)))
    print(f"{n:>6}" + "".join(f"{FA[k][-1]:>12.4f}" for k in POPS))
print(f"   정규에서만 {ALPHA:.2f} 다. 지수에서는 그 "
      f"{FA['지수'][-1] / ALPHA:.1f} 배, 로그정규에서는 "
      f"{FA['로그정규'][-1] / ALPHA:.1f} 배 울린다. "
      f"**n 을 키워도 줄지 않는다.**")

# === ② 까닭 =========================================================
print(f"\n② Var(S^2) = (kappa - 1 + 2/(n-1)) sigma^4 / n 인가 (sigma = 1)")
print(f"{'모집단':>8}{'초과첨도':>10}{'n=50 모의':>12}{'식':>10}"
      f"{'n=200 모의':>12}{'식':>10}")
KUR, V50, V200 = [], [], []
for k, (draw, ex) in POPS.items():
    row = []
    for n in (50, 200):
        x = draw((REP, n))
        row.append(float(x.var(axis=1, ddof=1).var()))
    kap = ex + 3.0
    f50 = (kap - 1 + 2 / 49) / 50
    f200 = (kap - 1 + 2 / 199) / 200
    KUR.append(ex)
    V50.append(row[0])
    V200.append(row[1])
    print(f"{k:>8}{ex:>10.2f}{row[0]:>12.5f}{f50:>10.5f}"
          f"{row[1]:>12.5f}{f200:>10.5f}")
print("   첨도가 그대로 들어온다. 카이제곱은 kappa = 3 을 가정한 특수한 경우다.")

# === ③ 문턱을 자료에서 뽑으면 =========================================
# 감시 장치는 '과거 자료'로 문턱을 잡고 '새 표본'을 잰다. 같은 표본으로
# 문턱을 잡으면 자기 자신을 넘을 수 없어 아무 뜻이 없다.
N3, M3, B, R3 = 50, 500, 2000, 4000
print(f"\n③ 과거 자료 {M3} 개로 문턱을 잡고 새 표본 {N3} 개를 잰다 (B = {B:,})")
print(f"{'모집단':>8}{'카이제곱 문턱':>14}{'부트스트랩 문턱(평균)':>22}"
      f"{'카이제곱 거짓경보':>18}{'부트스트랩 거짓경보':>20}")
CHI_FA, BOOT_FA, BOOT_CRIT = [], [], []
crit_chi = stats.chi2.ppf(1 - ALPHA, N3 - 1) / (N3 - 1)
for k, (draw, _) in POPS.items():
    hit_c = hit_b = 0
    crits = []
    for _ in range(R3):
        base = draw(M3)                      # 과거 자료
        idx = RNG.integers(0, M3, (B, N3))   # 거기서 n 개씩 다시 뽑는다
        cb = float(np.quantile(base[idx].var(axis=1, ddof=1), 1 - ALPHA))
        crits.append(cb)
        s2 = draw(N3).var(ddof=1)            # 새 표본 — 아무 일도 없었다
        hit_c += s2 > crit_chi
        hit_b += s2 > cb
    CHI_FA.append(hit_c / R3)
    BOOT_FA.append(hit_b / R3)
    BOOT_CRIT.append(float(np.mean(crits)))
    print(f"{k:>8}{crit_chi:>14.4f}{BOOT_CRIT[-1]:>22.4f}"
          f"{CHI_FA[-1]:>18.4f}{BOOT_FA[-1]:>20.4f}")
print("   문턱을 자료에서 뽑으면 꼬리가 두꺼운 쪽에서도 자릿수가 돌아온다. "
      "과거 자료가 짧으면 그만큼 문턱도 흔들린다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {"균등": "#2ca02c", "정규": "#333333", "지수": "#ff7f0e",
       "로그정규": "#d62728"}

for k in POPS:
    ax1.semilogx(NS, FA[k], "o-", color=COL[k], lw=2.4, ms=8, label=k)
ax1.axhline(ALPHA, color="#1f77b4", lw=1.8, ls="--")
ax1.text(11, ALPHA * 1.25, f"이름값 ${100 * ALPHA:.0f}\\%$", color="#1f77b4",
         fontsize=10.5)
ax1.set_xlabel("표본크기 $n$")
ax1.set_ylabel("실제 거짓경보율")
ax1.set_xticks([10, 100])
ax1.set_xticklabels(["$10$", "$10^{2}$"])
ax1.minorticks_off()
ax1.set_title(f"① 아무 일도 없는데 울린다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ks = list(POPS)
xs = np.arange(len(ks))
w = 0.38
th50 = [(KUR[i] + 3 - 1 + 2 / 49) / 50 for i in range(len(ks))]
ax2.bar(xs - w / 2, V50, w, color="#555555", label="모의한 $\\operatorname{Var}(S^2)$")
ax2.bar(xs + w / 2, th50, w, color="#9ecae1",
        label="$(\\kappa-1+\\frac{2}{n-1})\\sigma^4/n$")
for i in range(len(ks)):
    ax2.text(xs[i], max(V50[i], th50[i]) * 1.08,
             f"$\\kappa-3 = {KUR[i]:.1f}$", ha="center", fontsize=9.5)
ax2.set_yscale("log")
ax2.set_xticks(xs)
ax2.set_xticklabels(ks, fontsize=10.5)
ax2.set_ylabel("$\\operatorname{Var}(S^2)$  ($n = 50$)")
ax2.set_yticks([1e-2, 1e-1, 1, 10])
ax2.set_yticklabels(["$10^{-2}$", "$10^{-1}$", "$1$", "$10$"])
ax2.minorticks_off()
ax2.set_ylim(1e-2, 60)
ax2.set_title("② 첨도가 그대로 들어온다", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

ax3.bar(xs - w / 2, CHI_FA, w, color="#d62728", label="카이제곱 문턱")
ax3.bar(xs + w / 2, BOOT_FA, w, color="#2ca02c", label="부트스트랩 문턱")
ax3.axhline(ALPHA, color="#1f77b4", lw=1.8, ls="--")
for i in range(len(ks)):
    ax3.text(xs[i] - w / 2, CHI_FA[i] + 0.002, f"${CHI_FA[i]:.3f}$",
             ha="center", fontsize=9.5)
    ax3.text(xs[i] + w / 2, BOOT_FA[i] + 0.002, f"${BOOT_FA[i]:.3f}$",
             ha="center", fontsize=9.5)
ax3.set_xticks(xs)
ax3.set_xticklabels(ks, fontsize=10.5)
ax3.set_ylabel("실제 거짓경보율")
ax3.set_title(f"③ 과거 {M3} 개로 문턱을 잡으면", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/variance_alarm.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
