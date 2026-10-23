r"""평균이 같아도 꼬리가 다르면 다른 시스템이다 — 분산비를 묻게 되는 자리.

이 쪽은 $\sigma_1^2/\sigma_2^2$ 의 $F$ 구간을 세운다. 두 시스템의 **평균**이 아니라
**흩어짐**을 견주어야 하는 자리가 어디인지를 보이고, 그 다음 이 구간이 거기서 쓸 수
있는 물건인지를 본다.

① 딘과 바로소의 수를 다시 계산한다 — 서버 하나가 $100$ 번에 한 번 느리면, 서버
   $100$ 개에 퍼뜨린 요청의 $63\%$ 가 느려진다. $1-(1-p)^n$ 이다.
② 평균이 똑같은 두 시스템이 꼬리에서 갈린다.
③ 그런데 지연시간 같은 치우친 자료에서 $F$ 구간의 실제 포함률은 $95\%$ 가 아니다.

만드는 파일: docs/ch08/two_sample_intervals/img/tail_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
REP = 40_000

# === ① 퍼뜨릴수록 꼬리가 커진다 ======================================
NS = np.arange(1, 2001)
RATES = [(0.01, "$100$ 번에 한 번"), (0.001, "$1{,}000$ 번에 한 번"),
         (0.0001, "$10{,}000$ 번에 한 번")]
print("① 서버 하나가 느릴 확률 p, 서버 n 개에 퍼뜨린 요청이 느릴 확률 1-(1-p)^n")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("        p      n=1     n=10    n=100   n=2000")
CURVES = []
for p, _ in RATES:
    y = 1 - (1 - p) ** NS
    CURVES.append(y)
    print(f"   {p:>7.4f}  {p:>7.4f}  {1 - (1 - p) ** 10:>7.4f}  "
          f"{1 - (1 - p) ** 100:>7.4f}  {1 - (1 - p) ** 2000:>7.4f}")
print(f"   p = 0.01, n = 100 → {1 - 0.99 ** 100:.4f} — 논문의 **63%** 가 이 수다.")
print(f"   p = 0.0001, n = 2000 → {1 - 0.9999 ** 2000:.4f} — "
      f"**다섯에 하나 가까이**가 이 수다.")

# === ② 평균이 같은 두 시스템 =========================================
MEAN = 10.0
print(f"\n② 평균 응답시간이 {MEAN}ms 로 같은 두 시스템")
print("   시스템     표준편차     p99      p999   서버 100 개에 퍼뜨리면")
SYS = []
for name, cv in (("고른 쪽", 0.3), ("꼬리 긴 쪽", 1.5)):
    # 평균을 MEAN 으로 맞춘 로그정규
    s2 = np.log(1 + cv ** 2)
    mu = np.log(MEAN) - s2 / 2
    x = RNG.lognormal(mu, np.sqrt(s2), 400_000)
    p99 = float(np.percentile(x, 99))
    p999 = float(np.percentile(x, 99.9))
    thr = 100.0            # 100ms 를 '느리다'로 본다
    p_slow = float(np.mean(x > thr))
    fan = 1 - (1 - p_slow) ** 100
    SYS.append((name, cv, x.std(), p99, p999, p_slow, fan))
    print(f"   {name:<10} {x.std():>8.2f}  {p99:>7.2f}  {p999:>8.2f}   "
          f"{fan:>8.4f}")
print(f"   평균은 둘 다 {MEAN}ms 인데 100 개에 퍼뜨린 요청이 느려질 확률이 "
      f"{SYS[0][6]:.4f} 대 {SYS[1][6]:.4f} 다.")
print("   **평균을 견주는 검정은 이 차이를 전혀 보지 못한다.**")

# === ③ F 구간은 여기서 쓸 수 있는가 ==================================
NS3 = [10, 20, 50, 100, 200]
print(f"\n③ sigma1^2/sigma2^2 = 1 일 때 95% F 구간의 실제 포함률")
print("        n     정규     로그정규(cv=0.3)   로그정규(cv=1.5)")
COV = {k: [] for k in ("정규", "cv0.3", "cv1.5")}
for n in NS3:
    row = []
    for key, gen in (
        ("정규", lambda sz: RNG.normal(10, 3, sz)),
        ("cv0.3", lambda sz: RNG.lognormal(
            np.log(10) - np.log(1 + 0.3 ** 2) / 2,
            np.sqrt(np.log(1 + 0.3 ** 2)), sz)),
        ("cv1.5", lambda sz: RNG.lognormal(
            np.log(10) - np.log(1 + 1.5 ** 2) / 2,
            np.sqrt(np.log(1 + 1.5 ** 2)), sz)),
    ):
        a, b = gen((REP, n)), gen((REP, n))
        f = a.var(1, ddof=1) / b.var(1, ddof=1)
        lo = f / stats.f.ppf(0.975, n - 1, n - 1)
        hi = f / stats.f.ppf(0.025, n - 1, n - 1)
        c = float(np.mean((lo <= 1.0) & (1.0 <= hi)))
        COV[key].append(c)
        row.append(c)
    print(f"   {n:>6}   {row[0]:>6.4f}   {row[1]:>14.4f}   {row[2]:>14.4f}")
print("   정규에서는 0.95 를 지키지만 치우친 자료에서는 무너진다.")
print("   **n 을 키워도 돌아오지 않는다** — 이름값이 기대는 것은 표본 수가 아니라 모양이다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C = ["#d62728", "#ff7f0e", "#1f77b4"]

for (p, lab), y, c in zip(RATES, CURVES, C):
    ax1.plot(NS, y, lw=2.6, color=c, label=lab)
ax1.plot([100], [1 - 0.99 ** 100], "X", color="#333333", ms=15, zorder=5)
ax1.annotate(f"서버 $100$ 개\n${100 * (1 - 0.99 ** 100):.0f}\\%$",
             (100, 1 - 0.99 ** 100), xytext=(1.2, -0.4),
             textcoords="offset fontsize", fontsize=10.5, color="#333333")
ax1.plot([2000], [1 - 0.9999 ** 2000], "o", color="#333333", ms=13, zorder=5)
ax1.annotate(f"${100 * (1 - 0.9999 ** 2000):.0f}\\%$",
             (2000, 1 - 0.9999 ** 2000), xytext=(-2.6, 0.4),
             textcoords="offset fontsize", fontsize=10.5, color="#333333")
ax1.set_xlabel("요청 하나가 거치는 서버 수")
ax1.set_ylabel("응답이 $1$ 초를 넘을 확률")
ax1.set_ylim(0, 1.0)
ax1.set_title("① 느린 서버 하나가 전체를 끈다", fontsize=12)
ax1.legend(loc="center right", fontsize=10, title="서버 하나가 느린 빈도")
ax1.grid(alpha=0.25)

xs = np.linspace(0, 60, 500)
for (name, cv, sd, p99, p999, ps, fan), c in zip(SYS, ["#2ca02c", "#d62728"]):
    s2 = np.log(1 + cv ** 2)
    mu = np.log(MEAN) - s2 / 2
    ax2.plot(xs, stats.lognorm.pdf(xs, np.sqrt(s2), scale=np.exp(mu)),
             lw=2.8, color=c, label=f"{name} ($\\sigma = {sd:.1f}$, p99 $= {p99:.0f}$ms)")
ax2.axvline(MEAN, color="#333333", lw=2.0, ls="--")
ax2.text(MEAN + 1.5, ax2.get_ylim()[1] * 0.42,
         f"평균은 둘 다 ${MEAN:.0f}$ms", fontsize=10.5, color="#333333")
ax2.set_xlabel("응답시간 (ms)")
ax2.set_ylabel("밀도")
ax2.set_title("② 같은 평균, 다른 꼬리", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

for key, lab, c in (("정규", "정규", "#2ca02c"),
                    ("cv0.3", "로그정규 (약한 치우침)", "#ff7f0e"),
                    ("cv1.5", "로그정규 (강한 치우침)", "#d62728")):
    ax3.semilogx(NS3, COV[key], "o-", color=c, lw=2.6, ms=9, label=lab)
ax3.axhline(0.95, color="#333333", lw=1.8, ls="--")
ax3.text(10.4, 1.005, "이름값 $95\\%$", fontsize=10.5, color="#333333")
ax3.set_xlabel("두 표본의 크기 $n$")
ax3.set_ylabel("$F$ 구간의 실제 포함률")
ax3.set_xticks([10, 100])
ax3.set_xticklabels(["$10$", "$10^{2}$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.09)
ax3.set_title("③ 정작 그 자료에서는 쓸 수 없다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/two_sample_intervals/img/tail_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
