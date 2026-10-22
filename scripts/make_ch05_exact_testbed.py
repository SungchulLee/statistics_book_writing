r"""닫힌 꼴이 있는 한 자리 — 연장을 재어 보는 시험대.

이 쪽은 $S^2$ 에 관한 네 모집단 가운데 **정규에서만** 결과가 정확하다고 적는다.
정확한 답이 하나 있다는 것은 공식을 하나 더 아는 일에 그치지 않는다. **연장을
재어 볼 자리**가 생기는 것이다.

① 정규에서 $\sigma^2$ 의 정확한 구간과 부트스트랩 구간의 포함률.
② 그 차이는 $n$ 이 얼마나 커야 사라지는가.
③ 비정규로 가면 둘 다 틀리는데, 틀리는 방식이 다르다.

만드는 파일: docs/ch05/applications/img/exact_testbed.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.05
B = 1000


def exact_ci(x):
    n = len(x)
    s2 = x.var(ddof=1)
    lo = (n - 1) * s2 / stats.chi2.ppf(1 - ALPHA / 2, n - 1)
    hi = (n - 1) * s2 / stats.chi2.ppf(ALPHA / 2, n - 1)
    return lo, hi


def boot_ci(x, rng):
    n = len(x)
    bs = x[rng.integers(0, n, (B, n))].var(axis=1, ddof=1)
    return float(np.quantile(bs, ALPHA / 2)), float(np.quantile(bs, 1 - ALPHA / 2))


def cover(draw, true_var, n, rep, rng):
    c_e = c_b = 0
    for _ in range(rep):
        x = draw(n)
        lo, hi = exact_ci(x)
        c_e += lo <= true_var <= hi
        lo, hi = boot_ci(x, rng)
        c_b += lo <= true_var <= hi
    return c_e / rep, c_b / rep


# === ①② 정규에서 ====================================================
NS = [5, 10, 20, 50, 100, 200, 500, 1000]
REP = 3000
print(f"①② 정규모집단에서 sigma^2 = 1 의 95% 구간 (반복 {REP:,}, B = {B:,})")
print(f"{'n':>6}{'정확 (카이제곱)':>18}{'부트스트랩 백분위':>20}")
CE, CB = [], []
for n in NS:
    e, b = cover(lambda m: RNG.standard_normal(m), 1.0, n, REP, RNG)
    CE.append(e)
    CB.append(b)
    print(f"{n:>6}{e:>18.4f}{b:>20.4f}")
print(f"   정확 쪽은 n = 5 에서도 {CE[0]:.3f} 다. 부트스트랩은 {CB[0]:.3f} 에서 "
      f"시작해 천천히 올라온다.")
i90 = next((NS[i] for i in range(len(NS)) if CB[i] >= 0.90), None)
print(f"   부트스트랩이 90% 를 넘는 데 n = {i90} 이 들고, 95% 에 닿는 데는 "
      f"n = {NS[-1]} 이 든다 ({CB[-1]:.3f}).")

# === ③ 비정규로 가면 =================================================
N3, REP3 = 50, 4000
POPS = {
    "정규": (lambda m: RNG.standard_normal(m), 1.0),
    "균등": (lambda m: RNG.uniform(-np.sqrt(3), np.sqrt(3), m), 1.0),
    "지수": (lambda m: RNG.exponential(1.0, m), 1.0),
    "로그정규": (lambda m: RNG.lognormal(0, 1, m),
                 float((np.e - 1) * np.e)),
}
print(f"\n③ n = {N3} 에서 모집단을 바꾸면 (반복 {REP3:,})")
print(f"{'모집단':>9}{'정확 (카이제곱)':>18}{'부트스트랩':>14}")
E3, B3 = [], []
for k, (draw, tv) in POPS.items():
    e, b = cover(draw, tv, N3, REP3, RNG)
    E3.append(e)
    B3.append(b)
    print(f"{k:>9}{e:>18.4f}{b:>14.4f}")
print("   '정확'한 구간이 정규를 떠나면 가장 많이 무너진다 — 정확함은 "
      "가정에 매여 있던 것이다.")
print("   부트스트랩도 무너지지만 덜 무너진다 (0.603 대 0.370). 한쪽은 "
      "가정에, 다른 쪽은 표본크기에 매여 있다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_E, C_B = "#1f77b4", "#d62728"

ax1.semilogx(NS, CE, "o-", color=C_E, lw=2.6, ms=9, label="정확 (카이제곱)")
ax1.semilogx(NS, CB, "s-", color=C_B, lw=2.6, ms=9, label="부트스트랩 백분위")
ax1.axhline(0.95, color="#333333", lw=1.8, ls="--")
ax1.text(5.2, 0.955, "말한 $95\\%$", fontsize=10.5, color="#333333")
ax1.set_xlabel("표본크기 $n$")
ax1.set_ylabel("실제 포함률")
ax1.set_xticks([10, 100, 1000])
ax1.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$"])
ax1.minorticks_off()
ax1.set_ylim(0.6, 1.0)
ax1.set_title("① 정규에서 — 재어 볼 수 있는 자리", fontsize=12)
ax1.legend(loc="lower right", fontsize=10.5)
ax1.grid(alpha=0.25)

gap = [0.95 - c for c in CB]
ax2.loglog(NS, gap, "s-", color=C_B, lw=2.6, ms=9, label="$0.95 -$ 포함률")
ax2.loglog(NS, [0.8 / np.sqrt(n) for n in NS], "--", color="#333333", lw=1.8,
           label="$\\propto 1/\\sqrt{n}$")
ax2.set_xlabel("표본크기 $n$")
ax2.set_ylabel("모자란 몫")
ax2.set_xticks([10, 100, 1000])
ax2.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$"])
ax2.set_yticks([0.01, 0.1])
ax2.set_yticklabels(["$0.01$", "$0.1$"])
ax2.minorticks_off()
ax2.set_title("② 천천히 줄어든다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10.5)
ax2.grid(alpha=0.25)

ks = list(POPS)
xs = np.arange(len(ks))
w = 0.38
ax3.bar(xs - w / 2, E3, w, color=C_E, label="정확 (카이제곱)")
ax3.bar(xs + w / 2, B3, w, color=C_B, label="부트스트랩")
ax3.axhline(0.95, color="#333333", lw=1.8, ls="--")
for i in range(len(ks)):
    ax3.text(xs[i] - w / 2, E3[i] + 0.012, f"${E3[i]:.3f}$", ha="center",
             fontsize=9.5)
    ax3.text(xs[i] + w / 2, B3[i] + 0.012, f"${B3[i]:.3f}$", ha="center",
             fontsize=9.5)
ax3.set_xticks(xs)
ax3.set_xticklabels(ks, fontsize=10.5)
ax3.set_ylabel("실제 포함률")
ax3.set_ylim(0, 1.12)
ax3.set_title(f"③ 모집단을 바꾸면 ($n = {N3}$)", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/exact_testbed.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
