r"""6.3절 「그림으로 보는 통계학」 — 사례 170 건이 정한 수.

$2020$ 년 겨울, 한 코로나 백신 시험의 결론은 **확진 $170$ 건**에서 나왔다. 백신군
$8$ 건, 위약군 $162$ 건. 배정이 반반이면 「확진자 한 명이 백신군에서 나왔을 확률」
$\theta$ 는 베르누이 모수 하나이고, 효능은 그 함수다.

    VE = 1 - θ/(1-θ)

이 쪽의 MLE 와 불변성이 그대로 쓰인다 — $\hat\theta = 8/170$ 을 식에 넣으면 끝이다.

① 베르누이 가능도와 그 꼭대기.
② 불변성 — θ 의 구간을 그대로 옮기면 효능의 구간이 된다.
③ 왜 사례 170 건이었나 — 건수에 따라 구간이 좁아지는 모습.

만드는 파일: docs/ch06/mle/img/vaccine_efficacy.png
실행:  python3 scripts/make_ch06_vaccine_efficacy.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats

OUT = "docs/ch06/mle/img/vaccine_efficacy.png"
K, T = 8, 170                 # 백신군 사례, 전체 사례
ve = lambda th: 1 - th / (1 - th)


def cp(k, n, a=0.05):
    """클로퍼–피어슨 구간."""
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


th_hat = K / T
lo, hi = cp(K, T)
print(f"① θ̂ = {K}/{T} = {th_hat:.5f},  95% 구간 [{lo:.5f}, {hi:.5f}]")
print(f"② 효능 = {100*ve(th_hat):.2f}%,  구간 [{100*ve(hi):.2f}%, {100*ve(lo):.2f}%]")
print(f"   (논문이 보고한 값: 95.0%, 신용구간 90.3 ~ 97.6)")

# --- ③ 사례 건수를 바꾸면 ----------------------------------------------
TS = np.unique(np.round(np.logspace(np.log10(20), np.log10(600), 40)).astype(int))
widths, los = [], []
for t in TS:
    k = max(int(round(th_hat * t)), 0)
    a, b = cp(k, t)
    widths.append(100 * (ve(a) - ve(b)))
    los.append(100 * ve(b))
    if t in (40, 80, 170, 340):
        print(f"③ 사례 {t:>4} 건 (백신군 {k:>3}):  효능 구간 "
              f"[{100*ve(b):5.1f}%, {100*ve(a):5.1f}%]  폭 {widths[-1]:5.1f}")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
g = np.linspace(0.005, 0.16, 500)
ll = K * np.log(g) + (T - K) * np.log(1 - g)
ax.plot(g, np.exp(ll - ll.max()), color=C1, lw=2.2)
ax.axvline(th_hat, color=C2, lw=1.8)
ax.axvspan(lo, hi, color=C2, alpha=0.15)
ax.set_xlabel(r"$\theta$ — 확진자 한 명이 백신군에서 나올 확률")
ax.set_ylabel("가능도 (꼭대기를 1 로)")
ax.set_title(f"① 베르누이 모수 하나 ({K}/{T})", fontsize=11)
ax.text(th_hat + 0.004, 0.75, f"$\\hat\\theta$ = {th_hat:.4f}\n"
        f"95% 구간\n[{lo:.4f}, {hi:.4f}]", color=C2, fontsize=9)

ax = axes[1]
ax.plot(g, 100 * ve(g), color=C1, lw=2.2)
ax.axvline(th_hat, color=C2, lw=1.2, ls=":")
ax.axhline(100 * ve(th_hat), color=C2, lw=1.2, ls=":")
ax.axvspan(lo, hi, color=C2, alpha=0.15)
ax.axhspan(100 * ve(hi), 100 * ve(lo), color=C2, alpha=0.15)
ax.plot([th_hat], [100 * ve(th_hat)], "o", color=C2, ms=8)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("백신 효능 (%)")
ax.set_ylim(30, 100)
ax.set_title("② 불변성 — 구간이 그대로 옮겨 간다", fontsize=11)
ax.text(0.055, 55, f"효능 {100*ve(th_hat):.1f}%\n"
        f"[{100*ve(hi):.1f}%, {100*ve(lo):.1f}%]\n"
        f"논문: 95.0%\n[90.3%, 97.6%]", color=C2, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[2]
ax.plot(TS, los, color=C1, lw=2.2, label="구간의 아래끝")
ax.fill_between(TS, los, 100, color=C1, alpha=0.12)
ax.axhline(30, color=C2, lw=1.6, ls="--")
ax.text(TS[0] * 1.05, 31.5, "넘어야 할 문턱 — 효능 30%", color=C2, fontsize=9)
ax.axvline(T, color="0.35", lw=1.3, ls=":")
ax.text(T * 1.06, 45, f"사례 {T} 건", color="0.35", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("모인 확진 사례 수 (백신군 비율은 같다고 두고)")
ax.set_ylabel("효능 구간의 아래끝 (%)")
ax.set_ylim(20, 100)
ax.set_title("③ 왜 사람 수가 아니라 사례 수인가", fontsize=11)
ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
