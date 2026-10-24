r"""서로 **아무 관계도 없는** 두 계열을 회귀하면 거의 언제나 유의해진다.

추세가 있는 계열 — 특히 무작위 걷기 — 끼리는 상관계수와 $t$ 통계량이 제구실을
못 한다. 관측이 서로 독립이라는 가정이 깨졌는데 공식은 그대로 쓰기 때문이다.

① 독립인 무작위 걷기 두 개 — 눈으로 보면 관계가 있어 보인다.
② $n$ 이 커질수록 기각률이 $1$ 로 간다 (백색잡음은 $5\%$ 에 머문다).
③ 차분하면 돌아온다 — 그리고 높은 $R^2$ 에 낮은 더빈–왓슨이라는 자취.

만드는 파일: docs/ch10/confounding/img/spurious_regression.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
REPS = 4000
NS = np.array([20, 50, 100, 250, 500, 1000])


def ols_t(x, y):
    """기울기의 t 통계량, R^2, 더빈–왓슨."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (n - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - (r @ r) / ((y - y.mean()) ** 2).sum()
    dw = np.sum(np.diff(r) ** 2) / (r @ r)
    return b[1] / se, r2, dw


def gen(kind, n, rng):
    e = rng.standard_normal(n)
    return np.cumsum(e) if kind == "rw" else e


# === ②③ 모의실험 ==================================================
KINDS = [("rw", "무작위 걷기 둘", "#d62728"),
         ("wn", "독립 백색잡음 둘", "#1f77b4"),
         ("diff", "무작위 걷기를 차분한 것", "#2ca02c")]
rej = {k: [] for k, *_ in KINDS}
r2m = {k: [] for k, *_ in KINDS}
dwm = {k: [] for k, *_ in KINDS}
for n in NS:
    cur = {k: [[], [], []] for k, *_ in KINDS}
    for _ in range(REPS):
        xr, yr = gen("rw", n + 1, RNG), gen("rw", n + 1, RNG)
        pairs = {"rw": (xr[1:], yr[1:]),
                 "wn": (gen("wn", n, RNG), gen("wn", n, RNG)),
                 "diff": (np.diff(xr), np.diff(yr))}
        for k, (a, b) in pairs.items():
            t, r2, dw = ols_t(a, b)
            cur[k][0].append(abs(t) > stats.t.isf(0.025, len(a) - 2))
            cur[k][1].append(r2)
            cur[k][2].append(dw)
    for k, *_ in KINDS:
        rej[k].append(np.mean(cur[k][0]))
        r2m[k].append(np.mean(cur[k][1]))
        dwm[k].append(np.mean(cur[k][2]))
    print(f"  n = {n:>4}  " + "  ".join(
        f"{k} {rej[k][-1] * 100:5.1f}%" for k, *_ in KINDS))

# === ① 한 판 =======================================================
for seed in range(1, 300):
    g = np.random.default_rng(seed)
    x1, y1 = gen("rw", 200, g), gen("rw", 200, g)
    t1, r21, dw1 = ols_t(x1, y1)
    if r21 > 0.7 and abs(t1) > 15:
        break
p1 = 2 * stats.t.sf(abs(t1), len(x1) - 2)
print(f"\n① 씨앗 {seed} — R^2 = {r21:.3f}, t = {t1:.1f}, p = {p1:.2e}, "
      f"더빈–왓슨 = {dw1:.3f}")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1b = ax1.twinx()
ax1.plot(x1, color="#d62728", lw=2.0, label="계열 $X$")
ax1b.plot(y1, color="#1f77b4", lw=2.0, label="계열 $Y$")
ax1.set_xlabel("시점")
ax1.set_ylabel("$X$", color="#d62728")
ax1b.set_ylabel("$Y$", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1b.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_title("① 두 계열은 서로 **완전히 독립**이다".replace("**", ""),
              fontsize=12)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
ax1.text(0.97, 0.97,
         f"$Y$ 를 $X$ 에 회귀하면\n$R^2 = {r21:.2f}$,  $t = {t1:.1f}$,\n"
         f"$p < 10^{{{int(np.floor(np.log10(p1)))}}}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.45", fc="#fff6f6", ec="#d62728"))
ax1.grid(alpha=0.25)

for k, lab, c in KINDS:
    ax2.semilogx(NS, np.array(rej[k]) * 100, "o-", color=c, lw=2.4, ms=6,
                 label=lab)
ax2.axhline(5, color="black", lw=1.5, ls="--")
ax2.text(NS[0] * 1.05, 8.5, "명목 $5\\%$", fontsize=10)
ax2.set_xlabel("관측 수 $n$")
ax2.set_ylabel("기울기가 $\\alpha = 0.05$ 로 유의한 비율 ($\\%$)")
ax2.set_title("② 자료를 더 모으면 더 나빠진다", fontsize=12)
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v}$" for v in NS])
ax2.minorticks_off()
ax2.set_ylim(0, 103)
ax2.legend(loc="center right", fontsize=10)
ax2.grid(alpha=0.25)

w = 0.26
xs_ = np.arange(len(NS))
for i, (k, lab, c) in enumerate(KINDS):
    ax3.bar(xs_ + (i - 1) * w, np.array(r2m[k]), w, color=c, label=lab)
ax3.set_xticks(xs_)
ax3.set_xticklabels([f"${v}$" for v in NS])
ax3.set_xlabel("관측 수 $n$")
ax3.set_ylabel("평균 $R^2$")
ax3.set_title("③ 설명력까지 그럴듯해진다", fontsize=12)
ax3.set_ylim(0, 0.55)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")
ax3.text(len(NS) - 0.5, 0.345,
         f"무작위 걷기의 더빈–왓슨\n$n = {NS[-1]}$ 에서 ${dwm['rw'][-1]:.3f}$ "
         f"(독립이면 $2$ 쯤)",
         ha="right", fontsize=10, color="#d62728",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

fig.tight_layout()
out = "docs/ch10/confounding/img/spurious_regression.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

print(f"\n반복 {REPS} 회")
print(f"{'n':>6}" + "".join(f"{lab:>18}" for _, lab, _ in KINDS))
for i, n in enumerate(NS):
    print(f"{n:>6}" + "".join(f"{rej[k][i] * 100:>17.1f}%" for k, *_ in KINDS))
print(f"\n평균 R^2 와 더빈–왓슨 (무작위 걷기)")
for i, n in enumerate(NS):
    print(f"  n = {n:>4}  R^2 = {r2m['rw'][i]:.3f}  DW = {dwm['rw'][i]:.3f}"
          f"   (백색잡음 R^2 = {r2m['wn'][i]:.3f}, DW = {dwm['wn'][i]:.3f})")
