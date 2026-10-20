r"""충분성을 눈으로 — 같은 $(\bar x, s^2)$ 면 가능도가 포개지고, 같은 (중앙값, IQR) 이면 갈라진다.

위: 모양이 전혀 다른 세 표본이 $(\bar x, s^2)$ 만 같으면 가능도함수가 **한 줄**이 된다.
아래: 중앙값과 IQR 만 같은 세 표본은 가능도가 따로 논다.

만드는 파일: docs/ch06/estimator_quality/img/sufficiency.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N = 21
MU, SD = 50.0, 10.0
COL = ["#1f77b4", "#d62728", "#2ca02c"]


def standardize(v, mu=MU, sd=SD):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=0) * sd + mu


# === 위: (x̄, s²) 만 같은 세 표본 ===================================
shapes = [
    RNG.normal(0, 1, N),  # 평범한 모양
    np.concatenate([RNG.normal(-1.2, 0.25, N // 2 + 1), RNG.normal(1.2, 0.25, N // 2)]),
    np.concatenate([RNG.normal(-0.25, 0.3, N - 2), [2.5, 3.2]]),  # 한쪽에 치우침
]
top = [standardize(s) for s in shapes]

# === 아래: (중앙값, IQR) 만 같은 세 표본 ============================
# n = 21 이면 np.percentile 이 25/50/75 에서 정확히 6·11·16 번째 값을 집는다.
Q1, MED, Q3 = 45.0, 50.0, 55.0


def build(low, mid, high):
    """바깥 값들만 바꾸고 6·11·16 번째는 못박는다."""
    v = np.empty(N)
    v[5], v[10], v[15] = Q1, MED, Q3
    v[:5] = np.sort(low)
    v[6:10] = np.sort(mid[:4])
    v[11:15] = np.sort(mid[4:])
    v[16:] = np.sort(high)
    return v


bot = [
    build(np.linspace(41, 44.5, 5), np.linspace(45.5, 54.5, 8), np.linspace(55.5, 59, 5)),
    build(np.linspace(43.5, 44.8, 5), np.linspace(45.5, 54.5, 8), np.linspace(58, 86, 5)),
    build(np.linspace(14, 42, 5), np.linspace(45.5, 54.5, 8), np.linspace(55.2, 56.5, 5)),
]


def profile_loglik(x, mus):
    """sigma 를 프로파일로 없앤 로그가능도. (x̄, s²) 에만 의존한다."""
    n = len(x)
    out = []
    for m in mus:
        s2 = np.mean((x - m) ** 2)
        out.append(-n / 2 * np.log(s2))
    out = np.array(out)
    return out - out.max()


mus = np.linspace(36, 64, 500)

# === 그림 ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4))
(axA, axB), (axC, axD) = axes

for i, (v, c) in enumerate(zip(top, COL)):
    axA.plot(v, np.full(N, 2 - i), "o", color=c, ms=7, alpha=0.8)
    axA.text(
        6.0,
        2 - i,
        f"$\\bar x = {v.mean():.1f}$,  $s = {v.std(ddof=0):.1f}$",
        va="center",
        fontsize=10,
        color=c,
    )
axA.set_yticks([2, 1, 0])
axA.set_yticklabels(["표본 가", "표본 나", "표본 다"], fontsize=11)
axA.set_xlim(4, 92)
axA.set_ylim(-0.6, 2.95)
axA.set_title("① 모양은 다르고 $(\\bar x,\\, s^2)$ 는 같다", fontsize=12)
axA.grid(alpha=0.25, axis="x")

for v, c, lw in zip(top, COL, [6.5, 3.6, 1.6]):
    axB.plot(mus, profile_loglik(v, mus), color=c, lw=lw, alpha=0.85)
axB.set_xlabel("$\\mu$")
axB.set_ylabel("프로파일 로그가능도")
axB.set_title("② 가능도함수가 완전히 포개진다", fontsize=12)
axB.set_xlim(36, 64)
axB.set_ylim(-26, 2)
axB.grid(alpha=0.25)
axB.text(
    50,
    -22,
    "세 곡선이 겹쳐 있다 (굵기만 다르게 그렸다)",
    ha="center",
    fontsize=10,
    color="#555555",
)

for i, (v, c) in enumerate(zip(bot, COL)):
    axC.plot(v, np.full(N, 2 - i), "o", color=c, ms=7, alpha=0.8)
    axC.text(
        5.5,
        2 - i + 0.26,
        f"$\\bar x = {v.mean():.1f}$",
        va="bottom",
        fontsize=10,
        color=c,
    )
for q, lab in [(Q1, "$Q_1$"), (MED, "중앙값"), (Q3, "$Q_3$")]:
    axC.axvline(q, color="#777777", lw=1.0, ls=":")
    axC.text(q, 2.62, lab, ha="center", fontsize=9.5, color="#555555")
axC.set_yticks([2, 1, 0])
axC.set_yticklabels(["표본 라", "표본 마", "표본 바"], fontsize=11)
axC.set_xlim(4, 92)
axC.set_ylim(-0.6, 2.95)
axC.set_xlabel("값")
axC.set_title("③ 중앙값과 IQR 는 같고 나머지는 다르다", fontsize=12)
axC.grid(alpha=0.25, axis="x")

for v, c in zip(bot, COL):
    axD.plot(mus, profile_loglik(v, mus), color=c, lw=2.6, alpha=0.9)
    m = v.mean()
    axD.plot([m], [profile_loglik(v, [m])[0]], "o", color=c, ms=7, zorder=5)
axD.set_xlabel("$\\mu$")
axD.set_ylabel("프로파일 로그가능도")
axD.set_title("④ 가능도함수가 따로 논다", fontsize=12)
axD.set_xlim(36, 64)
axD.set_ylim(-26, 2)
axD.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/estimator_quality/img/sufficiency.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① (x̄, s²) 만 같은 세 표본")
for i, v in enumerate(top):
    print(
        f"  표본 {'가나다'[i]}: x̄ = {v.mean():.4f}, s = {v.std(ddof=0):.4f}, "
        f"중앙값 = {np.median(v):6.2f}, IQR = {np.percentile(v, 75) - np.percentile(v, 25):5.2f}"
    )
print("  → 세 프로파일 로그가능도의 최대 차이:", end=" ")
L = np.array([profile_loglik(v, mus) for v in top])
print(f"{np.abs(L - L[0]).max():.2e}")

print("\n③ (중앙값, IQR) 만 같은 세 표본")
for i, v in enumerate(bot):
    print(
        f"  표본 {'라마바'[i]}: 중앙값 = {np.median(v):.2f}, "
        f"IQR = {np.percentile(v, 75) - np.percentile(v, 25):.2f},  "
        f"x̄ = {v.mean():6.2f}, s = {v.std(ddof=0):5.2f}"
    )
print(f"  → 평균이 {min(v.mean() for v in bot):.2f} 에서 {max(v.mean() for v in bot):.2f} 까지 벌어진다")

# === 충분하지 않은 요약으로 치르는 값 ================================
print("\n충분하지 않은 요약을 쓰면 — 상대효율 (모의실험 40,000 회)")
REP = 40000
print(f"{'n':>6}{'중앙값 / 평균':>16}{'IQR 기반 σ̂ / MLE σ̂':>24}")
for n in (10, 30, 100, 400):
    x = RNG.normal(0, 1, (REP, n))
    e_mean = 1 / np.var(x.mean(axis=1), ddof=1)
    e_med = 1 / np.var(np.median(x, axis=1), ddof=1)
    s_mle = x.std(axis=1, ddof=0)
    s_iqr = (np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)) / 1.3490
    print(
        f"{n:>6}{e_med / e_mean:>15.4f}{np.var(s_mle, ddof=1) / np.var(s_iqr, ddof=1):>22.4f}"
    )
print(f"  이론값: 중앙값 2/π = {2 / np.pi:.4f}")
