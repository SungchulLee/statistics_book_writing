r"""전역 $F$ 가 기각한 뒤에 남는 세 개의 물음, 그리고 그중 하나의 답이 「모른다」인 것.

테네시 STAR 처럼 집단이 셋인 설계를 본뜬다 — 작은 반, 정규반, 보조교사를 둔 정규반.
참 평균을 $(0.2,\ 0,\ 0)$ 으로 두면, 세 쌍 가운데 **둘은 참 차이가 $0.2\sigma$ 이고
하나는 정확히 $0$** 이다.

① 표본이 커질수록 전역 $F$ 와 두 쌍이 올라간다. 참 차이가 $0$ 인 쌍은 $0.05$ 에 머문다.
② 그런데 그 $0.05$ 를 「같다」로 읽으면 안 된다 — 비유의는 $0$ 과 작은 차이를
   구별하지 못한다.
③ 「$0.1\sigma$ 안」이라고 **말하려면** 차이를 잡을 때보다 훨씬 많은 표본이 필요하다.

만드는 파일: docs/ch13/post_hoc/img/star_posthoc.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

MU = np.array([0.20, 0.0, 0.0])        # 작은 반, 정규반, 보조교사
NAMES = ["작은 반", "정규반", "보조교사"]
PAIRS = [(0, 1), (0, 2), (1, 2)]
NS = np.array([20, 40, 80, 150, 300, 600])
REPS = 4000
ALPHA = 0.05
K = len(MU)


def tukey_crit(k, nu, alpha=ALPHA):
    """투키 임계값을 쌍별 t 눈금으로 옮긴 것."""
    return stats.studentized_range.isf(alpha, k, nu) / np.sqrt(2)


print(f"참 평균 {MU} (표준편차 1), 집단 {K} 개, {REPS} 회")
print(f"{'집단마다 n':>10}{'전역 F':>9}"
      + "".join(f"{f'{NAMES[a]}-{NAMES[b]}':>16}" for a, b in PAIRS))
glob, pw = [], {p: [] for p in PAIRS}
for n in NS:
    n = int(n)
    nu = K * (n - 1)
    crit = tukey_crit(K, nu)
    se = np.sqrt(2.0 / n)
    g = RNG.normal(MU[:, None, None], 1.0, (K, REPS, n))
    m = g.mean(axis=2)
    msw = ((g - m[:, :, None]) ** 2).sum(axis=(0, 2)) / nu
    f = stats.f_oneway(*[g[i] for i in range(K)], axis=1)
    glob.append(float((f.pvalue < ALPHA).mean()))
    row = f"{n:>10}{glob[-1]:>9.3f}"
    for a, b in PAIRS:
        t = np.abs(m[a] - m[b]) / (np.sqrt(msw) * se)
        pw[(a, b)].append(float((t > crit).mean()))
        row += f"{pw[(a, b)][-1]:>16.3f}"
    print(row)
glob = np.array(glob)

# === ② 비유의가 뜻하는 것 ===========================================
NFIX = 150
DELTAS = np.linspace(0.0, 0.45, 16)
nu = K * (NFIX - 1)
crit = tukey_crit(K, nu)
se = np.sqrt(2.0 / NFIX)
print(f"\n② 집단마다 {NFIX} 명일 때, 참 차이 δ 인 쌍이 유의하지 않을 확률")
nonsig = []
for d in DELTAS:
    a = RNG.normal(0.0, 1.0, (REPS, NFIX))
    b = RNG.normal(d, 1.0, (REPS, NFIX))
    c = RNG.normal(0.0, 1.0, (REPS, NFIX))
    msw = (((a - a.mean(1, keepdims=True)) ** 2).sum(1)
           + ((b - b.mean(1, keepdims=True)) ** 2).sum(1)
           + ((c - c.mean(1, keepdims=True)) ** 2).sum(1)) / nu
    t = np.abs(b.mean(1) - a.mean(1)) / (np.sqrt(msw) * se)
    nonsig.append(float((t <= crit).mean()))
nonsig = np.array(nonsig)
for d, p in zip(DELTAS, nonsig):
    if np.isclose(d, 0.0) or np.isclose(d, DELTAS[3]) \
            or np.isclose(d, DELTAS[6]) or np.isclose(d, DELTAS[10]):
        print(f"   δ = {d:.2f} -> {p:.3f}")

# === ③ 차이를 잡는 일과 같음을 말하는 일 =============================
MARGIN = 0.10          # 「이만큼 안이면 같다고 보겠다」
EFFECT = 0.20
NG = np.arange(50, 3001, 50)


def power_detect(n, delta=EFFECT):
    """양측 t 검정으로 delta 를 잡아낼 확률 (보정 없이)."""
    nc = delta / np.sqrt(2.0 / n)
    c = stats.t.isf(ALPHA / 2, 2 * (n - 1))
    return float(stats.nct.sf(c, 2 * (n - 1), nc)
                 + stats.nct.cdf(-c, 2 * (n - 1), nc))


def power_equiv(n, margin=MARGIN):
    """참 차이가 0 일 때 TOST 로 「margin 안」이라고 말할 확률."""
    nc = margin / np.sqrt(2.0 / n)
    c = stats.t.isf(ALPHA, 2 * (n - 1))
    return float(max(0.0, stats.nct.cdf(nc - c, 2 * (n - 1), 0.0)
                     - stats.nct.cdf(c - nc, 2 * (n - 1), 0.0)))


pd_ = np.array([power_detect(int(n)) for n in NG])
pe_ = np.array([power_equiv(int(n)) for n in NG])
n_det = int(NG[np.argmax(pd_ >= 0.8)])
n_equ = int(NG[np.argmax(pe_ >= 0.8)])
print(f"\n③ 집단마다 필요한 표본 (검정력 0.8)")
print(f"   {EFFECT}σ 차이를 잡으려면      : {n_det}")
print(f"   {MARGIN}σ 안이라고 말하려면    : {n_equ}  ({n_equ / n_det:.1f} 배)")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = {(0, 1): "#1f77b4", (0, 2): "#2ca02c", (1, 2): "#d62728"}

ax1.semilogx(NS, glob, "k^--", lw=2.2, ms=8, label="전역 $F$ 검정")
STY = {(0, 1): ("o-", 3.4, 11), (0, 2): ("s--", 2.0, 7), (1, 2): ("D-", 2.6, 8)}
for (a, b) in PAIRS:
    fmt, lw, ms = STY[(a, b)]
    ax1.semilogx(NS, pw[(a, b)], fmt, color=COL[(a, b)], lw=lw, ms=ms,
                 alpha=0.9, label=f"{NAMES[a]} $-$ {NAMES[b]}")
ax1.axhline(ALPHA, color="black", lw=1.2, ls=":")
ax1.text(115, 0.085, "$\\alpha = 0.05$", fontsize=10)
ax1.set_xticks(NS)
ax1.set_xticklabels([f"${v}$" for v in NS], fontsize=9.5)
ax1.minorticks_off()
ax1.set_xlabel("집단마다의 인원")
ax1.set_ylabel("기각할 확률")
ax1.set_ylim(0, 1.05)
ax1.set_title("① 참 차이가 $0$ 인 쌍은 올라가지 않는다", fontsize=12)
ax1.legend(loc="center left", fontsize=9.5)
ax1.grid(alpha=0.25)

ax2.plot(DELTAS, nonsig, "o-", color="#d62728", lw=2.8, ms=7)
ax2.axhline(0.5, color="#999999", lw=1.2, ls=":")
for d in (0.0, 0.10):
    i = int(np.argmin(np.abs(DELTAS - d)))
    ax2.plot(DELTAS[i], nonsig[i], "*", color="black", ms=18, zorder=5)
    ax2.annotate(f"$\\delta = {d:.2f}$\n${nonsig[i]:.2f}$",
                 xy=(DELTAS[i], nonsig[i]),
                 xytext=(DELTAS[i] + 0.035, nonsig[i] - 0.17), fontsize=10.5,
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax2.set_xlabel("그 쌍의 참 차이 $\\delta$ (표준편차 단위)")
ax2.set_ylabel("투키 검정이 유의하지 않을 확률")
ax2.set_ylim(0, 1.05)
ax2.set_title(f"② 집단마다 ${NFIX}$ 명일 때", fontsize=12)
ax2.grid(alpha=0.25)

ax3.plot(NG, pd_, color="#1f77b4", lw=2.8,
         label=f"${EFFECT}\\sigma$ 차이를 잡기")
ax3.plot(NG, pe_, color="#d62728", lw=2.8,
         label=f"${MARGIN}\\sigma$ 안이라고 말하기")
ax3.axhline(0.8, color="black", lw=1.3, ls=":")
ax3.text(2450, 0.825, "검정력 $0.8$", fontsize=10, ha="right")
for n, c in ((n_det, "#1f77b4"), (n_equ, "#d62728")):
    ax3.axvline(n, color=c, lw=1.4, ls="-.")
    ax3.text(n + 55, 0.47 if c == "#d62728" else 0.28, f"${n}$", color=c,
             fontsize=11)
ax3.set_xlabel("집단마다의 인원")
ax3.set_ylabel("검정력")
ax3.set_ylim(0, 1.03)
ax3.set_title(f"③ 같다고 말하는 쪽이 ${n_equ / n_det:.1f}$ 배 비싸다", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/post_hoc/img/star_posthoc.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
