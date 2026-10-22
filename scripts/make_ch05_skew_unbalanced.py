r"""치우침은 균형이 맞을 때 상쇄된다 — 어긋나면 드러난다.

이 쪽은 두 평균의 차에서 중심극한정리가 비정규성을 지워 준다고 적는다. 맞는데,
**얼마나 빨리** 지워지는지는 두 표본이 얼마나 닮았느냐에 달려 있다.

① 크기도 치우침도 같으면 — 1종오류가 거의 지켜진다. 치우침이 상쇄된다.
② 크기가 어긋나면 무너진다 — A/B 시험의 트래픽 분할이 그 자리다.
③ 비를 그대로 두고 전체를 키워도 느리게만 돌아온다.

만드는 파일: docs/ch05/applications/img/skew_unbalanced.png
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
REP = 100_000


def draw_skew(shape, size):
    """감마(shape). 왜도는 2/sqrt(shape) 이고 평균 0, 분산 1 로 맞춘다."""
    g = RNG.gamma(shape, 1.0, size)
    return (g - shape) / np.sqrt(shape)


def welch_p(a, b):
    n1, n2 = a.shape[1], b.shape[1]
    m1, m2 = a.mean(1), b.mean(1)
    v1, v2 = a.var(1, ddof=1), b.var(1, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1)
                                     + (v2 / n2) ** 2 / (n2 - 1))
    return t, 2 * stats.t.sf(np.abs(t), df), stats.t.sf(t, df)


SHAPES = [(0.05, "왜도 $8.9$"), (0.2, "왜도 $4.5$"), (1.0, "왜도 $2.0$"),
          (10.0, "왜도 $0.6$")]
for sh, nm in SHAPES:
    print(f"   감마({sh}) — 왜도 {2 / np.sqrt(sh):.2f}")

# === ① 균형이 맞으면 =================================================
NS = [10, 20, 50, 100, 200, 500]
print(f"\n① 두 표본의 크기와 치우침이 같을 때 — 양측 {100 * ALPHA:.0f}% 검정의 "
      f"1종오류 (반복 {REP:,})")
print(f"{'n (양쪽 같음)':>14}" + "".join(f"{nm.replace('$',''):>14}"
                                     for _, nm in SHAPES))
BAL = {nm: [] for _, nm in SHAPES}
for n in NS:
    row = []
    for sh, nm in SHAPES:
        a = draw_skew(sh, (REP, n))
        b = draw_skew(sh, (REP, n))
        _, p2, _ = welch_p(a, b)
        BAL[nm].append(float(np.mean(p2 < ALPHA)))
        row.append(BAL[nm][-1])
    print(f"{n:>14}" + "".join(f"{v:>14.4f}" for v in row))
print("   왜도가 8.9 여도 n = 20 부터 0.05 에 가깝다. "
      "**같은 모양 둘을 빼면 치우침이 서로 지워진다.**")

# === ② 크기가 어긋나면 ===============================================
RATIOS = [(1, 1), (2, 1), (5, 1), (9, 1), (19, 1)]
NTOT = 400
SH2 = 0.05
print(f"\n② 전체 {NTOT} 을 나누는 비를 바꾸며 (감마 {SH2}, 왜도 "
      f"{2 / np.sqrt(SH2):.1f})")
print(f"{'비 n1:n2':>10}{'n1':>6}{'n2':>6}{'양측 1종오류':>14}"
      f"{'위쪽 꼬리':>12}{'아래쪽 꼬리':>13}")
UNB, UP, DN, LBL2 = [], [], [], []
for r1, r2 in RATIOS:
    n1 = int(NTOT * r1 / (r1 + r2))
    n2 = NTOT - n1
    a = draw_skew(SH2, (REP, n1))
    b = draw_skew(SH2, (REP, n2))
    _, p2, p1 = welch_p(a, b)
    UNB.append(float(np.mean(p2 < ALPHA)))
    UP.append(float(np.mean(p1 < ALPHA / 2)))
    DN.append(float(np.mean(p1 > 1 - ALPHA / 2)))
    LBL2.append(f"${r1}:{r2}$")
    print(f"{f'{r1}:{r2}':>10}{n1:>6}{n2:>6}{UNB[-1]:>14.4f}"
          f"{UP[-1]:>12.4f}{DN[-1]:>13.4f}")
print(f"   전체 표본 수는 {NTOT} 으로 같다. 나누는 비만 바꿨는데 "
      f"1종오류가 {min(UNB):.3f} 에서 {max(UNB):.3f} 까지 간다.")

# === ③ 비를 그대로 두고 전체를 키우면 ================================
TOTS = [100, 400, 1600, 6400, 25600]
RATS = [(1, 1), (5, 1), (19, 1)]
print(f"\n③ 비를 그대로 두고 전체 표본을 키우면 (감마 {SH2}, 왜도 "
      f"{2 / np.sqrt(SH2):.1f})")
print(f"{'전체 n':>9}" + "".join(f"{f'{r1}:{r2}':>12}" for r1, r2 in RATS))
GROW = {f"{r1}:{r2}": [] for r1, r2 in RATS}
for tot in TOTS:
    row = []
    rep = REP if tot <= 1600 else 20_000
    for r1, r2 in RATS:
        n1 = int(tot * r1 / (r1 + r2))
        n2 = tot - n1
        a_ = draw_skew(SH2, (rep, n1))
        b_ = draw_skew(SH2, (rep, n2))
        _, p2, _ = welch_p(a_, b_)
        v = float(np.mean(p2 < ALPHA))
        GROW[f"{r1}:{r2}"].append(v)
        row.append(v)
    print(f"{tot:>9,}" + "".join(f"{v:>12.4f}" for v in row))
print("   1:1 은 금세 0.05 로 간다. 19:1 은 전체를 25,600 까지 키워도 "
      "아직 돌아오지 않는다 — **CLT 가 구해 주기는 하는데 느리다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

for i, (sh, nm) in enumerate(SHAPES):
    ax1.semilogx(NS, BAL[nm], "o-", color=COL[i], lw=2.4, ms=8,
                 label=nm.replace("$", ""))
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax1.text(330, ALPHA * 0.90, "이름값 $0.05$", fontsize=10.5,
         color="#333333", ha="right", va="top")
ax1.set_xlabel("양쪽 표본크기 $n$")
ax1.set_ylabel("양측 검정의 1종오류")
ax1.set_xticks([10, 100])
ax1.set_xticklabels(["$10$", "$10^{2}$"])
ax1.minorticks_off()
ax1.set_title("① 균형이 맞으면 상쇄된다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25)

xs = np.arange(len(RATIOS))
ax2.plot(xs, UNB, "o-", color="#d62728", lw=2.8, ms=10, label="양측")
ax2.plot(xs, UP, "^--", color="#1f77b4", lw=2.2, ms=8, label="위쪽 꼬리")
ax2.plot(xs, DN, "v--", color="#2ca02c", lw=2.2, ms=8, label="아래쪽 꼬리")
ax2.axhline(ALPHA, color="#333333", lw=1.6, ls="--")
ax2.axhline(ALPHA / 2, color="#888888", lw=1.2, ls=":")
ax2.set_xticks(xs)
ax2.set_xticklabels(LBL2, fontsize=11)
ax2.set_xlabel(f"두 집단에 나눈 비 (전체 ${NTOT}$ 으로 같음)")
ax2.set_ylabel("1종오류")
ax2.set_title(f"② 비만 바꿔도 무너진다 (왜도 ${2 / np.sqrt(SH2):.1f}$)",
              fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

for i, (r1, r2) in enumerate(RATS):
    k = f"{r1}:{r2}"
    ax3.loglog(TOTS, GROW[k], "o-", color=COL[i], lw=2.6, ms=9,
               label=f"${r1}:{r2}$")
ax3.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax3.text(110, ALPHA * 1.1, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax3.set_xlabel("전체 표본크기")
ax3.set_ylabel("양측 검정의 1종오류")
ax3.set_xticks([100, 1000, 10000])
ax3.set_xticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$"])
ax3.set_yticks([0.01, 0.05, 0.1, 0.3])
ax3.set_yticklabels(["$0.01$", "$0.05$", "$0.1$", "$0.3$"])
ax3.minorticks_off()
ax3.set_title("③ 비를 그대로 두고 키우면", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/applications/img/skew_unbalanced.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
