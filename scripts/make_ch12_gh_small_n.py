r"""저자들이 붙여 둔 단서 — 작은 n 에서의 게임스-하월.

이 쪽은 분산이 다를 때 쓰는 쌍별 비교로 게임스-하월을 세운다. 그 방법을 내놓은
$1976$ 년 논문의 초록은 마지막 문장을 이렇게 닫는다 — 「n 이 셋부터인 불균등
표본에서는 이 통계량으로도 만족스러운 통제가 의심스럽다」.

그 단서를 수로 확인한다. 집단 넷, 평균은 모두 같고 분산과 표본크기는 제각각이다.

① 가장 작은 집단의 크기를 키우며 집단별 1종오류(FWER).
② 쌍 하나당 1종오류.
③ 이 쪽 절 6 의 대안 — 본페로니 보정 쌍별 웰치 — 과 견준다.

만드는 파일: docs/ch12/post_hoc/img/gh_small_n.png
"""

import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)

# studentized_range.ppf 는 자유도마다 수치적분을 한다. 자유도 20,000 개에
# 그대로 부르면 몇 시간이 걸리므로 격자에서 한 번 구해 보간한다.
_DF_GRID = np.concatenate([np.arange(2.0, 30.0, 0.25),
                           np.arange(30.0, 120.0, 2.0),
                           np.array([150.0, 200.0, 400.0, 1000.0, 1e6])])


_Q_CACHE = {}


def q_crit(k, df, alpha=0.05):
    """스튜던트화 범위의 상위 alpha 분위수 (격자 보간). 표는 한 번만 만든다."""
    key = (k, alpha)
    if key not in _Q_CACHE:
        _Q_CACHE[key] = stats.studentized_range.ppf(1 - alpha, k, _DF_GRID)
    tab = _Q_CACHE[key]
    return np.interp(np.clip(df, _DF_GRID[0], _DF_GRID[-1]), _DF_GRID, tab)
K = 4
SD = np.array([1.0, 1.0, 2.0, 3.0])      # 분산이 1, 1, 4, 9
ALPHA = 0.05
REP = 20_000
PAIRS = list(itertools.combinations(range(K), 2))


def run(nmin, rep=REP):
    """가장 작은 집단이 nmin 일 때 두 방법의 FWER 과 쌍별 오류율."""
    # 큰 분산에 작은 n 을 붙인다 — 불균등 설계에서 가장 불리한 짝이다
    n = np.array([3 * nmin, 2 * nmin, nmin, nmin])
    m = np.empty((rep, K))
    v = np.empty((rep, K))
    for g in range(K):
        x = RNG.normal(0.0, SD[g], (rep, n[g]))
        m[:, g] = x.mean(1)
        v[:, g] = x.var(1, ddof=1)

    rej_gh = np.zeros(rep, bool)
    rej_bw = np.zeros(rep, bool)
    cnt_gh = np.zeros(len(PAIRS))
    cnt_bw = np.zeros(len(PAIRS))
    for idx, (i, j) in enumerate(PAIRS):
        se2 = v[:, i] / n[i] + v[:, j] / n[j]
        # 웰치–새터스웨이트 자유도
        df = se2 ** 2 / ((v[:, i] / n[i]) ** 2 / (n[i] - 1)
                         + (v[:, j] / n[j]) ** 2 / (n[j] - 1))
        d = np.abs(m[:, i] - m[:, j])

        # 게임스-하월: 스튜던트화 범위를 sqrt(2) 로 나눈 것을 문턱으로
        q = q_crit(K, df, ALPHA)
        hit_gh = d > (q / np.sqrt(2)) * np.sqrt(se2)

        # 본페로니 보정 쌍별 웰치
        tcrit = stats.t.ppf(1 - ALPHA / (2 * len(PAIRS)), df)
        hit_bw = d > tcrit * np.sqrt(se2)

        rej_gh |= hit_gh
        rej_bw |= hit_bw
        cnt_gh[idx] = hit_gh.mean()
        cnt_bw[idx] = hit_bw.mean()
    return (rej_gh.mean(), rej_bw.mean(), cnt_gh.mean(), cnt_bw.mean())


NMIN = [3, 4, 5, 6, 8, 10, 15, 20, 30]
print(f"집단 {K} 개, 표준편차 {list(SD)}, 표본크기 (3n, 2n, n, n) — "
      f"큰 분산에 작은 n. 평균은 모두 같다")
print(f"이름값 {ALPHA}, 반복 {REP:,}")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n  가장 작은 n   GH 집단오류   GH 쌍별   본페로니웰치 집단오류   쌍별")
GH_F, BW_F, GH_P, BW_P = [], [], [], []
for nm in NMIN:
    a, b, c, d = run(nm)
    GH_F.append(a); BW_F.append(b); GH_P.append(c); BW_P.append(d)
    print(f"   {nm:>10}   {a:>11.4f}   {c:>8.4f}   {b:>20.4f}   {d:>6.4f}")

i6, i8 = NMIN.index(6), NMIN.index(8)
print(f"\n   n = 3 에서 게임스-하월의 집단오류가 {GH_F[0]:.4f} 로 "
      f"이름값의 {GH_F[0]/ALPHA:.1f} 배다.")
print(f"   n = 6 에서 {GH_F[i6]:.4f}, n = 8 에서 {GH_F[i8]:.4f} 로 가라앉는다 —")
print(f"   초록이 「여섯부터」라 적은 자리를 둘러싼다.")
print(f"   본페로니 보정 웰치도 n = 3 에서는 {BW_F[0]:.4f} 로 이름값을 넘지만,")
print(f"   n = 4 부터 {BW_F[1]:.4f} 로 내려와 줄곧 게임스-하월보다 낮다.")
print("   **작은 불균등 표본에서는 어느 쪽도 이름값을 지키지 못한다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_G, C_B, C_R = "#d62728", "#1f77b4", "#2ca02c"

ax1.plot(NMIN, GH_F, "o-", color=C_G, lw=2.8, ms=10, label="게임스-하월")
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax1.axvline(6, color=C_R, lw=2.0, ls=":")
ax1.text(6.3, max(GH_F) * 0.86, "초록이 말한\n「여섯부터」", fontsize=10.5,
         color=C_R)
ax1.text(16, ALPHA * 1.12, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax1.set_xlabel("가장 작은 집단의 크기")
ax1.set_ylabel("집단별 $1$ 종오류 (FWER)")
ax1.set_title("① 작은 $n$ 에서 느슨해진다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.plot(NMIN, GH_P, "o-", color=C_G, lw=2.8, ms=10, label="게임스-하월")
ax2.plot(NMIN, BW_P, "s-", color=C_B, lw=2.6, ms=9, label="본페로니 보정 웰치")
ax2.axhline(ALPHA / len(PAIRS), color="#333333", lw=1.8, ls="--")
ax2.text(16, ALPHA / len(PAIRS) * 1.3, "$0.05/6$", fontsize=10.5,
         color="#333333")
ax2.set_xlabel("가장 작은 집단의 크기")
ax2.set_ylabel("쌍 하나당 $1$ 종오류")
ax2.set_title("② 쌍 하나로 보아도 같다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.plot(NMIN, GH_F, "o-", color=C_G, lw=2.8, ms=10, label="게임스-하월")
ax3.plot(NMIN, BW_F, "s-", color=C_B, lw=2.6, ms=9, label="본페로니 보정 웰치")
ax3.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax3.text(16, ALPHA * 1.1, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax3.set_xlabel("가장 작은 집단의 크기")
ax3.set_ylabel("집단별 $1$ 종오류")
ax3.set_title("③ 절 6 의 대안과 견주면", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/post_hoc/img/gh_small_n.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
