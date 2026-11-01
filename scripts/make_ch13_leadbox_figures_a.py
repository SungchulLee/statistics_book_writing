"""13장 들어가는 이야기 그림 — 첫 묶음 다섯.

각 그림은 그 들머리가 세우는 것만 그린다. 쪽에 이미 있는 그림과 겹치지
않도록 각도를 골랐고, 고른 까닭을 그림마다 적어 둔다.

  1. index                  웰던의 열세 칸 — 칸을 하나씩 보면 답이 갈린다
  2. distribution           자유도를 하나 잘못 세면 치르는 값
  3. degrees_of_freedom     추정한 모수마다 사영의 계수가 하나 깎인다
  4. gof                    눈에 치우쳐 보이는 표가 어디서부터 기각되는가
  5. mcnemar_test           대각선을 키우면 두 검정이 갈라진다

만드는 파일:
  docs/ch13/img/weldon_thirteen_cells.png
  docs/ch13/distribution/img/one_off_in_df.png
  docs/ch13/distribution/img/estimating_costs_a_rank.png
  docs/ch13/test/img/how_lopsided_before_reject.png
  docs/ch13/test/img/diagonal_splits_the_two.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats
from scipy.special import gammaln

IDX = "docs/ch13/img/"
DIST = "docs/ch13/distribution/img/"
TEST = "docs/ch13/test/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. 장 개요 — 웰던의 열세 칸
#
# 들머리가 세우는 것: 「어느 칸을 들여다보느냐에 따라 답이 달라진다 —
# 열세 칸의 어긋남을 하나의 수로 모아 주는 자가 필요했다.」 그래서 칸별
# 어긋남이 **방향도 크기도 들쭉날쭉하다**는 것을 보이고, 그것이 수 하나로
# 모이는 자리까지 간다. 장 개요는 독자가 카이제곱도 자유도도 아직 배우지
# 않은 자리이므로 세는 일 말고는 아무것도 요구하지 않는다.
#
# 쓰는 수는 13.2절 gof.md 의 상자 출처 줄에 확인해 적어 둔 것뿐이다 —
# 관찰도수 열셋과 n = 26,306. 이론도수는 공정할 때의 이항분포로 내가
# 계산한다(상자가 적어 둔 5018 · 2927 과 맞는지 확인한다).
# ════════════════════════════════════════════════════════════════════
OBS = np.array([185, 1149, 3265, 5475, 6114, 5194, 3067, 1331, 403,
                105, 14, 4, 0], float)
NTHROW = 26306
EXP = NTHROW * stats.binom.pmf(np.arange(13), 12, 1 / 3)

assert OBS.sum() == NTHROW
print("[웰던] 상자가 적어 둔 이론도수와 맞춰 본다")
print(f"  다섯 개: 관찰 {OBS[5]:.0f} · 이론 {EXP[5]:.1f}  (상자는 5018)")
print(f"  여섯 개: 관찰 {OBS[6]:.0f} · 이론 {EXP[6]:.1f}  (상자는 2927)")
mean_obs = (OBS * np.arange(13)).sum() / NTHROW
print(f"  한 번당 평균 {mean_obs:.4f}  (상자는 4.0524)")

contrib = (OBS - EXP) ** 2 / EXP
chi2_weldon = contrib.sum()
print(f"  열세 칸을 모은 수 = {chi2_weldon:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8))
x = np.arange(13)
w = 0.42
ax1.bar(x - w / 2, EXP, width=w, color="0.68", label="공정하다면 이만큼")
ax1.bar(x + w / 2, OBS, width=w, color="tab:blue", label="실제로 이만큼")
ax1.set_xticks(x)
ax1.set_xlabel("한 번 던져 5 나 6 을 보인 주사위의 개수", fontsize=11)
ax1.set_ylabel("그런 일이 일어난 횟수", fontsize=11)
ax1.set_title("웰던이 적은 표 — 26,306 번을 열세 칸으로", fontsize=13)
ax1.legend(fontsize=10.5)
ax1.grid(axis="y", alpha=0.25)

diff = OBS - EXP
cols = ["tab:red" if d > 0 else "tab:blue" for d in diff]
ax2.bar(x, diff, color=cols, width=0.68)
ax2.axhline(0, color="0.2", lw=1.3)
ax2.set_ylim(-280, 310)
ax2.annotate(f"{OBS[5]:.0f} 인데 {EXP[5]:.0f} 이어야 했다",
             xy=(5, diff[5]), xytext=(6.6, 268), fontsize=10.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax2.annotate(f"{OBS[6]:.0f} 인데 {EXP[6]:.0f} 이어야 했다",
             xy=(6, diff[6]), xytext=(7.4, 186), fontsize=10.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax2.annotate("이 칸만 보면 거꾸로 적다", xy=(4, diff[4]),
             xytext=(5.0, -245), fontsize=10.5, color="tab:blue",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
ax2.set_xticks(x)
ax2.set_xlabel("한 번 던져 5 나 6 을 보인 주사위의 개수", fontsize=11)
ax2.set_ylabel("실제 횟수 빼기 공정할 때의 횟수", fontsize=11)
ax2.set_title("칸을 하나씩 보면 답이 갈린다 — 넘친 칸도 모자란 칸도 있다",
              fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)
ax2.text(0.025, 0.055, "고른 칸이 답을 정해 버린다", transform=ax2.transAxes,
         ha="left", va="bottom", fontsize=11, color="0.25",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fff4e0", ec="0.7"))

fig.suptitle("주사위를 26,306 번 던진 사람", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, IDX + "weldon_thirteen_cells.png")


# ════════════════════════════════════════════════════════════════════
# 2. distribution — 자유도를 하나 잘못 세면 치르는 값
#
# 쪽의 df_is_k_minus_one 은 「두 표준화 가운데 어느 쪽이 카이제곱이 되는가」
# 를 그린다. 여기서는 그 쪽이 건드리지 않는 것, 곧 **값**만 본다 —
# 들머리의 머리글이 「5% 가 2.4% 가 된다」이기 때문이다. 틀어지는 방향이
# 한쪽이 아니라는 것(너무 크게 잡으면 놓치고 작게 잡으면 기각한다)까지
# 함께 그린다.
#
# 설정은 쪽의 것과 같다 — k = 4, n = 200, p = (0.4, 0.3, 0.2, 0.1).
# 쪽은 20 만 회 모의실험으로 0.0492 와 0.0237 을 얻었다. 여기서는 표를
# **전부 열거해 정확히** 센다 — n = 200, k = 4 이면 가능한 표가
# C(203, 3) = 1,373,701 가지뿐이라 셈이 끝난다. 그래서 이 그림의 수에는
# 모의실험 오차가 없고, 쪽의 두 수가 그 정확값의 읽음이라는 것도 보인다.
# ════════════════════════════════════════════════════════════════════
P4 = np.array([0.4, 0.3, 0.2, 0.1])
N4 = 200
E4 = N4 * P4

i, j, k_ = np.meshgrid(np.arange(N4 + 1), np.arange(N4 + 1),
                       np.arange(N4 + 1), indexing="ij")
keep = (i + j + k_) <= N4
c1, c2, c3 = i[keep], j[keep], k_[keep]
c4 = N4 - c1 - c2 - c3
TAB = np.stack([c1, c2, c3, c4], 1).astype(float)
logp = (gammaln(N4 + 1)
        - gammaln(TAB + 1).sum(1)
        + (TAB * np.log(P4)).sum(1))
PR = np.exp(logp)
pearson = ((TAB - E4) ** 2 / E4).sum(1)

print("\n[distribution] 가능한 표를 전부 열거해 정확히 센다")
print(f"  표 {len(TAB):,} 가지 · 확률의 합 {PR.sum():.10f}")
m = (pearson * PR).sum()
v = (pearson ** 2 * PR).sum() - m ** 2
print(f"  평균 {m:.4f} · 분산 {v:.4f}   (쪽의 모의실험은 3.003 · 6.001)")
ASSUMED = [2, 3, 4, 5]
crits = [stats.chi2.ppf(0.95, d) for d in ASSUMED]
rates = [PR[pearson >= c].sum() for c in crits]
for d, c, r in zip(ASSUMED, crits, rates):
    tag = "  ← 맞는 자유도" if d == 3 else ""
    print(f"  자유도를 {d} 로 잡으면 임계값 {c:.3f} · 기각률 {r:.4f}{tag}")
print(f"  쪽이 적은 0.0492 · 0.0237 과 맞는다")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                               gridspec_kw={"width_ratios": [1.3, 1]})

# 꼬리만 보이게 자른다 — 보여야 하는 것은 봉우리가 아니라 두 문턱이다
grid = np.linspace(4, 20, 600)
ax1.hist(pearson, bins=np.linspace(4, 20, 110), weights=PR, density=True,
         color="0.84", edgecolor="none",
         label="통계량의 정확한 분포 (표를 전부 열거했다)")
ax1.plot(grid, stats.chi2.pdf(grid, 3), color="0.25", lw=2.2,
         label="맞는 분포")
c3, c4 = stats.chi2.ppf(0.95, 3), stats.chi2.ppf(0.95, 4)
tail3 = grid[grid >= c3]
tail4 = grid[grid >= c4]
ax1.fill_between(tail3, 0, stats.chi2.pdf(tail3, 3), color="tab:blue",
                 alpha=0.40)
ax1.fill_between(tail4, 0, stats.chi2.pdf(tail4, 3), color="tab:red",
                 alpha=0.80)
ax1.axvline(c3, color="tab:blue", lw=2.4)
ax1.axvline(c4, color="tab:red", lw=2.4, ls="--")
ax1.annotate(f"자유도 3 의 문턱 {c3:.3f}\n→ 기각률 {rates[1]:.4f}",
             xy=(c3 + 0.35, 0.018), xytext=(11.6, 0.0485), fontsize=11.5,
             color="tab:blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax1.annotate(f"자유도 4 로 잘못 잡으면 {c4:.3f}\n→ 기각률 {rates[2]:.4f}",
             xy=(c4 + 0.45, 0.0095), xytext=(11.6, 0.0255), fontsize=11.5,
             color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.set_xlim(4, 20)
ax1.set_ylim(0, 0.072)
ax1.set_xlabel("피어슨 통계량 (꼬리만 잘라 그렸다)", fontsize=11)
ax1.set_ylabel("밀도", fontsize=11)
ax1.set_title("같은 통계량, 다른 문턱 — 꼬리가 절반 아래로 줄어든다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="upper right")

bx = np.arange(len(ASSUMED))
bcol = ["tab:orange" if d < 3 else ("0.35" if d == 3 else "tab:red")
        for d in ASSUMED]
ax2.bar(bx, rates, color=bcol, width=0.6)
for i, r in enumerate(rates):
    ax2.text(i, r + 0.004, f"{r:.4f}", ha="center", va="bottom",
             fontsize=11.5, fontweight="bold")
ax2.axhline(0.05, color="tab:blue", lw=2.0, ls="--")
ax2.text(3.42, 0.0525, "명목 5%", ha="right", va="bottom", fontsize=11,
         color="tab:blue", fontweight="bold")
ax2.set_xticks(bx, [f"{d}" for d in ASSUMED], fontsize=12)
ax2.set_xlabel("내가 쓴 자유도 (맞는 것은 3)", fontsize=11)
ax2.set_ylabel("실제 기각률", fontsize=11)
ax2.set_ylim(0, 0.148)
ax2.set_title("틀어지는 방향이 한쪽이 아니다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)
ax2.text(0.97, 0.80, "작게 잡으면 기각하는 쪽\n크게 잡으면 놓치는 쪽",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11,
         color="0.25", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="0.75"))

fig.suptitle("자유도를 하나 잘못 세면 5% 가 2.4% 가 된다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, DIST + "one_off_in_df.png")


# ════════════════════════════════════════════════════════════════════
# 3. degrees_of_freedom — 추정한 모수마다 사영의 계수가 하나 깎인다
#
# 쪽의 df_counting 은 「자유로운 칸을 센다」를, df_dispute 는 「틀린 자유도를
# 쓰면 어떻게 되는가」를 그린다. 연습문제 1 은 모수를 추정하지 않은 경우의
# 고윳값 0, 1, 1, 1 을 이미 수로 확인한다. 그래서 여기서는 들머리가
# **묻고 끝나는 것**만 그린다 — 「자료에서 추정한 모수 하나가 왜 그 계수를
# 하나 깎는가.」 깎이는 자리를 재어서 보인다.
#
# 모형: 칸이 k 개인 도수를 이항 Bin(k-1, θ) 의 도수로 본다. θ 를 알고
# 있으면 사영의 계수가 k-1 이고, θ 를 자료에서 ML 로 추정하면 k-2 다.
# 둘 다 공분산행렬을 추정해 고윳값을 직접 센다.
# ════════════════════════════════════════════════════════════════════
def rank_by_measurement(k, n, theta, fit, reps, seed):
    """표준화 편차벡터의 공분산행렬을 추정해 고윳값과 대각합을 돌려준다."""
    r = np.random.default_rng(seed)
    cells = np.arange(k)
    p_true = stats.binom.pmf(cells, k - 1, theta)
    p_true = p_true / p_true.sum()
    obs = r.multinomial(n, p_true, size=reps).astype(float)
    if fit:
        # θ 의 ML 추정 — 이항이므로 성공 비율이 그대로다
        th = (obs * cells).sum(1) / (n * (k - 1))
        pm = stats.binom.pmf(cells[None, :], k - 1, th[:, None])
        pm = pm / pm.sum(1, keepdims=True)
        E = n * pm
    else:
        E = n * p_true[None, :]
    U = (obs - E) / np.sqrt(E)
    C = (U.T @ U) / reps
    ev = np.linalg.eigvalsh(C)
    return np.sort(ev), np.trace(C)


KS = [4, 5, 6, 7, 8]
NN, R3 = 4000, 60_000
print("\n[degrees_of_freedom] 사영의 계수를 재어서 센다")
tr_known, tr_fit = [], []
for k in KS:
    _, t0 = rank_by_measurement(k, NN, 0.42, False, R3, 900 + k)
    _, t1 = rank_by_measurement(k, NN, 0.42, True, R3, 900 + k)
    tr_known.append(t0)
    tr_fit.append(t1)
    print(f"  k = {k}:  θ 를 알면 대각합 {t0:.3f} (k-1 = {k-1}) · "
          f"θ 를 추정하면 {t1:.3f} (k-2 = {k-2})")

ev0, _ = rank_by_measurement(6, NN, 0.42, False, R3, 777)
ev1, _ = rank_by_measurement(6, NN, 0.42, True, R3, 777)
print(f"  k = 6 고윳값 — θ 를 알면 {np.round(ev0, 3)}")
print(f"  k = 6 고윳값 — θ 를 추정하면 {np.round(ev1, 3)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8))
ex = np.arange(6)
w = 0.4
ax1.bar(ex - w / 2, ev0, width=w, color="0.45",
        label=f"모수를 알고 있다 — 대각합 {ev0.sum():.2f}")
ax1.bar(ex + w / 2, ev1, width=w, color="tab:red",
        label=f"모수 하나를 자료에서 추정했다 — 대각합 {ev1.sum():.2f}")
ax1.axhline(1, color="0.3", lw=1.1, ls=":")
# 0 인 고윳값은 막대가 보이지 않으므로 「없음」과 갈리도록 직접 적어 준다
for i, (v0, v1) in enumerate(zip(ev0, ev1)):
    if v0 < 0.05:
        ax1.text(i - w / 2, 0.035, "0", ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="0.45")
    if v1 < 0.05:
        ax1.text(i + w / 2, 0.035, "0", ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="tab:red")
ax1.annotate("모수를 추정하면\n고윳값 하나가\n여기로 더 내려앉는다",
             xy=(1 + w / 2, 0.115), xytext=(2.35, 0.50), fontsize=11.5,
             color="tab:red", fontweight="bold", ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="tab:red",
                       alpha=0.95),
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax1.set_xticks(ex, [f"{i+1}" for i in ex], fontsize=11)
ax1.set_xlabel("작은 것부터 늘어놓은 고윳값 (범주 k = 6)", fontsize=11)
ax1.set_ylabel("고윳값", fontsize=11)
ax1.set_ylim(0, 1.36)
ax1.set_title("재어 보면 0 과 1 만 나온다 — 그 1 의 개수가 자유도다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(axis="y", alpha=0.25)

kk = np.array(KS, float)
ax2.plot(kk, kk - 1, "-", color="0.72", lw=5.5, alpha=0.6,
         label="세어서 빼는 수 k-1")
ax2.plot(kk, kk - 2, "-", color="#f0a0a0", lw=5.5, alpha=0.7,
         label="세어서 빼는 수 k-2")
ax2.plot(kk, tr_known, "o-", color="0.25", lw=2.2, ms=10,
         label="재어 본 대각합 (모수를 안다)")
ax2.plot(kk, tr_fit, "s--", color="tab:red", lw=2.2, ms=9,
         label="재어 본 대각합 (모수를 추정했다)")
ax2.set_xticks(KS)
ax2.set_xlabel("범주의 개수 k", fontsize=11)
ax2.set_ylabel("사영의 계수 (= 대각합)", fontsize=11)
ax2.set_title("세어서 빼는 수가 재어서 그대로 나온다", fontsize=12.5)
ax2.legend(fontsize=9.8, loc="upper left")
ax2.grid(alpha=0.25)

fig.suptitle("세어서 빼는 수를 재어서 보인다 — 추정한 모수가 계수를 하나 "
             "깎는다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, DIST + "estimating_costs_a_rank.png")


# ════════════════════════════════════════════════════════════════════
# 4. gof — 눈에 치우쳐 보이는 표가 어디서부터 기각되는가
#
# 쪽의 gof_exact_null 은 「24 경기의 관측·기대와 칸별 기여, 그리고 통계량이
# 가질 수 있는 44 개 값」을 그린다. 들머리의 머리글은 그것이 아니라
# 「눈에 치우쳐 보이는 표가 기각되지 않는다」다. 그래서 눈과 자의 간극을
# 직접 그린다 — 패를 하나씩 늘려 가며 어디서부터 기각되는지 본다.
#
# 승을 4 로 묶고 패를 늘려 무에서 덜어 낸다. n = 24 는 그대로다.
# ════════════════════════════════════════════════════════════════════
WIN = 4
LOSSES = np.arange(10, 17)
rows, ps, chis = [], [], []
for L in LOSSES:
    D = 24 - WIN - L
    o = np.array([WIN, L, D], float)
    c = ((o - 8.0) ** 2 / 8.0).sum()
    p = stats.chi2.sf(c, 2)
    rows.append((WIN, L, D))
    chis.append(c)
    ps.append(p)
chis, ps = np.array(chis), np.array(ps)
first_rej = LOSSES[ps < 0.05][0]

print("\n[gof] 승 4 를 묶고 패를 늘려 본다 (n = 24, 기대 8)")
for (w_, l_, d_), c, p in zip(rows, chis, ps):
    mark = "  ← 기각" if p < 0.05 else ""
    print(f"  승 {w_} 패 {l_:2d} 무 {d_:2d}   chi2 = {c:5.3f}  p = {p:.4f}{mark}")
print(f"  쪽의 표는 패 13 (p = {ps[LOSSES == 13][0]:.4f}) 이고 "
      f"기각은 패 {first_rej} 부터다")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.2]})

labels = ["승", "패", "무"]
tx = np.arange(3)
w = 0.4
page = np.array([4, 13, 7], float)
edge = np.array(rows[list(LOSSES).index(first_rej)], float)
ax1.bar(tx - w / 2, page, width=w, color="tab:blue",
        label=f"쪽의 표 (4, 13, 7) — p = {ps[LOSSES == 13][0]:.4f}")
ax1.bar(tx + w / 2, edge, width=w, color="tab:red",
        label=f"기각이 시작되는 표 ({edge[0]:.0f}, {edge[1]:.0f}, "
              f"{edge[2]:.0f}) — p = {ps[LOSSES == first_rej][0]:.4f}")
ax1.axhline(8, color="0.3", lw=2.0, ls="--")
ax1.text(2.42, 8.3, "고르다면 칸마다 8", ha="right", va="bottom",
         fontsize=10.5, color="0.3", fontweight="bold")
ax1.set_xticks(tx, labels, fontsize=13)
ax1.set_ylabel("경기 수 (n = 24)", fontsize=11)
ax1.set_ylim(0, 17)
ax1.set_title("한 칸을 하나 옮기면 판정이 뒤집힌다", fontsize=12.5)
ax1.legend(fontsize=9.6, loc="upper left")
ax1.grid(axis="y", alpha=0.25)

ax2.plot(LOSSES, ps, "o-", color="0.3", lw=2.2, ms=9)
ax2.axhline(0.05, color="tab:red", lw=2.0, ls="--")
ax2.text(10.05, 0.058, "0.05", ha="left", va="bottom", fontsize=11,
         color="tab:red", fontweight="bold")
ax2.axvspan(first_rej - 0.5, 16.5, color="tab:red", alpha=0.08)
i13 = list(LOSSES).index(13)
ax2.plot([13], [ps[i13]], "o", ms=15, mfc="none", mec="tab:blue", mew=3)
ax2.annotate(f"패가 13 — 꽤 치우쳐 보이는데\np = {ps[i13]:.4f} 로 "
             f"기각되지 않는다", xy=(13, ps[i13]), xytext=(13.1, 0.155),
             fontsize=11, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
ir = list(LOSSES).index(first_rej)
ax2.annotate(f"패가 {first_rej} 부터 기각", xy=(first_rej, ps[ir]),
             xytext=(first_rej - 0.4, 0.012), fontsize=11, color="tab:red",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax2.set_xticks(LOSSES)
ax2.set_xlabel("패의 도수 (승은 4 로 묶고 무에서 덜어 냈다)", fontsize=11)
ax2.set_ylabel("p 값", fontsize=11)
ax2.set_ylim(0, 0.26)
ax2.set_title("자가 끊는 자리는 눈이 보는 자리보다 멀다", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("눈에 치우쳐 보이는 표가 기각되지 않는다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TEST + "how_lopsided_before_reject.png")


# ════════════════════════════════════════════════════════════════════
# 5. mcnemar_test — 같은 표에서 두 검정이 아무 방향으로나 갈라진다
#
# 쪽의 mcnemar_discordant 는 「314 쌍 가운데 상태가 바뀐 180 쌍만 들어가고,
# 그 180 쌍의 방향은 동전과 같다」를 그린다. 들머리에서 그 그림이 건드리지
# 않는 것은 머리글 그 자체다 — 「수는 나오지만 **다른 물음의 답**이다.」
#
# 처음에는 「대각선을 키우면 보통의 검정이 작아진다」를 그리려 했는데
# 재어 보니 거꾸로였다. 대각선이 커지면 전·후가 강하게 **연관**되므로
# 보통의 검정은 오히려 커진다. 쪽이 적은 20.7 -> 8.31 은 일반적 방향이
# 아니라 이 표에서 그렇게 나온 값이다.
#
# 그래서 재어 본 것을 그린다. 대각선을 101·33 으로 붙들고 불일치 쌍의
# 나눔만 훑으면 두 검정이 **아무 방향으로나** 갈라진다 — 변화가 전혀 없는
# 표(호전 90 · 악화 90)에서 맥니마는 0 인데 보통의 검정은 12.09 로
# 기각하고, 호전이 159 로 쏠린 자리에서는 맥니마가 105 인데 보통의 검정이
# 0.00, 곧 「완전히 독립」이라고 말한다.
# ════════════════════════════════════════════════════════════════════
A0, D0 = 101, 33        # 쪽의 표에서 바뀌지 않은 사람 (101 + 33 = 134)
NDISC = 180             # 바뀐 사람 (호전 121 + 악화 59)
BS = np.arange(90, 181)
mc, pear = [], []
for b in BS:
    c = NDISC - b
    mc.append((abs(b - c) - 1) ** 2 / NDISC if b != c else 0.0)
    tbl = np.array([[A0, b], [c, D0]], float)
    pear.append(stats.chi2_contingency(tbl)[0])      # 예이츠 기본값
mc, pear = np.array(mc), np.array(pear)

i121 = int(np.where(BS == 121)[0][0])
i90 = 0
i_indep = int(np.argmin(pear))
print("\n[mcnemar] 대각선을 101·33 으로 붙들고 불일치 쌍의 나눔만 훑는다")
print(f"  쪽의 표 (호전 121 · 악화 59): 맥니마 {mc[i121]:.2f} · "
      f"보통의 검정 {pear[i121]:.2f}   (쪽은 20.7 과 8.31)")
print(f"  변화가 전혀 없는 표 (90 · 90): 맥니마 {mc[i90]:.2f} · "
      f"보통의 검정 {pear[i90]:.2f}")
print(f"  호전 {BS[i_indep]} · 악화 {NDISC - BS[i_indep]}: "
      f"맥니마 {mc[i_indep]:.2f} · 보통의 검정 {pear[i_indep]:.2f}")
print(f"  호전 180 · 악화 0: 맥니마 {mc[-1]:.2f} · "
      f"보통의 검정 {pear[-1]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9),
                               gridspec_kw={"width_ratios": [1.3, 1]})

CRIT = stats.chi2.ppf(0.95, 1)
ax1.plot(BS, mc, "-", color="tab:red", lw=3.2,
         label="바뀐 쌍만 쓰는 검정 (맥니마)")
ax1.plot(BS, pear, "-", color="tab:blue", lw=3.2,
         label="두 관측을 다른 사람으로 보는 검정")
ax1.axhline(CRIT, color="0.35", lw=1.8, ls=":")
ax1.text(104, -3.4, "가로 점선은 자유도 1 의 5% 문턱 3.84", ha="left",
         va="center", fontsize=10, color="0.3", fontweight="bold")
ax1.plot([BS[i121]], [mc[i121]], "o", ms=11, color="tab:red", mec="white",
         mew=2)
ax1.plot([BS[i121]], [pear[i121]], "o", ms=11, color="tab:blue",
         mec="white", mew=2)
ax1.annotate(f"쪽의 표 — 호전 121 · 악화 59\n{mc[i121]:.1f} 대 "
             f"{pear[i121]:.2f}", xy=(BS[i121], mc[i121]),
             xytext=(104, 72), fontsize=11, color="0.2", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.2", lw=1.7))
ax1.annotate(f"변화가 전혀 없는 표인데\n보통의 검정은 {pear[i90]:.2f} 로 "
             f"기각한다", xy=(90.6, pear[i90]), xytext=(96, 128),
             fontsize=11, color="tab:blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.7))
ax1.annotate(f"맥니마가 {mc[i_indep]:.0f} 인데\n보통의 검정은 "
             f"{pear[i_indep]:.2f} —\n「완전히 독립」이라고 말한다",
             xy=(BS[i_indep], pear[i_indep] + 1.5), xytext=(179, 40),
             fontsize=11, color="tab:blue", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.7))
ax1.set_xlim(89, 181)
ax1.set_ylim(-6, 190)
ax1.set_xlabel("호전한 사람의 수 (악화는 180 빼기 그 수 · 바뀌지 않은 "
               "134 명은 그대로)", fontsize=10.5)
ax1.set_ylabel("검정통계량", fontsize=11)
ax1.set_title("같은 표, 갈라지는 두 수 — 방향이 한쪽이 아니다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(alpha=0.25)

picks = [90, 121, BS[i_indep], 180]
pi = [int(np.where(BS == q)[0][0]) for q in picks]
bx = np.arange(len(picks))
w = 0.38
ax2.bar(bx - w / 2, mc[pi], width=w, color="tab:red", label="맥니마")
ax2.bar(bx + w / 2, pear[pi], width=w, color="tab:blue",
        label="보통의 분할표 검정")
for s, (m_, p_) in enumerate(zip(mc[pi], pear[pi])):
    ax2.text(s - w / 2, m_ + 3, f"{m_:.1f}", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color="tab:red")
    ax2.text(s + w / 2, p_ + 3, f"{p_:.2f}", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color="tab:blue")
ax2.axhline(CRIT, color="0.35", lw=1.8, ls=":")
ax2.set_xticks(bx, [f"90 대 90\n(변화 없음)", "121 대 59\n(쪽의 자료)",
                    f"{picks[2]} 대 {NDISC - picks[2]}", "180 대 0"],
               fontsize=10.5)
ax2.set_xlabel("호전 대 악화", fontsize=11)
ax2.set_ylabel("검정통계량", fontsize=11)
ax2.set_ylim(0, 205)
ax2.set_title("한쪽이 0 이라 말할 때 다른 쪽은 기각한다", fontsize=12.5)
ax2.legend(fontsize=10, loc="upper left")
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("수는 나오지만 다른 물음의 답이다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TEST + "paired_vs_unpaired_disagree.png")
