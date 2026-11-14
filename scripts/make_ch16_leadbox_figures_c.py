"""16장 들어가는 이야기 그림 — 셋째 그룹 일곱 (마지막).

  15. ks_two_sample    다섯씩이면 바닥이 0.0079 라 겹치지 않아야 기각된다
  16. kruskal_wallis   다섯씩 세 집단에서 근사가 덜 기각하는 쪽으로
  17. dunn             쌍의 수가 느는 속도와, 상관이 덜어 주는 몫
  18. friedman         통계량이 가질 수 있는 값이 마흔다섯 개뿐이다
  19. median_test      전체 중앙값과 똑같은 값을 어디에 넣는가
  20. spearman         축약 공식이 몰래 쓰는 가정
  21. kendall          타우와 스피어만은 같은 자가 아니다

만드는 파일:
  docs/ch16/two_sample_nonparametric/img/floor_needs_no_overlap.png
  docs/ch16/multi_group_nonparametric/img/approximation_under_rejects.png
  docs/ch16/multi_group_nonparametric/img/pairs_grow_but_correlate.png
  docs/ch16/multi_group_nonparametric/img/only_forty_five_values.png
  docs/ch16/multi_group_nonparametric/img/where_the_ties_go.png
  docs/ch16/correlation/img/the_shortcut_assumes_fortytwo.png
  docs/ch16/correlation/img/tau_and_rho_are_not_one_ruler.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from itertools import combinations, permutations, product
from scipy import stats

TWO = "docs/ch16/two_sample_nonparametric/img/"
MG = "docs/ch16/multi_group_nonparametric/img/"
COR = "docs/ch16/correlation/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 15. ks_two_sample — 바닥이 2/252 라 겹치지 않아야 기각된다
#
# 쪽의 ks_ecdf_gap 이 두 누적분포의 간격을, ftc__cdf_gap 이 거리에 따른
# 넓이와 중첩을 그린다. 들머리의 **셋째 문단**은 그려져 있지 않다 —
# 「이 설계에서 나올 수 있는 가장 작은 p 값이 2/252 = 0.0079 다.
# 두 표본이 아예 겹치지 않는 자료가 아니면 기각이 일어나지 않는다.」
# ════════════════════════════════════════════════════════════════════
def ks_exact_floor(n1, n2):
    """주변 배치를 전부 세어 이 설계의 가장 작은 양측 p 를 구한다."""
    N = n1 + n2
    ds = []
    for idx in combinations(range(N), n1):
        lab = np.zeros(N, int)
        lab[list(idx)] = 1
        a = np.cumsum(lab) / n1
        b = np.cumsum(1 - lab) / n2
        ds.append(np.abs(a - b).max())
    ds = np.array(ds)
    dmax = ds.max()
    return (ds >= dmax - 1e-12).mean(), len(ds), np.unique(np.round(ds, 9))


SIZES = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
floors, counts = [], []
for n1, n2 in SIZES:
    fl, cnt, vals = ks_exact_floor(n1, n2)
    floors.append(fl)
    counts.append(cnt)
    if n1 == 5:
        vals5 = vals
print("[ks_two_sample] 두 집단이 같은 크기일 때의 바닥")
for (n1, n2), fl, cnt in zip(SIZES, floors, counts):
    print(f"  {n1} 대 {n2}:  배치 {cnt:6d} 가지 · 가장 작은 양측 p "
          f"{fl:.5f}")
print(f"  5 대 5 에서 D 가 가질 수 있는 값 {vals5}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ns = [n1 for n1, _ in SIZES]
ax1.plot(ns, floors, "o-", color="tab:blue", lw=3.2, ms=14)
ax1.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax1.axhline(0.01, color="0.5", lw=1.8, ls=":")
ax1.set_yscale("log")
ax1.set_yticks([0.001, 0.0079, 0.05, 0.1, 0.5],
               ["0.001", "0.0079", "0.05", "0.1", "0.5"])
ax1.set_xticks(ns)
ax1.set_xlabel("집단당 표본크기 (두 집단이 같은 크기)", fontsize=11.5)
ax1.set_ylabel("어떤 자료가 와도 그보다 작아질 수 없는 양측 p",
               fontsize=11.5)
ax1.grid(alpha=0.25, which="both")
i5 = ns.index(5)
ax1.annotate(f"다섯씩이면 {floors[i5]:.4f} —\n"
             f"두 표본이 아예 겹치지 않는 자료에서만 나온다",
             xy=(5, floors[i5]), xytext=(5.2, 0.22), fontsize=11.5,
             color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax1.set_title("바닥은 자료가 아니라 설계가 정한다", fontsize=12.5)

A15 = np.array([10.1, 10.3, 10.2, 10.5, 10.4])
B15 = np.array([10.0, 10.2, 10.6, 10.8, 10.4])
gridx = np.linspace(9.9, 10.95, 400)
ea = np.searchsorted(np.sort(A15), gridx, side="right") / len(A15)
eb = np.searchsorted(np.sort(B15), gridx, side="right") / len(B15)
ax2.step(gridx, ea, where="post", color="tab:blue", lw=3.0, label="공정 A")
ax2.step(gridx, eb, where="post", color="tab:red", lw=3.0, label="공정 B")
j = int(np.argmax(np.abs(ea - eb)))
ax2.annotate("", xy=(gridx[j], ea[j]), xytext=(gridx[j], eb[j]),
             arrowprops=dict(arrowstyle="<->", color="0.15", lw=3.0))
ax2.text(gridx[j] + 0.03, (ea[j] + eb[j]) / 2,
         f"가장 큰 간격 D = {abs(ea[j] - eb[j]):.1f}", ha="left",
         va="center", fontsize=12, color="0.15", fontweight="bold")
ax2.set_xlabel("지름 (mm)", fontsize=11.5)
ax2.set_ylabel("그 값 이하인 비율", fontsize=11.5)
ax2.set_ylim(-0.05, 1.08)
ax2.legend(fontsize=11.5, loc="upper left")
ax2.grid(alpha=0.25)
ax2.set_title("볼베어링 자료 — D = 0.4 인데 정확 p 는 0.8730", fontsize=12)

fig.suptitle("어떤 차이든 잡으려는 대신, 다섯씩으로는 거의 아무것도 "
             "잡지 못한다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TWO + "floor_needs_no_overlap.png")


# ════════════════════════════════════════════════════════════════════
# 16. kruskal_wallis — 근사가 덜 기각하는 쪽으로 틀린다
#
# 쪽의 h_is_anova_on_ranks 가 H 와 F 의 대응을, kw_rank_spread 가 평균
# 순위의 흩어짐을 그린다. 들머리의 **셋째 문단**은 그려져 있지 않다 —
# 「카이제곱 표에서 읽은 p 값은 0.0090 인데 756,756 가지를 전부 세어
# 얻은 정확한 값은 0.0024975 다. 근사가 정확값의 3.6 배, 곧 기각을
# 덜 하는 쪽으로 틀려 있다.」
# ════════════════════════════════════════════════════════════════════
# 들머리가 드는 수(9.4136 · 0.0090 · 0.0024975)는 1 문단의 비료 자료가
# 아니라 **보기 2 의 자료**에서 나온다. 독자가 방금 읽은 수와 맞추려고
# 그쪽을 쓴다. 비료 자료로 하면 같은 현상이 86 배로 나와 더 세지만,
# 들머리의 3.6 배와 어긋나 읽는 사람을 헷갈리게 한다.
G16 = [np.array([85., 78., 92., 88., 76.]),
       np.array([72., 68., 81., 75., 70.]),
       np.array([90., 95., 88., 92., 87.])]
sizes16 = [len(g) for g in G16]
N16 = sum(sizes16)


def kw_H(ranks, sizes):
    N = sum(sizes)
    pos = 0
    s = 0.0
    for n in sizes:
        s += ranks[pos:pos + n].sum() ** 2 / n
        pos += n
    return 12 / (N * (N + 1)) * s - 3 * (N + 1)


obs_H = stats.kruskal(*G16).statistic
# 이 자료에는 10 이 두 번 나와 동점이 있다. 열거도 **중간순위**로 해야
# 관측값과 같은 자를 쓴다 (동점 보정인자는 모든 배치에서 같아 약분된다).
allr = stats.rankdata(np.concatenate(G16))
obs_raw = kw_H(allr, sizes16)
cnt = tot = 0
for c1 in combinations(range(N16), sizes16[0]):
    rest = [i for i in range(N16) if i not in c1]
    for c2 in combinations(rest, sizes16[1]):
        c3 = [i for i in rest if i not in c2]
        order = np.r_[list(c1), list(c2), c3]
        if kw_H(allr[order], sizes16) >= obs_raw - 1e-9:
            cnt += 1
        tot += 1
p_exact16 = cnt / tot
p_chi16 = stats.chi2.sf(obs_H, len(G16) - 1)
print(f"\n[kruskal_wallis] H = {obs_H:.4f}, 배치 {tot:,} 가지")
print(f"  카이제곱 표 {p_chi16:.6f} · 정확 {p_exact16:.7f}  "
      f"(비 {p_chi16 / p_exact16:.2f})")

# 집단당 크기를 키우며 근사가 정확값에서 어떻게 멀어지는지 (모의실험)
r16 = np.random.default_rng(16_216)
NS16 = [3, 4, 5, 6, 8, 12, 20]
size_chi = []
for n in NS16:
    rej = 0
    REP = 20_000
    for _ in range(REP):
        g = [r16.standard_normal(n) for _ in range(3)]
        rej += stats.kruskal(*g).pvalue < 0.05
    size_chi.append(rej / REP)
print("  명목 5% 에서 카이제곱 표를 쓸 때의 실제 크기")
for n, v in zip(NS16, size_chi):
    print(f"    집단당 {n:3d}: {v:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.15]})
ax1.bar([0, 1], [p_chi16, p_exact16], color=["tab:red", "tab:blue"],
        width=0.5)
for i, v in enumerate([p_chi16, p_exact16]):
    ax1.text(i, v + 0.0004, f"{v:.5f}", ha="center", va="bottom",
             fontsize=14, fontweight="bold")
ax1.axhline(0.05, color="0.2", lw=2.0, ls="--")
ax1.set_xticks([0, 1], ["카이제곱 표에서 읽기",
                        f"{tot:,} 가지를 다 세기"], fontsize=11.5)
ax1.set_ylabel("p 값", fontsize=11.5)
ax1.set_ylim(0, max(p_chi16, 0.012) * 1.3)
ax1.grid(axis="y", alpha=0.25)
ax1.set_title(f"같은 자료, 같은 통계량 — {p_chi16 / p_exact16:.1f} 배 다르다",
              fontsize=12.5)

ax2.plot(NS16, size_chi, "o-", color="tab:red", lw=3.0, ms=12)
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xscale("log")
ax2.set_xticks(NS16, [str(n) for n in NS16])
ax2.set_xlabel("집단당 표본크기 (로그 척도) · 집단 셋", fontsize=11.5)
ax2.set_ylabel("명목 5% 인데 실제로 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, 0.065)
ax2.grid(alpha=0.25)
ax2.annotate(f"다섯씩이면 {size_chi[2]:.4f} —\n명목보다 **아래**다"
             .replace("**", ""), xy=(5, size_chi[2]), xytext=(6.0, 0.016),
             fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax2.set_title("「집단마다 다섯이면 믿을 만하다」가 놓인 자리", fontsize=12.5)

fig.suptitle("집단마다 다섯 개체면 확인할 길이 없다 — 표도 믿을 수 없다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, MG + "approximation_under_rejects.png")


# ════════════════════════════════════════════════════════════════════
# 17. dunn — 쌍의 수가 느는 속도와, 상관이 덜어 주는 몫
#
# 쪽의 dunn_fwer 가 「집단 수에 따른 집단별 오류율」을 그린다. 들머리의
# **둘째 문단**은 그 곡선 하나로는 말할 수 없다 — 「쌍별 검정들이 서로
# 독립이라면 여덟 집단에서 0.762 여야 하는데 실제는 0.526 이다. 모든
# 쌍이 같은 합친 순위에서 만들어져 서로 상관되어 있기 때문이다.」
# 독립이었다면의 곡선을 함께 그려야 그 몫이 보인다.
# ════════════════════════════════════════════════════════════════════
KS17 = [2, 3, 4, 5, 6, 8, 10]
N17, REPS17 = 8, 6000
r17 = np.random.default_rng(16_217)
fwer, indep = [], []
for k in KS17:
    m = k * (k - 1) // 2
    indep.append(1 - 0.95 ** m)
    hit = 0
    for _ in range(REPS17):
        g = [r17.standard_normal(N17) for _ in range(k)]
        allv = np.concatenate(g)
        rk = stats.rankdata(allv)
        N = len(allv)
        means = []
        pos = 0
        for x in g:
            means.append(rk[pos:pos + len(x)].mean())
            pos += len(x)
        means = np.array(means)
        se = np.sqrt(N * (N + 1) / 12 * (2 / N17))
        z = np.abs(means[:, None] - means[None, :]) / se
        iu = np.triu_indices(k, 1)
        hit += (2 * stats.norm.sf(z[iu]) < 0.05).any()
    fwer.append(hit / REPS17)
print("\n[dunn] 보정 없이 모든 쌍을 견줄 때")
for k, a, b in zip(KS17, fwer, indep):
    print(f"  집단 {k:2d} (쌍 {k * (k - 1) // 2:2d}):  실제 {a:.4f} · "
          f"독립이었다면 {b:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(KS17, indep, "s--", color="0.5", lw=2.6, ms=11,
        label="쌍별 검정이 서로 독립이었다면")
ax.plot(KS17, fwer, "o-", color="tab:red", lw=3.2, ms=14,
        label="실제 (모든 쌍이 같은 합친 순위에서 나온다)")
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.axhline(0.5, color="0.6", lw=1.6, ls=":")
for i, k in enumerate(KS17):
    if k in (5, 8):
        ax.annotate("", xy=(k, fwer[i]), xytext=(k, indep[i]),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=2.0))
        ax.text(k + 0.14, (fwer[i] + indep[i]) / 2,
                f"상관이 {indep[i] - fwer[i]:.3f} 를\n덜어 준다", ha="left",
                va="center", fontsize=11, color="0.25", fontweight="bold")
ax.set_xticks(KS17)
ax.set_xlabel("집단의 개수", fontsize=12)
ax.set_ylabel("아무 차이도 없는데 적어도 한 쌍을 유의하다고 할 확률",
              fontsize=11.5)
ax.set_ylim(0, 0.88)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
ax.text(2.1, 0.056, "명목 0.05", ha="left", va="bottom", fontsize=11,
        color="0.2", fontweight="bold")
ax.text(9.9, 0.515, "절반", ha="right", va="bottom", fontsize=11,
        color="0.5", fontweight="bold")
ax.annotate("적게 새는 것과 새지 않는 것은 다르다 —\n"
            "집단 수가 몇이든 명목을 한참 넘는다", xy=(6, fwer[4]),
            xytext=(3.1, 0.70), fontsize=12, color="tab:red",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.set_title(f"집단당 {N17} 개, 되풀이 {REPS17:,} 회", fontsize=13)
fig.suptitle("여덟 집단이면 절반은 헛것을 집는다 — 그래도 독립보다는 덜하다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, MG + "pairs_grow_but_correlate.png")


# ════════════════════════════════════════════════════════════════════
# 18. friedman — 통계량이 가질 수 있는 값이 마흔다섯 개뿐이다
#
# 쪽의 friedman_blocks_1 이 블록 사이 변동과 네 검정의 검정력을 그린다.
# (§5 가 설명하는데 없던 그림은 따로 그려 넣었다.) 들머리의 **셋째 문단
# 끝**이 남는다 — 「통계량이 가질 수 있는 값이 마흔다섯 개뿐이고 최댓값이
# 15 에서 막혀 있기 때문이다.」 그 성긴 격자 위에 관측값과 두 p 값을
# 얹으면 왜 카이제곱 표가 26 배 틀리는지가 보인다.
# ════════════════════════════════════════════════════════════════════
B18, K18 = 5, 4
perms = list(permutations(range(1, K18 + 1)))
vals = {}
for combo in product(perms, repeat=B18):
    R = np.sum(np.array(combo), axis=0)
    chi = (12 / (B18 * K18 * (K18 + 1)) * (R ** 2).sum()
           - 3 * B18 * (K18 + 1))
    key = round(chi, 9)
    vals[key] = vals.get(key, 0) + 1
keys = np.array(sorted(vals))
probs = np.array([vals[k] for k in keys], float)
probs /= probs.sum()
D18 = np.array([[30, 45, 60, 50], [25, 40, 55, 35], [35, 50, 65, 45],
                [20, 30, 50, 40], [40, 55, 70, 60]], float)
obs18 = stats.friedmanchisquare(*D18.T).statistic
p_exact18 = probs[keys >= obs18 - 1e-9].sum()
p_chi18 = stats.chi2.sf(obs18, K18 - 1)
print(f"\n[friedman] b = {B18}, k = {K18} — 값이 {len(keys)} 개")
print(f"  최댓값 {keys.max():.1f} · 관측값 {obs18:.1f}")
print(f"  정확 {p_exact18:.6f} · 카이제곱 표 {p_chi18:.6f}  "
      f"(비 {p_chi18 / p_exact18:.1f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.2, 1]})
ax1.vlines(keys, 0, probs, color="0.55", lw=2.0)
ax1.plot(keys, probs, "o", ms=5, color="0.35")
g18 = np.linspace(0, keys.max() + 1, 400)
ax1.plot(g18, stats.chi2.pdf(g18, K18 - 1), color="tab:red", lw=2.6,
         label="카이제곱 표가 쓰는 매끄러운 곡선")
ax1.axvline(obs18, color="tab:blue", lw=3.0)
ax1.annotate(f"관측값 {obs18:.1f}", xy=(obs18, 0.06), xytext=(obs18 + 0.9,
                                                           0.13),
             fontsize=12, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax1.set_xlim(-0.5, keys.max() + 1)
ax1.set_ylim(0, 0.2)
ax1.set_xlabel(f"통계량이 가질 수 있는 값 ({len(keys)} 개뿐이다)",
               fontsize=11.5)
ax1.set_ylabel("확률 / 밀도", fontsize=11.5)
ax1.legend(fontsize=10.8, loc="upper right")
ax1.grid(alpha=0.25)
ax1.set_title(f"최댓값이 {keys.max():.0f} 에서 막혀 있다", fontsize=12.5)

ax2.bar([0, 1], [p_chi18, p_exact18], color=["tab:red", "tab:blue"],
        width=0.5)
for i, v in enumerate([p_chi18, p_exact18]):
    ax2.text(i, v + 0.00012, f"{v:.6f}", ha="center", va="bottom",
             fontsize=13, fontweight="bold")
ax2.set_xticks([0, 1], ["카이제곱 표에서 읽기", "순열을 다 세기"],
               fontsize=11.5)
ax2.set_ylabel("p 값", fontsize=11.5)
ax2.set_ylim(0, p_chi18 * 1.35)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"{p_chi18 / p_exact18:.0f} 배 — 기각을 덜 하는 쪽이다",
              fontsize=12.5)

fig.suptitle("작은 설계에서는 카이제곱 표를 그대로 믿으면 안 된다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, MG + "only_forty_five_values.png")


# ════════════════════════════════════════════════════════════════════
# 19. median_test — 전체 중앙값과 똑같은 값을 어디에 넣는가
#
# 쪽의 그림 셋은 검정력 비교와 정보 손실을 다룬다. 들머리의 **넷째
# 문단**은 그려져 있지 않다 — 「전체 중앙값과 정확히 같은 값을 위로
# 넣을지 아래로 넣을지다. 마흔여덟 가운데 셋이 그 자리에 걸린 자료에서
# 그 선택만으로 p 값이 0.0639 와 0.1261 사이를 오간다.」
#
# 동점이 몇 개나 걸릴 수 있는지는 자료의 성김이 정하므로, 눈금이 거칠
# 때 그 틈이 얼마나 벌어지는지를 재는 쪽이 일반적이다.
# ════════════════════════════════════════════════════════════════════
r19 = np.random.default_rng(16_219)
LEVELS19 = [100, 20, 10, 7, 5, 4, 3]
N19, K19, REPS19 = 16, 3, 3000
gap_med, tie_med = [], []
for L in LEVELS19:
    gaps, ties = [], []
    for _ in range(REPS19):
        g = [np.ceil(r19.random(N19) * L) for _ in range(K19)]
        allv = np.concatenate(g)
        med = np.median(allv)
        ps = []
        for rule in ("above", "below"):
            tab = []
            for x in g:
                if rule == "above":
                    hi = int((x >= med).sum())
                else:
                    hi = int((x > med).sum())
                tab.append([hi, len(x) - hi])
            tab = np.array(tab, float)
            if tab.sum(0).min() == 0 or tab.sum(1).min() == 0:
                ps.append(1.0)
            else:
                ps.append(stats.chi2_contingency(tab)[1])
        gaps.append(abs(ps[0] - ps[1]))
        ties.append(int((allv == med).sum()))
    gap_med.append(np.mean(gaps))
    tie_med.append(np.mean(ties))
print("\n[median_test] 중앙값과 똑같은 값을 어디에 넣는가")
for L, g_, t_ in zip(LEVELS19, gap_med, tie_med):
    print(f"  {L:4d} 칸:  중앙값에 걸린 값 평균 {t_:5.2f} 개 · "
          f"두 선택의 p 차이 평균 {g_:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(tie_med, gap_med, "o-", color="tab:red", lw=3.2, ms=14)
for L, t_, g_ in zip(LEVELS19, tie_med, gap_med):
    if L in (100, 10, 5, 3):
        ax.annotate(f"{L} 칸", xy=(t_, g_), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=11, color="0.3", fontweight="bold")
ax.set_xlabel("전체 중앙값과 똑같은 값의 개수 (평균)", fontsize=12)
ax.set_ylabel("위로 넣을 때와 아래로 넣을 때의 p 값 차이 (평균)",
              fontsize=12)
ax.set_ylim(0, max(gap_med) * 1.25)
ax.grid(alpha=0.25)
ax.text(0.97, 0.06, "같은 자료 · 같은 검정인데\n"
        "**규칙 하나**로 p 가 이만큼 갈린다".replace("**", ""),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        color="tab:red", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="tab:red"))
ax.set_title(f"집단 {K19} 개 · 집단당 {N19} 개 · 되풀이 {REPS19:,} 회",
             fontsize=13)
fig.suptitle("사람마다 한 비트만 남긴다면 — 그 비트를 어디에 넣을 것인가",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, MG + "where_the_ties_go.png")


# ════════════════════════════════════════════════════════════════════
# 20. spearman — 축약 공식이 몰래 쓰는 가정
#
# 쪽의 spearman_exact_null 이 정확 귀무분포를, spearman_rank_straightens
# 가 순위 산점도를 그린다. 들머리의 **둘째 문단**은 그려져 있지 않다 —
# 「축약 공식은 「순위의 제곱합이 양쪽 모두 42 다」라는 사실을 몰래 쓰고
# 있다. 같은 값이 섞이면 그 제곱합이 41 과 40.5 로 줄어드는데 공식은
# 여전히 42 를 쓰므로 참값보다 크게 나온다.」
# ════════════════════════════════════════════════════════════════════
r20 = np.random.default_rng(16_220)
N20, REPS20 = 12, 3000
LV20 = [50, 12, 8, 6, 4, 3]
err_med, tie_frac20 = [], []
for L in LV20:
    errs, tfs = [], []
    for _ in range(REPS20):
        x = np.ceil(r20.random(N20) * L)
        y = np.ceil(r20.random(N20) * L)
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        true = stats.spearmanr(x, y).statistic
        d2 = ((rx - ry) ** 2).sum()
        short = 1 - 6 * d2 / (N20 * (N20 ** 2 - 1))
        errs.append(short - true)
        _, c1 = np.unique(x, return_counts=True)
        _, c2 = np.unique(y, return_counts=True)
        tfs.append((c1[c1 > 1].sum() + c2[c2 > 1].sum()) / (2 * N20))
    err_med.append(np.mean(errs))
    tie_frac20.append(np.mean(tfs))
print("\n[spearman] 축약 공식과 참값의 차이")
for L, t_, e_ in zip(LV20, tie_frac20, err_med):
    print(f"  {L:3d} 칸 (동점에 묶인 비율 {t_:.3f}):  "
          f"축약 - 참값 = {e_:+.5f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(tie_frac20, err_med, "o-", color="tab:red", lw=3.2, ms=13)
ax1.axhline(0, color="0.2", lw=2.2, ls="--")
for L, t_, e_ in zip(LV20, tie_frac20, err_med):
    if L in (50, 6, 3, 2):
        ax1.annotate(f"{L} 칸", xy=(t_, e_), xytext=(0, 13),
                     textcoords="offset points", ha="center",
                     fontsize=11, color="0.3", fontweight="bold")
ax1.set_xlabel("동점에 묶인 관측값의 비율", fontsize=11.5)
ax1.set_ylabel("축약 공식 빼기 참값 (평균)", fontsize=11.5)
ax1.grid(alpha=0.25)
ax1.set_title("동점이 늘수록 축약 공식이 위로 틀어진다", fontsize=12.5)

ss = np.arange(1, N20 + 1, dtype=float)
base = ((ss - ss.mean()) ** 2).sum()
tiecounts = np.arange(0, 7)
shrunk = []
for m in tiecounts:
    r = ss.copy()
    for j in range(m):
        a_, b_ = 2 * j, 2 * j + 1
        if b_ < N20:
            r[[a_, b_]] = (r[a_] + r[b_]) / 2
    shrunk.append(((r - r.mean()) ** 2).sum())
ax2.plot(tiecounts, shrunk, "o-", color="tab:blue", lw=3.2, ms=13)
ax2.axhline(base, color="tab:red", lw=2.4, ls="--")
ax2.text(tiecounts[-1], base + 1.5,
         f"축약 공식이 늘 쓰는 값 {base:.0f}", ha="right", va="bottom",
         fontsize=11.5, color="tab:red", fontweight="bold")
ax2.set_xlabel("같은 값으로 묶인 쌍의 개수", fontsize=11.5)
ax2.set_ylabel(f"순위의 제곱합 (n = {N20})", fontsize=11.5)
ax2.grid(alpha=0.25)
ax2.set_title("동점이 생기면 제곱합이 실제로 줄어든다", fontsize=12.5)
ax2.text(0.5, 0.08, "공식은 줄어든 것을 모르고 큰 수로 나눈다 —\n"
         "그래서 결과가 참값보다 **크게** 나온다".replace("**", ""),
         transform=ax2.transAxes, ha="center", va="bottom", fontsize=11.3,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("외워 둔 공식이 틀린 값을 준다 — 몰래 쓰는 가정 하나 때문에",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, COR + "the_shortcut_assumes_fortytwo.png")


# ════════════════════════════════════════════════════════════════════
# 21. kendall — 타우와 스피어만은 같은 자가 아니다
#
# 쪽의 kendall_exact_recursion 이 정확 귀무분포를, kendall_pairs 가
# 쌍별 선분을 그린다. 들머리의 **넷째 문단**은 그려져 있지 않다 —
# 「이변량 정규 자료에서 두 값의 비를 재면 중앙값이 1.41 이라서,
# 타우 0.3 과 스피어만 0.3 은 같은 세기의 연관이 아니다.」
# ════════════════════════════════════════════════════════════════════
r21 = np.random.default_rng(16_221)
RHOS = np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9])
N21, REPS21 = 60, 1500
tau_med, rho_med = [], []
for rho in RHOS:
    ts, rs = [], []
    for _ in range(REPS21 // 3):
        z = r21.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N21)
        ts.append(stats.kendalltau(z[:, 0], z[:, 1]).statistic)
        rs.append(stats.spearmanr(z[:, 0], z[:, 1]).statistic)
    tau_med.append(np.median(ts))
    rho_med.append(np.median(rs))
tau_med = np.array(tau_med)
rho_med = np.array(rho_med)
ratio21 = rho_med / tau_med
print(f"\n[kendall] 이변량 정규에서 두 계수 (n = {N21})")
for rho, t_, s_, q in zip(RHOS, tau_med, rho_med, ratio21):
    print(f"  참 상관 {rho:4.2f}:  타우 {t_:.4f} · 스피어만 {s_:.4f}  "
          f"(비 {q:.3f})")
# 이론값 — tau = (2/pi) arcsin(rho), rho_s = (6/pi) arcsin(rho/2)
tau_th = 2 / np.pi * np.arcsin(RHOS)
rho_th = 6 / np.pi * np.arcsin(RHOS / 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(RHOS, tau_th, "-", color="0.72", lw=5.5, alpha=0.8,
         label="타우 (이론)")
ax1.plot(RHOS, rho_th, "-", color="#f6c6c6", lw=5.5, alpha=0.9,
         label="스피어만 (이론)")
ax1.plot(RHOS, tau_med, "o", color="tab:blue", ms=12, label="타우 (재어 봄)")
ax1.plot(RHOS, rho_med, "s", color="tab:red", ms=11,
         label="스피어만 (재어 봄)")
ax1.plot([0, 0.95], [0, 0.95], color="0.3", lw=1.6, ls=":")
ax1.set_xlabel("바탕의 참 상관", fontsize=11.5)
ax1.set_ylabel("계수의 가운데값", fontsize=11.5)
ax1.set_xlim(0, 0.95)
ax1.set_ylim(0, 0.95)
ax1.legend(fontsize=10.6, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_title("같은 자료에서 두 계수가 다른 수를 낸다", fontsize=12.5)

ax2.plot(tau_med, ratio21, "o-", color="tab:purple", lw=3.2, ms=13)
ax2.axhline(1.0, color="0.2", lw=2.2, ls="--")
ax2.set_xlabel("타우의 값", fontsize=11.5)
ax2.set_ylabel("같은 자료의 스피어만 ÷ 타우", fontsize=11.5)
ax2.set_ylim(0.95, max(ratio21) * 1.1)
ax2.grid(alpha=0.25)
j3 = int(np.argmin(np.abs(tau_med - 0.3)))
ax2.annotate(f"타우가 {tau_med[j3]:.2f} 인 자료에서\n"
             f"스피어만은 {rho_med[j3]:.2f} 다 ({ratio21[j3]:.2f} 배)",
             xy=(tau_med[j3], ratio21[j3]),
             xytext=(tau_med[j3] + 0.08, max(ratio21) * 1.02),
             fontsize=11.5, color="tab:purple", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.9))
ax2.set_title("두 계수는 같은 척도가 아니다", fontsize=12.5)

fig.suptitle("타우 0.3 과 스피어만 0.3 은 같은 세기의 연관이 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, COR + "tau_and_rho_are_not_one_ruler.png")
