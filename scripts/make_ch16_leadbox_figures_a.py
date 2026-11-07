"""16장 들어가는 이야기 그림 — 첫 그룹 일곱.

  1. index                  값을 바꿔도 순서는 그대로다
  2. motivation             칸 사이 간격을 아무도 보장하지 않았다
  3. ranks                  동점을 평균으로 메우는 법과 그 대가
  4. nonparametric_suite    짝을 버리면 p 는 커지기만 한다
  5. power_comparison       효과가 뚜렷해지면 이득이 1 쪽으로 눌린다
  6. sign_test              부호검정이 보는 것은 평균이 아니라 중앙값이다
  7. wilcoxon_signed_rank   크기를 들여와 번 것

만드는 파일:
  docs/ch16/img/order_survives_the_value.png
  docs/ch16/foundations/img/the_spacing_was_never_promised.png
  docs/ch16/foundations/img/ties_cost_this.png
  docs/ch16/foundations/img/dropping_pairs_only_raises_p.png
  docs/ch16/foundations/img/gain_shrinks_as_effect_grows.png
  docs/ch16/one_sample_nonparametric/img/median_not_mean.png
  docs/ch16/one_sample_nonparametric/img/what_magnitude_buys.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

IDX = "docs/ch16/img/"
FND = "docs/ch16/foundations/img/"
ONE = "docs/ch16/one_sample_nonparametric/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. 장 개요 — 값을 바꿔도 순서는 그대로다
#
# 쪽에 그림이 하나도 없다. 들머리의 알맹이는 한 문장이다 — 「1 이든
# 100 이든 「가장 큰 값」이기는 마찬가지이기 때문이다.」 장 개요는 독자가
# 순위도 검정도 아직 배우지 않은 자리이므로 **기제**를 보인다. 죽은 벌레
# 수처럼 0 이 수두룩하고 큰 수가 하나 섞인 여덟 마리짜리 자료에서,
# 그 큰 값만 바꿔 가며 두 방법의 답이 어떻게 되는지 본다.
# ════════════════════════════════════════════════════════════════════
A1 = np.array([0, 0, 1, 0, 2, 0, 1, 0], float)       # 약을 뿌리지 않은 쪽
B1 = np.array([3, 5, 2, 4, 8, 3, 6, 4], float)       # 약을 뿌린 쪽
BIGS = np.array([8, 20, 50, 100, 300, 1000], float)
p_t, p_rank = [], []
for big in BIGS:
    b = B1.copy()
    b[4] = big
    p_t.append(stats.ttest_ind(A1, b, equal_var=False).pvalue)
    p_rank.append(stats.mannwhitneyu(A1, b, alternative="two-sided").pvalue)
print("[장 개요] 가장 큰 값 하나만 바꿔 본다 (여덟 마리씩)")
for big, a, b in zip(BIGS, p_t, p_rank):
    print(f"  가장 큰 값 {big:7.0f}:  평균을 쓰는 검정 p = {a:.4f} · "
          f"순서만 쓰는 검정 p = {b:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.15]})
b_show = B1.copy()
b_show[4] = 100
# 0 이 여럿이라 그대로 찍으면 점이 겹쳐 여덟 개가 보이지 않는다.
# 같은 값끼리 세로로 조금 벌려 놓는다.
def spread(vals, base):
    y = np.full(len(vals), float(base))
    for v in np.unique(vals):
        idx = np.where(vals == v)[0]
        if len(idx) > 1:
            off = np.linspace(-0.16, 0.16, len(idx))
            y[idx] = base + off
    return y


allv = np.r_[A1, b_show]
rk = stats.rankdata(allv)
ya = spread(A1, 1.0)
yb = spread(b_show, 2.0)
ax1.plot(A1, ya, "o", ms=13, color="0.55", label="약을 뿌리지 않은 쪽")
ax1.plot(b_show, yb, "o", ms=13, color="tab:blue", label="약을 뿌린 쪽")
for i, (v, y) in enumerate(zip(A1, ya)):
    ax1.text(max(v, 0.012), y + 0.10, f"{rk[i]:.0f}", ha="center",
             va="bottom", fontsize=9.5, color="0.35")
for i, (v, y) in enumerate(zip(b_show, yb)):
    ax1.text(v, y + 0.10, f"{rk[len(A1) + i]:.0f}", ha="center",
             va="bottom", fontsize=9.5, color="tab:blue")
ax1.set_xscale("symlog", linthresh=1)
ax1.set_xticks([0, 1, 2, 5, 10, 100], ["0", "1", "2", "5", "10", "100"])
ax1.set_yticks([1, 2], ["안 뿌린 쪽", "뿌린 쪽"], fontsize=12.5)
ax1.set_ylim(0.45, 2.75)
ax1.set_xlabel("죽은 벌레 수 (가로를 눌러 그렸다)", fontsize=11.5)
ax1.set_title("여덟 마리씩 — 0 이 수두룩하고 큰 수가 하나", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)
ax1.legend(fontsize=10.8, loc="upper left")
ax1.text(0.5, 0.03, "점 위의 수는 열여섯 개를 통틀어 매긴 순위다",
         transform=ax1.transAxes, ha="center", va="bottom", fontsize=10.8,
         color="0.35", fontweight="bold")

ax2.plot(BIGS, p_t, "s-", color="tab:red", lw=3.0, ms=12,
         label="평균을 쓰는 검정")
ax2.plot(BIGS, p_rank, "o-", color="tab:blue", lw=3.0, ms=12,
         label="순서만 쓰는 검정")
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xscale("log")
ax2.set_xticks(BIGS, [f"{b:.0f}" for b in BIGS])
ax2.set_xlabel("가장 큰 값 하나를 바꿔 본다 (나머지 열다섯은 그대로)",
               fontsize=11.5)
ax2.set_ylabel("두 쪽이 다르다는 증거의 세기 (p 값)", fontsize=11.5)
ax2.set_ylim(0, max(p_t) * 1.25)
ax2.legend(fontsize=11.5, loc="upper left")
ax2.grid(alpha=0.25)
ax2.text(0.97, 0.52, "순서만 쓰면 한 자리도 움직이지 않는다 —\n"
         "1 이든 1,000 이든 「가장 큰 값」이기는 마찬가지다",
         transform=ax2.transAxes, ha="right", va="center", fontsize=11.5,
         color="tab:blue", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="tab:blue"))
ax2.text(0.03, 0.055, "0.05", transform=ax2.transAxes, ha="left",
         va="bottom", fontsize=11, color="0.2", fontweight="bold")

fig.suptitle("값을 버리고 순서만 남기면 어떤 모양이든 답이 흔들리지 않는다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, IDX + "order_survives_the_value.png")


# ════════════════════════════════════════════════════════════════════
# 2. motivation — 칸 사이 간격을 아무도 보장하지 않았다
#
# 쪽의 t_skew_asymmetry 는 「지수분포 자료에서 t 와 이론 귀무분포의
# 불일치」을 그려 들머리의 둘째·셋째 문단을 담는다. **첫 문단**은 전혀
# 다른 이야기인데 그려져 있지 않다 — 「3 과 4 의 간격이 8 과 9 의
# 간격과 같다고 아무도 보장하지 않았다.」
#
# 그 말을 재려면 같은 응답에 **다른 눈금**을 씌워 보면 된다. 환자가
# 고른 칸은 그대로인데 칸에 매기는 수만 바꾸면, 평균을 쓰는 검정의 답은
# 바뀌고 순위를 쓰는 검정의 답은 바뀌지 않는다.
# ════════════════════════════════════════════════════════════════════
r2 = np.random.default_rng(16_002)
NT, NC = 12, 10
probs_t = np.array([.05, .08, .10, .12, .14, .14, .13, .11, .08, .05])
probs_c = np.array([.02, .04, .06, .09, .12, .15, .16, .15, .12, .09])
# 둘 다 1~10 칸을 고르는 **순서**는 같고, 칸에 매기는 수만 다르다.
SCALES = {
    "고른 눈금": np.arange(1, 11, dtype=float),
    "위가 벌어진 눈금": np.array([1, 2, 3, 4, 5, 6.5, 8.5, 11, 15, 22.]),
    "아래가 벌어진 눈금": np.array([1, 4, 7, 10, 12, 14, 15.5, 17, 18, 19.]),
}
REPS2 = 6000
names2 = list(SCALES)
flip = 0
pvals = {k: [] for k in names2}
p_rank = []
for _ in range(REPS2):
    a_box = r2.choice(np.arange(10), NT, p=probs_t)
    b_box = r2.choice(np.arange(10), NC, p=probs_c)
    verd = []
    for k, sc in SCALES.items():
        pv = stats.ttest_ind(sc[a_box], sc[b_box], equal_var=False).pvalue
        pvals[k].append(pv)
        verd.append(pv < 0.05)
    flip += (len(set(verd)) > 1)
    p_rank.append(stats.mannwhitneyu(a_box, b_box,
                                     alternative="two-sided").pvalue)
for k in names2:
    pvals[k] = np.array(pvals[k])
p_rank = np.array(p_rank)
print("\n[motivation] 고른 칸은 그대로 두고 칸에 매기는 수만 바꾼다")
for k in names2:
    print(f"  {k:14s} 기각률 {np.mean(pvals[k] < 0.05):.4f}")
print(f"  {'순위를 쓰면':14s} 기각률 {np.mean(p_rank < 0.05):.4f} "
      f"(어느 눈금에서나 같은 수다)")
print(f"  눈금에 따라 **판정이 뒤집힌** 표본 {flip}/{REPS2} "
      f"= {flip / REPS2:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.1]})
for (k, sc), c in zip(SCALES.items(), ["0.35", "tab:red", "tab:green"]):
    ax1.plot(np.arange(1, 11), sc, "o-", color=c, lw=2.6, ms=10, label=k)
ax1.set_xticks(np.arange(1, 11))
ax1.set_xlabel("환자가 고른 칸 (세 척도에서 모두 같다)", fontsize=11.5)
ax1.set_ylabel("그 칸에 매긴 수", fontsize=11.5)
ax1.set_title("칸의 순서는 같고 간격만 다르다", fontsize=12.5)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(alpha=0.25)

ax2.plot(pvals["위가 벌어진 눈금"], pvals["아래가 벌어진 눈금"], "o",
         ms=3.0, color="0.55", alpha=0.35)
ax2.axhline(0.05, color="tab:red", lw=2.2, ls="--")
ax2.axvline(0.05, color="tab:red", lw=2.2, ls="--")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks([1e-4, 1e-3, 0.01, 0.05, 0.5],
               ["0.0001", "0.001", "0.01", "0.05", "0.5"])
ax2.set_yticks([1e-4, 1e-3, 0.01, 0.05, 0.5],
               ["0.0001", "0.001", "0.01", "0.05", "0.5"])
ax2.set_xlim(1e-5, 1.3)
ax2.set_ylim(1e-5, 1.3)
ax2.set_xlabel("위가 벌어진 척도로 잰 p 값", fontsize=11.5)
ax2.set_ylabel("아래가 벌어진 척도로 잰 p 값", fontsize=11.5)
ax2.set_title("같은 응답, 다른 척도 — 네 칸에 다 흩어진다", fontsize=12.5)
ax2.grid(alpha=0.25, which="both")
ax2.text(0.97, 0.955, f"판정이 뒤집힌 표본이\n"
         f"{REPS2} 벌 가운데 {flip} 벌 ({100 * flip / REPS2:.1f}%)",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11.5,
         color="tab:red", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="tab:red"))
ax2.text(0.03, 0.05, "순위를 쓰면 이 산점도가 생기지 않는다 —\n"
         "세 눈금이 **한 점**을 준다".replace("**", ""),
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=11.3,
         color="tab:blue", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="tab:blue"))

fig.suptitle("3 과 4 의 간격이 8 과 9 의 간격과 같다고 아무도 보장하지 "
             "않았다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FND + "the_spacing_was_never_promised.png")


# ════════════════════════════════════════════════════════════════════
# 3. ranks — 동점을 평균으로 메우는 법과 그 대가
#
# 쪽의 information_ladder 가 「같은 자료를 세 단계로 줄여 가며」를 그려
# 들머리의 셋째 문단을 담고, rank_transform 이 첫 문단을 담는다. 남은
# 것은 **둘째 문단의 뒷부분**이다 — 동점을 평균으로 메우는 일과 「그
# 대가를 세는 보정계수」. 동점이 많아질 때 보정이 무엇을 하는지는
# 그려져 있지 않다.
# ════════════════════════════════════════════════════════════════════
r3 = np.random.default_rng(16_003)
LEVELS = [200, 50, 20, 10, 6, 4, 3, 2]     # 값을 몇 칸으로 뭉갤 것인가
N3, REPS3 = 20, 20_000
size_corr, size_naive, tie_frac = [], [], []
for L in LEVELS:
    hit_c = hit_n = 0
    tf = []
    for _ in range(REPS3 // 4):
        a = np.ceil(r3.random(N3) * L)
        b = np.ceil(r3.random(N3) * L)
        allv = np.r_[a, b]
        rk = stats.rankdata(allv)
        _, cnt = np.unique(allv, return_counts=True)
        tf.append((cnt[cnt > 1].sum()) / len(allv))
        # 보정한 쪽 — scipy 가 동점 보정을 넣어 준다
        hit_c += stats.mannwhitneyu(a, b, alternative="two-sided",
                                    method="asymptotic").pvalue < 0.05
        # 보정하지 않은 쪽 — 동점이 없다고 보고 분산을 쓴다
        n1, n2, n = N3, N3, 2 * N3
        U = rk[:N3].sum() - n1 * (n1 + 1) / 2
        mu = n1 * n2 / 2
        sd = np.sqrt(n1 * n2 * (n + 1) / 12)
        hit_n += 2 * stats.norm.sf(abs(U - mu) / sd) < 0.05
    size_corr.append(hit_c / (REPS3 // 4))
    size_naive.append(hit_n / (REPS3 // 4))
    tie_frac.append(np.mean(tf))
print("\n[ranks] 값을 몇 칸으로 뭉갤 때 동점 보정이 하는 일")
for L, tf, a, b in zip(LEVELS, tie_frac, size_corr, size_naive):
    print(f"  {L:4d} 칸 (동점에 묶인 비율 {tf:.3f}):  보정 {a:.4f} · "
          f"보정 없이 {b:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(tie_frac, size_naive, "s-", color="tab:red", lw=3.0, ms=12,
        label="동점이 없다고 보고 쓴 분산")
ax.plot(tie_frac, size_corr, "o-", color="tab:blue", lw=3.0, ms=12,
        label="동점 보정을 넣은 분산")
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
for i, L in enumerate(LEVELS):
    if L in (200, 10, 3, 2):
        ax.annotate(f"{L} 칸", xy=(tie_frac[i], size_naive[i]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=10.5, color="0.3",
                    fontweight="bold")
ax.set_xlabel("동점에 묶인 관측값의 비율", fontsize=12)
ax.set_ylabel("두 집단이 같은데 다르다고 답하는 비율", fontsize=12)
ax.set_ylim(0, max(max(size_naive), 0.06) * 1.25)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
ax.text(0.97, 0.06, "보정은 동점이 줄여 놓은 분산을 되돌려 준다 —\n"
        "넣지 않으면 분모가 너무 커서 **덜** 기각한다".replace("**", ""),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5,
        color="0.2", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))
ax.set_title(f"집단당 {N3} 개, 두 집단이 같은 분포에서 왔다", fontsize=13)
fig.suptitle("동점을 평균으로 메우면 분산이 줄고, 그 몫이 보정계수다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, FND + "ties_cost_this.png")


# ════════════════════════════════════════════════════════════════════
# 4. nonparametric_suite — 짝을 버리면 p 는 커지기만 한다
#
# 쪽의 paired_structure_lost 가 「대응 구조를 살린 그림과 버린 그림」을,
# test_shopping 이 넷을 다 돌려 보는 일을 그린다. 들머리의 둘째 문단은
# 방향을 못 박는다 — 「설계로 벌어 둔 검정력을 반납하는 쪽이라 p 값은
# **커지기만** 한다.」 「늘」이라는 말은 **한 자리의 그림으로는 못 보이고**
# 많은 표본을 흩뿌려야 보인다.
# ════════════════════════════════════════════════════════════════════
r4 = np.random.default_rng(16_004)
N4, REPS4 = 15, 1200
base = 75.0
ps_pair, ps_ind = [], []
for _ in range(REPS4):
    ability = r4.normal(0, 12, N4)          # 학생마다 다른 실력
    before = base + ability + r4.normal(0, 4, N4)
    after = before + 5.5 + r4.normal(0, 4, N4)
    ps_pair.append(stats.wilcoxon(after - before).pvalue)
    ps_ind.append(stats.mannwhitneyu(after, before,
                                     alternative="two-sided").pvalue)
ps_pair = np.array(ps_pair)
ps_ind = np.array(ps_ind)
print("\n[nonparametric_suite] 같은 자료를 짝으로 보기와 독립으로 보기")
print(f"  짝을 버린 쪽의 p 가 더 큰 표본 "
      f"{(ps_ind > ps_pair).sum()}/{REPS4}")
print(f"  0.05 에서 기각: 짝을 살리면 {(ps_pair < 0.05).mean():.4f} · "
      f"버리면 {(ps_ind < 0.05).mean():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(np.maximum(ps_pair, 1e-12), np.maximum(ps_ind, 1e-12), "o",
         ms=3.4, color="tab:blue", alpha=0.35)
lim = [1e-12, 1.2]
ax1.plot(lim, lim, color="tab:red", lw=2.4, ls="--",
         label="두 p 가 같다면 이 선 위")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks([1e-10, 1e-6, 1e-3, 0.05, 1],
               ["1e-10", "1e-6", "0.001", "0.05", "1"])
ax1.set_yticks([1e-10, 1e-6, 1e-3, 0.05, 1],
               ["1e-10", "1e-6", "0.001", "0.05", "1"])
ax1.set_xlim(1e-12, 1.4)
ax1.set_ylim(1e-12, 1.4)
ax1.set_xlabel("짝을 살린 검정의 p 값", fontsize=11.5)
ax1.set_ylabel("짝을 버린 검정의 p 값", fontsize=11.5)
ax1.set_title(f"표본 {REPS4} 벌이 모두 선 위쪽에 있다", fontsize=12.5)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25, which="both")

ax2.hist(np.log10(np.maximum(ps_ind, 1e-12)
                  / np.maximum(ps_pair, 1e-12)), bins=50,
         color="tab:blue", alpha=0.75)
ax2.axvline(0, color="tab:red", lw=2.6, ls="--")
ax2.set_xlabel("짝을 버린 p 가 몇 배 커졌는가 (상용로그)", fontsize=11.5)
ax2.set_ylabel("표본 수", fontsize=11.5)
ax2.set_title("한 벌도 왼쪽으로 가지 않는다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)
med = np.median(np.log10(np.maximum(ps_ind, 1e-12)
                         / np.maximum(ps_pair, 1e-12)))
ax2.text(0.97, 0.93, f"가운데값이 {10 ** med:,.0f} 배다", ha="right",
         va="top", transform=ax2.transAxes, fontsize=12.5, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))

fig.suptitle("설계로 벌어 둔 것을 반납하면 p 는 커지기만 한다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, FND + "dropping_pairs_only_raises_p.png")


# ════════════════════════════════════════════════════════════════════
# 5. power_comparison — 효과가 뚜렷해지면 이득이 1 쪽으로 눌린다
#
# 쪽의 worst_case_density 와 are_power_curves 가 ARE 와 검정력 곡선을
# 그린다. 들머리의 **셋째 문단 끝**이 남는다 — 「이 수들은 모두 효과가
# 0 으로 작아질 때의 극한값이다. 효과가 뚜렷하면 비가 1 쪽으로 눌려
# Laplace 가 1.42, t(3) 이 1.70 에 그친다.」 효과크기를 가로축에 놓은
# 그림이 없으므로 그것만 그린다.
# ════════════════════════════════════════════════════════════════════
r5 = np.random.default_rng(16_005)
DELTAS = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
N5, REPS5 = 25, 12_000
FAM5 = [("정규", lambda m, n: r5.standard_normal((m, n))),
        ("라플라스", lambda m, n: r5.laplace(0, 1 / np.sqrt(2), (m, n))),
        ("t(3)", lambda m, n: r5.standard_t(3, (m, n)) / np.sqrt(3))]
eff = {nm: [] for nm, _ in FAM5}
for nm, gen in FAM5:
    for d in DELTAS:
        x = gen(REPS5, N5) + d
        pw_t = (stats.ttest_1samp(x, 0, axis=1).pvalue < 0.05).mean()
        pw_w = np.mean([stats.wilcoxon(row).pvalue < 0.05
                        for row in x[:2500]])
        eff[nm].append((pw_w, pw_t))
print("\n[power_comparison] 효과를 키우며 두 검정의 검정력")
for nm, _ in FAM5:
    print(f"  {nm:6s}: " + " · ".join(
        f"{d:.2f}→{w:.3f}/{t:.3f}" for d, (w, t) in zip(DELTAS, eff[nm])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
COLS5 = ["0.4", "tab:blue", "tab:red"]
for (nm, _), c in zip(FAM5, COLS5):
    ax1.plot(DELTAS, [w for w, _ in eff[nm]], "o-", color=c, lw=2.8,
             ms=10, label=f"{nm} — 순위검정")
    ax1.plot(DELTAS, [t for _, t in eff[nm]], "s--", color=c, lw=2.0,
             ms=8, alpha=0.65)
ax1.set_xlabel("참 중앙값이 0 에서 떨어진 양", fontsize=11.5)
ax1.set_ylabel("기각하는 비율", fontsize=11.5)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=10.6, loc="lower right")
ax1.grid(alpha=0.25)
ax1.set_title("실선이 순위검정, 파선이 t 검정", fontsize=12.5)

for (nm, _), c in zip(FAM5, COLS5):
    ratio = [w / t if t > 0 else np.nan for w, t in eff[nm]]
    ax2.plot(DELTAS, ratio, "o-", color=c, lw=3.0, ms=11, label=nm)
ax2.axhline(1.0, color="0.2", lw=2.2, ls="--")
ax2.set_xlabel("참 중앙값이 0 에서 떨어진 양", fontsize=11.5)
ax2.set_ylabel("순위검정의 검정력 ÷ t 검정의 검정력", fontsize=11.5)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title("효과가 뚜렷해질수록 1 쪽으로 눌린다", fontsize=12.5)
ax2.text(0.03, 0.06, "줄어드는 것은 **이득의 크기**이고\n"
         "어느 쪽이 이기는지는 그대로다".replace("**", ""),
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=11.3,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle(f"손해를 수로 받아 보면 — 그 수가 효과크기에 따라 움직인다 "
             f"(n = {N5})", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FND + "gain_shrinks_as_effect_grows.png")


# ════════════════════════════════════════════════════════════════════
# 6. sign_test — 부호검정이 보는 것은 평균이 아니라 중앙값이다
#
# 쪽의 magnitude_discarded 가 「부호가 같고 크기가 다른 두 자료」를,
# normal_vs_exact_tail 이 근사의 꼬리를 그린다. 들머리가 적는 한 줄 —
# 「중앙값에 대한 물음을 공정한 동전에 대한 물음으로 바꾼다」 — 이것이
# 함의하는 것은 그려져 있지 않다. **부호검정이 재는 모수가 평균이
# 아니라는 것**이다. 치우친 분포에서 평균과 중앙값이 갈리면 두 검정이
# 서로 다른 것을 묻는 것이 드러난다.
# ════════════════════════════════════════════════════════════════════
r6 = np.random.default_rng(16_006)
N6, REPS6 = 30, 20_000
SIG6 = 0.9
med6 = 1.0
mean6 = np.exp(SIG6 ** 2 / 2)
print(f"\n[sign_test] 대수정규 — 중앙값 {med6:.4f} · 평균 {mean6:.4f}")
d6 = np.exp(r6.normal(0, SIG6, size=(REPS6, N6)))
rows = []
for h0, nm in [(med6, "중앙값"), (mean6, "평균")]:
    pos = (d6 > h0).sum(1)
    p_sign = 2 * np.minimum(stats.binom.cdf(pos, N6, 0.5),
                            stats.binom.sf(pos - 1, N6, 0.5))
    rej_sign = (np.minimum(p_sign, 1.0) < 0.05).mean()
    rej_t = (stats.ttest_1samp(d6, h0, axis=1).pvalue < 0.05).mean()
    rows.append((nm, h0, rej_sign, rej_t))
    print(f"  귀무값을 {nm}({h0:.4f}) 으로 두면:  부호검정 {rej_sign:.4f} · "
          f"t 검정 {rej_t:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.1, 1]})
g = np.linspace(0.01, 6, 500)
ax1.plot(g, stats.lognorm.pdf(g, SIG6), color="0.3", lw=3.0)
ax1.fill_between(g, 0, stats.lognorm.pdf(g, SIG6), color="0.85")
ax1.axvline(med6, color="tab:blue", lw=3.0)
ax1.axvline(mean6, color="tab:red", lw=3.0, ls="--")
ax1.text(med6 - 0.07, 0.68, f"중앙값 {med6:.3f}", ha="right", va="top",
         fontsize=12.5, color="tab:blue", fontweight="bold", rotation=90)
ax1.text(mean6 + 0.09, 0.68, f"평균 {mean6:.3f}", ha="left", va="top",
         fontsize=12.5, color="tab:red", fontweight="bold", rotation=90)
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 0.75)
ax1.set_xlabel("값", fontsize=11.5)
ax1.set_ylabel("밀도", fontsize=11.5)
ax1.set_title("치우치면 평균과 중앙값이 갈린다", fontsize=12.5)
ax1.grid(alpha=0.25)

xx = np.arange(2)
w = 0.36
ax2.bar(xx - w / 2, [rows[0][2], rows[1][2]], width=w, color="tab:blue",
        label="부호검정")
ax2.bar(xx + w / 2, [rows[0][3], rows[1][3]], width=w, color="tab:red",
        label="t 검정")
for i in xx:
    ax2.text(i - w / 2, rows[i][2] + 0.015, f"{rows[i][2]:.3f}",
             ha="center", va="bottom", fontsize=11.5, fontweight="bold",
             color="tab:blue")
    ax2.text(i + w / 2, rows[i][3] + 0.015, f"{rows[i][3]:.3f}",
             ha="center", va="bottom", fontsize=11.5, fontweight="bold",
             color="tab:red")
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xticks(xx, [f"귀무값 = 중앙값 ({med6:.3f})",
                    f"귀무값 = 평균 ({mean6:.3f})"], fontsize=11.5)
ax2.set_ylabel("기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, max(rows[0][2], rows[1][3]) * 1.3)
ax2.legend(fontsize=11.5, loc="upper left")
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("각자 자기 모수에서만 0.05 를 지킨다", fontsize=12.5)

fig.suptitle("부호검정이 공정한 동전으로 바꿔 묻는 것은 중앙값이다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, ONE + "median_not_mean.png")


# ════════════════════════════════════════════════════════════════════
# 7. wilcoxon_signed_rank — 크기를 들여와 번 것
#
# 쪽의 symmetry_broken 이 「대칭성이 깨졌을 때 실제로 검정하는 것」을
# 그려 들머리의 둘째·셋째 문단을 담는다. 첫 문단은 벌이 쪽이다 —
# 「얼마나 벗어났는지를 순위로 바꾸어 함께 넣으면 같은 자료에서 결론이
# 또렷해진다 (0.0209 에서 0.0086 으로).」 그 벌이가 일반적으로 얼마인지는
# 그려져 있지 않다. **대칭이 지켜지는 자리**에서만 재는 것이 정직하다.
# ════════════════════════════════════════════════════════════════════
r7 = np.random.default_rng(16_007)
NS7 = [8, 12, 20, 30, 50]
DELTA7 = 0.5
REPS7 = 4000
pw_sign, pw_sr = [], []
for n in NS7:
    x = r7.standard_normal((REPS7, n)) + DELTA7
    pos = (x > 0).sum(1)
    p_sign = 2 * np.minimum(stats.binom.cdf(pos, n, 0.5),
                            stats.binom.sf(pos - 1, n, 0.5))
    pw_sign.append((np.minimum(p_sign, 1.0) < 0.05).mean())
    pw_sr.append(np.mean([stats.wilcoxon(row).pvalue < 0.05 for row in x]))
print(f"\n[wilcoxon_signed_rank] 대칭인 자리에서 (참 중앙값 {DELTA7})")
for n, a, b in zip(NS7, pw_sign, pw_sr):
    print(f"  n = {n:3d}:  부호검정 {a:.4f} · 부호순위검정 {b:.4f}  "
          f"(더 잡는 양 {b - a:+.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS7, pw_sign, "s-", color="0.5", lw=3.0, ms=12,
         label="부호만 세는 검정")
ax1.plot(NS7, pw_sr, "o-", color="tab:blue", lw=3.0, ms=12,
         label="크기의 순위까지 쓰는 검정")
for i, n in enumerate(NS7):
    if n in (12, 30):
        ax1.annotate("", xy=(n, pw_sr[i]), xytext=(n, pw_sign[i]),
                     arrowprops=dict(arrowstyle="<->", color="0.3", lw=2.0))
        ax1.text(n + 1.2, (pw_sr[i] + pw_sign[i]) / 2,
                 f"{pw_sr[i] - pw_sign[i]:+.3f}", ha="left", va="center",
                 fontsize=11.5, color="0.2", fontweight="bold")
ax1.set_xlabel("표본크기 n (대칭인 정규 자료)", fontsize=11.5)
ax1.set_ylabel("참으로 다를 때 잡아내는 비율", fontsize=11.5)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25)
ax1.set_title("크기를 들여오면 이만큼 더 잡는다", fontsize=12.5)

ax2.plot(NS7, [b / a if a > 0 else np.nan
               for a, b in zip(pw_sign, pw_sr)], "o-", color="tab:blue",
         lw=3.0, ms=12)
ax2.axhline(1.0, color="0.2", lw=2.2, ls="--")
ax2.set_xlabel("표본크기 n", fontsize=11.5)
ax2.set_ylabel("부호순위검정의 검정력 ÷ 부호검정의 검정력", fontsize=11.5)
ax2.grid(alpha=0.25)
ax2.set_title("벌이는 작은 표본에서 가장 크다", fontsize=12.5)
ax2.text(0.97, 0.93, "그 대신 차이의 분포가 0 을 중심으로\n"
         "대칭이라는 가정을 새로 들여야 한다", transform=ax2.transAxes,
         ha="right", va="top", fontsize=11.3, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))

fig.suptitle("크기를 다시 들여오면 가정도 하나 들어온다 — 그 교환의 양쪽",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, ONE + "what_magnitude_buys.png")
