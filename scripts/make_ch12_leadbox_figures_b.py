"""10장 들머리 그림 여섯 장 (사후비교 여섯 쪽).

만드는 파일:
  ch10/post_hoc/img/correction_or_sample.png     보정 탓인가 표본 탓인가
  ch10/post_hoc/img/one_pair_carries_it.png      한 쌍이 삼분의 이를 낸다
  ch10/post_hoc/img/family_you_actually_searched.png 실제로 뒤진 가족
  ch10/post_hoc/img/dunnett_uses_the_overlap.png 더닛이 더 쓰는 정보
  ch10/post_hoc/img/pooled_cuts_both_ways.png    합동분산은 양쪽으로 틀어진다
  ch10/post_hoc/img/averaged_effect_misleads.png 평균된 주효과가 하는 일

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/post_hoc/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def tukey_crit(k, df, alpha=0.05):
    return stats.studentized_range.ppf(1 - alpha, k, df) / np.sqrt(2)


# ════════════════════════════════════════════════════════════════════
def fig_correction_or_sample():
    """12.3 사후비교 — 놓친 쌍의 책임을 보정과 표본으로 가른다.

    쪽의 posthoc_thresholds 는 네 방법의 임계값을 집단 수에 따라 그리고,
    star_posthoc 상자는 다른 이야기를 따라간다. 여기서는 들머리의 셋째
    문단 --- **보정이 앗아간 몫은 0.128 이고 나머지 0.666 은 표본이
    작아서 없던 검정력** --- 을 그리고, 표본을 키우면 그 두 몫이 어떻게
    달라지는지 본다.
    """
    MU = np.array([10.0, 12.0, 15.0])
    SIG = 3.4          # 쪽이 적은 0.206 / 0.334 에 맞춘 값
    K = 3
    NS = np.array([15, 25, 40, 60, 100, 160])
    REPS = 6000
    rng = np.random.default_rng(1230)
    PAIR = (0, 1)                 # 참 차이가 2 인 가까운 쌍

    def rates(n, gen):
        df = K * (n - 1)
        q = tukey_crit(K, df)
        t_lsd = stats.t.ppf(0.975, df)
        hit_t = hit_l = 0
        for _ in range(REPS):
            y = gen.normal(MU[:, None], SIG, (K, n))
            m = y.mean(axis=1)
            msw = ((y - m[:, None]) ** 2).sum() / df
            se = np.sqrt(msw * 2 / n)
            t = abs(m[PAIR[0]] - m[PAIR[1]]) / se
            hit_t += t > q
            hit_l += t > t_lsd
        return hit_t / REPS, hit_l / REPS

    out = [rates(n, rng) for n in NS]
    pt = np.array([a for a, _ in out])
    pl = np.array([b for _, b in out])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    i0 = 0
    parts = [pt[i0], pl[i0] - pt[i0], 1 - pl[i0]]
    labs = [f"튜키가 잡는다\n{parts[0]:.3f}",
            f"보정이 앗아간 몫\n{parts[1]:.3f}",
            f"표본이 작아 없던 검정력\n{parts[2]:.3f}"]
    cols = ["tab:blue", "tab:orange", "0.62"]
    left = 0.0
    for v, lab, c in zip(parts, labs, cols):
        ax1.barh([0], [v], left=left, height=0.5, color=c, alpha=0.9)
        ax1.text(left + v / 2, 0, lab, ha="center", va="center",
                 fontsize=11.5, fontweight="bold",
                 color="white" if c != "0.62" else "0.15")
        left += v
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_yticks([])
    ax1.set_xlabel("가까운 한 쌍을 잡아낼 확률", fontsize=11)
    ax1.set_title(f"집단당 {NS[i0]} · 참 차이 ${MU[1] - MU[0]:.0f}$ · "
                  f"표준편차 ${SIG:.0f}$ ({REPS:,} 번)", fontsize=12.5)
    ax1.grid(axis="x", alpha=0.25)
    ax1.text(0.5, -0.55, "보정을 통째로 버려도 열 번에 "
             f"{10 * (1 - pl[i0]):.0f} 번은 여전히 놓친다", ha="center",
             va="center", fontsize=11.5, fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.stackplot(NS, 100 * pt, 100 * (pl - pt), 100 * (1 - pl),
                  colors=cols, alpha=0.9,
                  labels=["튜키가 잡는다", "보정이 앗아간 몫",
                          "표본이 작아 없던 검정력"])
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [str(v) for v in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_xlim(NS[0], NS[-1])
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("집단당 표본 크기 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("몫 (%)", fontsize=11)
    ax2.set_title("표본을 키우면 회색이 먼저 사라진다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="center left", framealpha=0.95)
    ax2.grid(alpha=0.2)

    fig.suptitle("놓친 쌍의 책임은 보정에 있는가 표본에 있는가", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, "correction_or_sample.png")

    print(f"\n[사후비교] 참 평균 {MU.tolist()} · 표준편차 {SIG}")
    for n, a, b in zip(NS, pt, pl):
        print(f"  집단당 {n:>4d}  튜키 {a:.3f}  LSD {b:.3f}  "
              f"보정의 몫 {b - a:.3f}  표본 탓 {1 - b:.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_one_pair_carries_it():
    """12.3 튜키 — 집단 간 제곱합의 삼분의 이가 한 쌍에서 온다.

    쪽의 tukey_studentized_range 는 스튜던트화 범위 분포를, tukey_vs_k 는
    집단 수에 따른 임계값을 그린다. 여기서는 들머리의 둘째·셋째 문단 ---
    **그 한 숫자의 삼분의 이가 한 쌍에서 온 것인데 어느 쌍인지는 표에
    적히지 않는다**, 그리고 **얽힌 비교의 실제 가족오류율** --- 만 본다.
    자료는 쪽과 같은 PlantGrowth 의 집단평균 셋이다.
    """
    NAMES = ["대조군", "처치 1", "처치 2"]
    MEANS = np.array([5.032, 4.661, 5.526])
    N = 10
    pairs = [(0, 1), (0, 2), (1, 2)]
    ss = np.array([N * (MEANS[i] - MEANS[j]) ** 2 / 2 for i, j in pairs])
    share = ss / ss.sum()
    ssb = N * ((MEANS - MEANS.mean()) ** 2).sum()

    KS = np.arange(2, 9)
    REPS = 20000
    rng = np.random.default_rng(1231)
    fwer = []
    for k in KS:
        n, df = 10, k * 9
        t_lsd = stats.t.ppf(0.975, df)
        hit = 0
        for _ in range(REPS):
            y = rng.normal(0, 1, (k, n))
            m = y.mean(axis=1)
            msw = ((y - m[:, None]) ** 2).sum() / df
            se = np.sqrt(msw * 2 / n)
            d = np.abs(m[:, None] - m[None, :])[np.triu_indices(k, 1)]
            hit += (d / se > t_lsd).any()
        fwer.append(hit / REPS)
    indep = 1 - 0.95 ** (KS * (KS - 1) / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(3)
    cols = ["0.65", "0.65", "tab:red"]
    ax1.bar(xs, 100 * share, width=0.52, color=cols, alpha=0.9)
    for x_, v, s in zip(xs, share, ss):
        ax1.text(x_, 100 * v + 1.5, f"{100 * v:.1f}%\n(제곱합 {s:.3f})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="0.15")
    ax1.set_xticks(xs, [f"{NAMES[i]} 대 {NAMES[j]}" for i, j in pairs],
                   fontsize=11.5)
    ax1.set_ylim(0, 85)
    ax1.set_ylabel("쌍별 제곱합이 차지하는 몫 (%)", fontsize=11)
    ax1.set_title(f"집단 간 제곱합 {ssb:.3f} 을 쌍별로 쪼개면",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.5, 0.95, "분산분석표에는 이 셋 가운데 어느 쌍인지\n"
             "적혀 있지 않다", transform=ax1.transAxes, ha="center",
             va="top", fontsize=11.5, fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(KS, 100 * np.array(fwer), "o-", ms=9, lw=2.8, color="tab:red",
             label="실제 (같은 평균을 돌려 쓴다)")
    ax2.plot(KS, 100 * indep, "s--", ms=8, lw=2.4, color="0.45",
             label="서로 독립이라면  $1 - 0.95^m$")
    ax2.axhline(5, color="0.2", ls=":", lw=1.8)
    ax2.text(8, 7, "약속한 $5\\%$", ha="right", va="bottom", fontsize=10.5,
             color="0.2", fontweight="bold")
    for k in (3, 4):
        i = int(np.where(KS == k)[0][0])
        ax2.text(k, 100 * fwer[i] - 4, f"{100 * fwer[i]:.1f}%", ha="center",
                 va="top", fontsize=11, color="tab:red", fontweight="bold")
    ax2.set_xticks(KS)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("집단 수 k", fontsize=11)
    ax2.set_ylabel("거짓 기각이 하나라도 나올 확률 (%)", fontsize=11)
    ax2.set_title(f"보정 없이 모든 쌍을 $5\\%$ 로 검정하면 "
                  f"(각 {REPS:,} 번)", fontsize=12)
    ax2.legend(fontsize=10.5, loc="lower right", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("표는 「적어도 한 집단이 다르다」까지만 말한다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, "one_pair_carries_it.png")

    print(f"\n[튜키] 집단 간 제곱합 {ssb:.4f} · 쌍별 합 {ss.sum():.4f}")
    for (i, j), s, v in zip(pairs, ss, share):
        print(f"  {NAMES[i]} 대 {NAMES[j]}  제곱합 {s:.4f}  몫 "
              f"{100 * v:.1f}%")
    for k, f, idp in zip(KS, fwer, indep):
        print(f"  k = {k}  쌍 {k*(k-1)//2:>2d}  실제 {100*f:5.1f}%  "
              f"독립 가정 {100*idp:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_family_you_actually_searched():
    """12.3 본페로니와 셰페 — 자료를 보고 고르면 가족이 커진다.

    쪽의 scheffe_and_bonferroni 는 두 임계값이 비교 수에 따라 갈리는 모습을,
    scheffe_max_contrast 는 최대 대비를 다룬다. 여기서는 들머리의 둘째·셋째
    문단 --- **자료를 보고 고른다는 것은 가능한 모든 대비를 뒤진 것과
    같다** --- 를 재어 본다. 모든 평균이 같은 자료에서 「가장 커 보이는
    대비」를 고른 뒤 본페로니로 셋을 보정하면 약속이 깨지고, 셰페는
    지킨다.
    """
    K, N, REPS = 4, 8, 20000
    df = K * (N - 1)
    rng = np.random.default_rng(1232)

    m_pre = 3                                # 미리 정해 둔 비교 수
    t_bonf = stats.t.ppf(1 - 0.05 / (2 * m_pre), df)
    f_scheffe = (K - 1) * stats.f.ppf(0.95, K - 1, df)

    hit_pre = hit_post_b = hit_post_s = 0
    maxF = np.empty(REPS)
    for r in range(REPS):
        y = rng.normal(0, 1, (K, N))          # 모든 평균이 같다
        m = y.mean(axis=1)
        msw = ((y - m[:, None]) ** 2).sum() / df
        # 미리 정해 둔 비교 셋 (1-2, 1-3, 1-4)
        se = np.sqrt(msw * 2 / N)
        t_pre = np.abs(m[0] - m[1:]) / se
        hit_pre += (t_pre > t_bonf).any()
        # 자료를 보고 고른 「가장 커 보이는」 대비 --- 최대 F_L
        dev = m - m.mean()
        FL = N * (dev ** 2).sum() / msw       # 최대 대비의 F_L
        maxF[r] = FL
        hit_post_b += FL > t_bonf ** 2
        hit_post_s += FL > f_scheffe

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.hist(maxF, bins=90, color="0.68", alpha=0.9)
    ax1.axvline(t_bonf ** 2, color="tab:red", lw=2.6)
    ax1.axvline(f_scheffe, color="tab:blue", lw=2.6)
    ax1.text(t_bonf ** 2 - 0.3, ax1.get_ylim()[1] * 0.95,
             f"본페로니(비교 {m_pre} 개)\n임계값 {t_bonf ** 2:.2f}",
             ha="right", va="top", fontsize=10.5, color="tab:red",
             fontweight="bold")
    ax1.text(f_scheffe + 0.3, ax1.get_ylim()[1] * 0.95,
             f"셰페 임계값 {f_scheffe:.2f}", ha="left", va="top",
             fontsize=10.5, color="tab:blue", fontweight="bold")
    ax1.set_xlim(0, 22)
    ax1.set_xlabel("자료를 보고 고른 「가장 커 보이는」 대비의 $F_L$",
                   fontsize=11)
    ax1.set_ylabel("횟수", fontsize=11)
    ax1.set_title(f"모든 평균이 같은 자료 (집단 {K} · 각 {N} 개 · "
                  f"{REPS:,} 번)", fontsize=12)
    ax1.grid(axis="y", alpha=0.25)

    xs = np.arange(3)
    vals = [hit_pre / REPS, hit_post_b / REPS, hit_post_s / REPS]
    cols = ["0.62", "tab:red", "tab:blue"]
    ax2.bar(xs, [100 * v for v in vals], width=0.52, color=cols, alpha=0.9)
    for x_, v in zip(xs, vals):
        ax2.text(x_, 100 * v / 2, f"{100 * v:.1f}%", ha="center",
                 va="center", fontsize=14, fontweight="bold", color="white")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_xticks(xs, [f"미리 정해 둔 {m_pre} 개\n+ 본페로니",
                        "자료를 보고 고른 뒤\n+ 본페로니",
                        "자료를 보고 고른 뒤\n+ 셰페"], fontsize=10.5)
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("거짓 기각이 하나라도 나올 확률 (%)", fontsize=11)
    ax2.set_title("약속한 $5\\%$ 를 지키는가", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("무엇을 비교할지 언제 정했는지가 지켜야 할 가족의 크기를 "
                 "바꾼다", fontsize=13.5, y=1.0)
    plt.tight_layout()
    save(fig, "family_you_actually_searched.png")

    print(f"\n[본페로니·셰페] 집단 {K} · 각 {N} · 자유도 {df}")
    print(f"  본페로니 임계값 t = {t_bonf:.4f} (F 로는 {t_bonf ** 2:.4f}) · "
          f"셰페 임계값 F_L = {f_scheffe:.4f}")
    print(f"  미리 정해 둔 {m_pre} 개 + 본페로니  {100 * vals[0]:.2f}%")
    print(f"  자료를 보고 고른 뒤 + 본페로니      {100 * vals[1]:.2f}%")
    print(f"  자료를 보고 고른 뒤 + 셰페          {100 * vals[2]:.2f}%")


# ════════════════════════════════════════════════════════════════════
def fig_dunnett_uses_the_overlap():
    """12.3 더닛 — 더닛이 더 쓰고 있는 정보는 겹침이다.

    쪽의 dunnett_control 과 dunnett_critical 은 대조군 비교와 임계값을
    그린다. 여기서는 들머리의 마지막 물음 --- **셋이 모두 5% 를 약속하는데
    임계값이 다르다면 더닛이 더 쓰고 있는 정보는 무엇인가** --- 에 답한다.
    대조군을 함께 쓰는 비교들은 상관이 0.5 로 양이고, 양의 상관이 있으면
    최댓값의 분포가 독립일 때보다 왼쪽에 놓인다.
    """
    N = 8
    KS = np.arange(2, 11)                    # 처치의 수
    REPS = 40000
    rng = np.random.default_rng(1233)

    def mv_t(k, rho, df, reps):
        """상관 rho 인 다변량 t 를 reps 개 뽑는다."""
        z = rng.normal(0, 1, (reps, k))
        c = rng.normal(0, 1, (reps, 1))
        v = np.sqrt(rho) * c + np.sqrt(1 - rho) * z
        u = np.sqrt(rng.chisquare(df, (reps, 1)) / df)
        return v / u

    dunnett, bonf, tukey = [], [], []
    for k in KS:
        df = (k + 1) * (N - 1)
        dunnett.append(np.quantile(np.abs(mv_t(k, 0.5, df, REPS)).max(axis=1),
                                   0.95))
        bonf.append(stats.t.ppf(1 - 0.05 / (2 * k), df))
        tukey.append(tukey_crit(k + 1, df))

    K_PAGE = 6
    i = int(np.where(KS == K_PAGE)[0][0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    df_page = (K_PAGE + 1) * (N - 1)
    mx_dep = np.abs(mv_t(K_PAGE, 0.5, df_page, REPS)).max(axis=1)
    mx_ind = np.abs(mv_t(K_PAGE, 0.0, df_page, REPS)).max(axis=1)
    bins = np.linspace(0, 5, 140)
    ax1.hist(mx_ind, bins=bins, density=True, histtype="step", lw=2.6,
             color="tab:red", label="서로 독립이라면")
    ax1.hist(mx_dep, bins=bins, density=True, histtype="step", lw=2.6,
             color="tab:blue",
             label="대조군을 함께 쓴다 (상관 $0.5$)")
    for v, col, nm in ((np.quantile(mx_ind, 0.95), "tab:red", "본페로니 쪽"),
                       (np.quantile(mx_dep, 0.95), "tab:blue", "더닛 쪽")):
        ax1.axvline(v, color=col, ls="--", lw=2.0)
        ax1.text(v, 1.02, f"{nm}\n{v:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color=col, fontweight="bold",
                 transform=ax1.get_xaxis_transform())
    ax1.set_xlim(0, 5)
    ax1.set_xlabel(f"처치 {K_PAGE} 개 비교의 최대 절댓값", fontsize=11)
    ax1.set_ylabel("밀도", fontsize=11)
    ax1.set_title("양의 상관이 있으면 최댓값이 덜 커진다", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(KS, tukey, "^-", ms=8, lw=2.6, color="0.4",
             label="튜키 — 묻지 않은 비교까지 보호")
    ax2.plot(KS, bonf, "s-", ms=8, lw=2.6, color="tab:red",
             label="본페로니 — 겹침을 쓰지 않는다")
    ax2.plot(KS, dunnett, "o-", ms=9, lw=2.8, color="tab:blue",
             label="더닛 — 겹침을 쓴다")
    ax2.axvline(K_PAGE, color="0.5", ls=":", lw=1.8)
    ax2.text(K_PAGE + 0.15, 1.95, f"이 쪽의 자리\n처치 {K_PAGE} 개",
             ha="left", va="bottom", fontsize=10.5, color="0.3",
             fontweight="bold")
    ax2.set_xticks(KS)
    ax2.set_xlabel("처치의 수 (대조군은 따로)", fontsize=11)
    ax2.set_ylabel("$5\\%$ 를 지키는 임계값", fontsize=11)
    ax2.set_title(f"집단당 {N} 개일 때 세 임계값", fontsize=12.5)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("더닛이 더 쓰는 정보는 비교들이 대조군을 함께 쓴다는 "
                 "사실이다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "dunnett_uses_the_overlap.png")

    print(f"\n[더닛] 집단당 {N} · {REPS:,} 번")
    for k, d, b, t in zip(KS, dunnett, bonf, tukey):
        print(f"  처치 {k:>2d}  더닛 {d:.4f}  본페로니 {b:.4f}  "
              f"튜키 {t:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_pooled_cuts_both_ways():
    """12.3 게임스–하웰 — 합동분산은 쌍마다 다른 방향으로 틀어진다.

    쪽의 games_howell_pairs 와 gh_small_n 은 쌍별 임계값과 작은 n 을
    다룬다. 여기서는 들머리의 마지막 물음 --- **어느 쌍이 과대평가되고
    어느 쌍이 과소평가되는가** --- 에 답한다. 합동분산은 모든 쌍에
    평균짜리 분산을 주므로, 분산이 큰 쌍에서는 분모를 작게 잡고 작은
    쌍에서는 크게 잡는다.
    """
    SD = np.array([np.sqrt(3.5), np.sqrt(6.0), np.sqrt(9.0), np.sqrt(15.0)])
    NS = np.array([14, 14, 14, 11])
    K = len(SD)
    df_pool = NS.sum() - K
    msw = ((NS - 1) * SD ** 2).sum() / df_pool
    q_t = tukey_crit(K, df_pool)

    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            se_pool = np.sqrt(msw * (1 / NS[i] + 1 / NS[j]))
            se_sep = np.sqrt(SD[i] ** 2 / NS[i] + SD[j] ** 2 / NS[j])
            num = (SD[i] ** 2 / NS[i] + SD[j] ** 2 / NS[j]) ** 2
            den = (SD[i] ** 2 / NS[i]) ** 2 / (NS[i] - 1) + \
                (SD[j] ** 2 / NS[j]) ** 2 / (NS[j] - 1)
            df_w = num / den
            q_gh = stats.studentized_range.ppf(0.95, K, df_w) / np.sqrt(2)
            rows.append((i, j, (SD[i] ** 2 + SD[j] ** 2) / 2 / msw,
                         q_t * se_pool, q_gh * se_sep))
    rows.sort(key=lambda r: r[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    xs = np.arange(len(rows))
    w = 0.3
    ax1.bar(xs - w / 2, [r[3] for r in rows], width=w * 0.92,
            color="tab:red", alpha=0.9, label="튜키 (합동분산 하나)")
    ax1.bar(xs + w / 2, [r[4] for r in rows], width=w * 0.92,
            color="tab:blue", alpha=0.9, label="게임스–하웰 (쌍별 분산)")
    ax1.set_xticks(xs, [f"{r[0]+1}–{r[1]+1}\n분산비 {r[2]:.2f}"
                        for r in rows], fontsize=10)
    ax1.set_xlabel("쌍 (그 두 집단의 평균 분산 ÷ 합동분산)", fontsize=11)
    ax1.set_ylabel("$5\\%$ 에서 필요한 차이", fontsize=11)
    ax1.set_title(f"집단 {K} · 분산이 $3.5$ 에서 $15.0$ 까지 흩어진 자료",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ratios = np.array([r[3] / r[4] for r in rows])
    vr = np.array([r[2] for r in rows])
    ax2.plot(vr, ratios, "o", ms=12, color="tab:purple", zorder=5)
    for v, rt, r in zip(vr, ratios, rows):
        ax2.text(v, rt + 0.018, f"{r[0]+1}–{r[1]+1}", ha="center",
                 va="bottom", fontsize=10.5, color="tab:purple",
                 fontweight="bold")
    ax2.axhline(1, color="0.2", ls="--", lw=2.0)
    ax2.axvline(1, color="0.6", ls=":", lw=1.8)
    ax2.set_xlabel("그 두 집단의 평균 분산 ÷ 합동분산", fontsize=11)
    ax2.set_ylabel("튜키 임계값 ÷ 게임스–하웰 임계값", fontsize=11)
    ax2.set_title("분산이 작은 쌍은 놓치고 큰 쌍은 쉽게 기각한다",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.5, 0.95, "1 보다 위면 튜키가 더 엄하고\n"
             "아래면 튜키가 더 느슨하다", transform=ax2.transAxes,
             ha="center", va="top", fontsize=11, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("합동분산 하나로 여섯 쌍을 모두 재면 쌍마다 다른 방향으로 "
                 "틀어진다", fontsize=13.5, y=1.0)
    plt.tight_layout()
    save(fig, "pooled_cuts_both_ways.png")

    print(f"\n[게임스–하웰] 합동분산 {msw:.4f} · 자유도 {df_pool}")
    for i, j, vr_, a, b in rows:
        print(f"  쌍 {i+1}-{j+1}  분산비 {vr_:.3f}  튜키 {a:.4f}  "
              f"게임스–하웰 {b:.4f}  비 {a / b:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_averaged_effect_misleads():
    """12.3 이원배치 사후비교 — 평균된 주효과를 먼저 보면 무엇이 어긋나나.

    쪽의 toothgrowth_interaction 은 두 선을 그린다. 여기서는 들머리가
    정한 **순서** --- 교호작용이 유의하면 주효과의 사후비교부터 보지
    않고 바로 단순효과로 내려간다 --- 가 왜 규칙이어야 하는지를 잰다.
    교호작용의 크기를 키워 가며 「주효과만 보고 모든 수준에 그 값을
    적용했을 때」 적어도 한 수준에서 부호나 유무를 틀리는 비율을 센다.
    """
    LV, REP, REPS = 3, 10, 4000
    SIG = 4.0
    BASE = np.array([5.25, 5.93, -0.08])
    MEAN_GAP = BASE.mean()
    SCALES = np.linspace(0, 1.6, 33)
    SE = np.sqrt(2 * SIG ** 2 / REP)

    # 주효과 하나로 말할 때 수준마다 몇 표준오차만큼 어긋나는가 (평균)
    frac = []
    for t in SCALES:
        gaps = MEAN_GAP + t * (BASE - MEAN_GAP)
        frac.append(np.mean(np.abs(gaps - MEAN_GAP)) / SE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    DOSES = ["낮은 용량", "가운데 용량", "높은 용량"]
    xs = np.arange(3)
    ax1.bar(xs, BASE, width=0.5, color=["tab:blue", "tab:blue", "tab:red"],
            alpha=0.88, label="그 용량에서 실제로 일어나는 격차")
    ax1.axhline(MEAN_GAP, color="0.15", ls="--", lw=2.4)
    ax1.text(2.42, MEAN_GAP + 0.3, f"평균된 주효과 {MEAN_GAP:.2f}",
             ha="right", va="bottom", fontsize=11.5, color="0.15",
             fontweight="bold")
    ax1.axhline(0, color="0.5", lw=1.4)
    for x_, g in zip(xs, BASE):
        ax1.annotate("", xy=(x_, g), xytext=(x_, MEAN_GAP),
                     arrowprops=dict(arrowstyle="->", lw=2.4,
                                     color="tab:orange"))
        ax1.text(x_ + 0.2, (g + MEAN_GAP) / 2,
                 f"{'작게 말한다' if g > MEAN_GAP else '크게 말한다'}\n"
                 f"{g - MEAN_GAP:+.2f}", ha="left", va="center",
                 fontsize=10.5, color="tab:orange", fontweight="bold")
    ax1.set_xticks(xs, DOSES, fontsize=11.5)
    ax1.set_xlim(-0.5, 2.95)
    ax1.set_ylim(-2.5, 8.5)
    ax1.set_ylabel("두 보충제의 격차", fontsize=11)
    ax1.set_title("$3.70$ 은 세 용량 어디에서도 일어나지 않는다",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(SCALES, frac, "-", lw=2.8, color="tab:red")
    i1 = int(np.argmin(np.abs(SCALES - 1.0)))
    ax2.plot([1.0], [frac[i1]], "o", ms=13, color="tab:red", zorder=5)
    ax2.annotate(f"이 쪽의 자료\n평균 {frac[i1]:.2f} 표준오차",
                 xy=(1.0, frac[i1]), xytext=(0.3, 1.9), fontsize=11.5,
                 color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
    ax2.axhline(1, color="0.3", ls="--", lw=1.8)
    ax2.text(1.62, 1.06, "표준오차 한 칸", ha="right", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")
    ax2.set_xlim(-0.05, 1.65)
    ax2.set_ylim(0, 2.9)
    ax2.set_xlabel("교호작용의 크기 (이 쪽의 자료를 1 로)", fontsize=11)
    ax2.set_ylabel("주효과로 말할 때 수준마다 어긋나는 양\n(표준오차 단위)",
                   fontsize=11)
    ax2.set_title("교호작용에 비례해 커진다", fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.05, "교호작용이 $0$ 이면 세 격차가 같아\n"
             "평균이 곧 세 값이 된다", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("교호작용이 유의하면 주효과의 사후비교부터 보지 않는다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "averaged_effect_misleads.png")

    print(f"\n[이원배치 사후비교] 격차 {BASE.tolist()} · 평균 "
          f"{MEAN_GAP:.4f}")
    for t, v in zip(SCALES[::4], frac[::4]):
        print(f"  교호작용 {t:.2f} 배  평균 어긋남 {v:.3f} 표준오차")


fig_correction_or_sample()
fig_one_pair_carries_it()
fig_family_you_actually_searched()
fig_dunnett_uses_the_overlap()
fig_pooled_cuts_both_ways()
fig_averaged_effect_misleads()
