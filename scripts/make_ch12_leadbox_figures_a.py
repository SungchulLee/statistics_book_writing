"""10장 들머리 그림 여섯 장 (일원배치 셋 · 이원배치 셋).

만드는 파일:
  ch10/anova_one_way/img/f_is_not_effect_size.png    F 는 효과크기가 아니다
  ch10/anova_one_way/img/computation_vs_design.png   셈의 정확함과 설계의 타당함
  ch10/anova_one_way/img/lambda_explains_it.png      비중심모수 하나가 설명한다
  ch10/anova_two_way/img/when_blocking_costs.png     블록이 손해가 되는 자리
  ch10/anova_two_way/img/spread_of_the_gaps.png      격차들의 흩어짐
  ch10/anova_two_way/img/gaps_with_error_bars.png    그림이 숨기는 것

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

ONE = "docs/ch10/anova_one_way/img/"
TWO = "docs/ch10/anova_two_way/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_f_is_not_effect_size():
    """12.1 일원배치 모형 — F 가 크다는 것은 효과가 크다는 뜻이 아니다.

    쪽의 manual_ss_decomposition 은 제곱합을 손으로 쪼개고,
    oneway_signal_noise 는 신호와 잡음을 그린다. 여기서는 들머리가 끝에
    한 줄로만 적어 둔 경고 --- **같은 효과에서도 표본이 커지면 F 가
    커진다** --- 를 그린다. 왼쪽은 쪽이 드는 두 비(2.79 와 4.85)가
    자유도로 나누는 순간 뒤집히는 모습이다.
    """
    SSB, SSW, DFB, DFW = 3.766, 10.492, 2, 27
    MSB, MSW = SSB / DFB, SSW / DFW

    K, NS = 3, np.array([5, 10, 20, 50, 100, 200, 500])
    MU = np.array([3.0, 5.0, 6.0])
    SIG = 10.0
    eta2_pop = ((MU - MU.mean()) ** 2).mean() / \
        (((MU - MU.mean()) ** 2).mean() + SIG ** 2)
    lam = NS * ((MU - MU.mean()) ** 2).sum() / SIG ** 2
    EF = 1 + lam / (K - 1)                 # 비중심 F 의 기댓값에 가까운 값

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(2)
    w = 0.3
    ax1.bar(xs - w / 2, [SSB, MSB], width=w * 0.92, color="tab:blue",
            alpha=0.9, label="집단 간")
    ax1.bar(xs + w / 2, [SSW, MSW], width=w * 0.92, color="tab:red",
            alpha=0.9, label="집단 내")
    for x_, (a, b) in zip(xs, ((SSB, SSW), (MSB, MSW))):
        ax1.text(x_ - w / 2, a + 0.2, f"{a:.4f}", ha="center", va="bottom",
                 fontsize=11, color="tab:blue", fontweight="bold")
        ax1.text(x_ + w / 2, b + 0.2, f"{b:.4f}", ha="center", va="bottom",
                 fontsize=11, color="tab:red", fontweight="bold")
    ax1.text(0, -1.45, f"집단 내가 {SSW / SSB:.2f} 배 크다\n"
             f"— 「집단 차이는 자잘하다」", ha="center", va="top",
             fontsize=11, color="0.2", fontweight="bold")
    ax1.text(1, -1.45, f"집단 간이 {MSB / MSW:.2f} 배 크다\n"
             f"— 그 수가 곧 $F$ 다", ha="center", va="top", fontsize=11,
             color="0.2", fontweight="bold")
    ax1.set_xticks(xs, [f"제곱합\n(조각 수를 안 센다)",
                        f"평균제곱\n(각자의 자유도 {DFB} 와 {DFW} 로 나눈다)"],
                   fontsize=11)
    ax1.set_ylim(-4.6, 12.5)
    ax1.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax1.set_ylabel("값", fontsize=11)
    ax1.set_title("나누는 순간 크기 관계가 뒤집힌다", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(NS, EF, "o-", ms=9, lw=2.8, color="tab:red",
             label="$F$ 의 기댓값")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(NS, [str(v) for v in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_yticks([1, 10, 100], ["1", "10", "100"])
    ax2.set_yticks([], minor=True)
    ax2.set_xlabel("집단당 표본 크기 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("$F$ 의 기댓값 (로그 눈금)", fontsize=11, color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax2.grid(alpha=0.25, which="major")
    ax3 = ax2.twinx()
    ax3.plot(NS, [100 * eta2_pop] * len(NS), "s--", ms=8, lw=2.6,
             color="tab:blue")
    ax3.set_ylim(0, 10)
    ax3.set_ylabel("모집단 효과크기 $\\eta^2$ (%)", fontsize=11,
                   color="tab:blue")
    ax3.tick_params(axis="y", colors="tab:blue")
    ax3.text(500, 100 * eta2_pop + 0.45,
             f"{100 * eta2_pop:.2f}% 에서 꿈쩍도 않는다", ha="right",
             va="bottom", fontsize=11, color="tab:blue", fontweight="bold")
    ax2.set_title("평균도 표준편차도 그대로 두고 n 만 키우면", fontsize=12.5)

    fig.suptitle("$F$ 가 크다는 것은 효과가 크다는 뜻이 아니다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, ONE + "f_is_not_effect_size.png")

    print(f"\n[일원배치] 제곱합 비 {SSW / SSB:.4f} · 평균제곱 비 "
          f"{MSB / MSW:.4f} (= F)")
    print(f"  두 비의 곱 {(SSW / SSB) * (MSB / MSW):.4f} = "
          f"자유도의 비 {DFW / DFB:.1f}")
    print(f"  모집단 eta^2 = {eta2_pop:.6f}")
    for n, f in zip(NS, EF):
        print(f"  집단당 {n:>4d}  F 의 기댓값 {f:9.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_computation_vs_design():
    """12.1 F 검정 — 여섯째 자리까지 맞는 계산이 보장하지 않는 것.

    쪽의 lanarkshire 상자는 고르게 나누려던 일이 어긋난 자리를 다룬다.
    여기서는 들머리의 마지막 문단 --- **아홉 단계 어디에도 이 세 집단이
    어떻게 갈렸는지 묻는 칸은 없다** --- 만 본다. 참 처치효과를 정확히
    0 으로 두고 집단이 갈린 방식만 바꾸면, 같은 아홉 단계가 한쪽에서는
    5% 를 지키고 다른 쪽에서는 지키지 않는다.
    """
    K, N, REPS = 3, 20, 4000
    GAMMAS = np.linspace(0, 1.2, 7)
    rng = np.random.default_rng(1210)

    def one(gamma, design, gen):
        u = gen.normal(0, 1, K * N)             # 재지 않은 특성
        g = np.repeat(np.arange(K), N)
        if design == "self":
            # 특성이 큰 사람일수록 뒤쪽 집단으로 간다
            order = np.argsort(u + gen.normal(0, 1 / max(gamma, 1e-9),
                                              K * N))
            g = np.empty(K * N, int)
            g[order] = np.repeat(np.arange(K), N)
        y = 50 + gamma * u + gen.normal(0, 1, K * N)   # 참 처치효과는 0
        gm = np.array([y[g == k].mean() for k in range(K)])
        ssb = N * ((gm - y.mean()) ** 2).sum()
        ssw = sum(((y[g == k] - gm[k]) ** 2).sum() for k in range(K))
        f = (ssb / (K - 1)) / (ssw / (K * N - K))
        return stats.f.sf(f, K - 1, K * N - K) < 0.05

    rate_self, rate_rand = [], []
    for gm in GAMMAS:
        rate_self.append(np.mean([one(gm, "self", rng)
                                  for _ in range(REPS)]))
        rate_rand.append(np.mean([one(gm, "rand", rng)
                                  for _ in range(REPS)]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    ax1.plot(GAMMAS, [100 * v for v in rate_self], "o-", ms=9, lw=2.8,
             color="tab:red", label="참가자가 스스로 골랐다")
    ax1.plot(GAMMAS, [100 * v for v in rate_rand], "D-", ms=8, lw=2.8,
             color="tab:blue", label="무작위로 배정했다")
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.text(1.19, 7, "약속한 $5\\%$", ha="right", va="bottom", fontsize=10.5,
             color="0.2", fontweight="bold")
    ax1.set_xlim(-0.03, 1.23)
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("재지 않은 특성이 결과에 들어가는 세기", fontsize=11)
    ax1.set_ylabel("$H_0$ 을 기각한 비율 (%)", fontsize=11)
    ax1.set_title(f"참 처치효과는 둘 다 정확히 $0$ 이다 "
                  f"(집단 {K} · 각 {N} 명 · {REPS:,} 번)", fontsize=12)
    ax1.legend(fontsize=10.5, loc="center right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    i = len(GAMMAS) - 1
    xs = np.arange(2)
    ax2.bar(xs, [100 * rate_self[i], 100 * rate_rand[i]], width=0.5,
            color=["tab:red", "tab:blue"], alpha=0.88)
    for x_, v in zip(xs, (rate_self[i], rate_rand[i])):
        ax2.text(x_, 100 * v + 1.5, f"{100 * v:.1f}%", ha="center",
                 va="bottom", fontsize=13, fontweight="bold", color="0.15")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_xticks(xs, ["스스로 골랐다", "무작위로 배정했다"], fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("기각률 (%)", fontsize=11)
    ax2.set_title(f"세기 {GAMMAS[i]:.1f} 에서", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.52, "아홉 단계는 두 자료를 구별하지 못한다\n"
             "— 같은 식에 같은 수를 넣을 뿐이다", transform=ax2.transAxes,
             ha="center", va="center", fontsize=11.5, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("셈의 정확함과 설계의 타당함은 다른 일이다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, ONE + "computation_vs_design.png")

    print(f"\n[F 검정] 참 처치효과 0 · 각 {REPS:,} 번")
    for g, a, b in zip(GAMMAS, rate_self, rate_rand):
        print(f"  세기 {g:.2f}  스스로 고름 {100*a:5.1f}%  "
              f"무작위 {100*b:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_lambda_explains_it():
    """12.1 F 모의실험 — 비중심모수 하나가 다 설명한다.

    쪽의 f_null_vs_alternative 는 귀무분포와 대립분포를 겹쳐 그린다.
    여기서는 들머리의 둘째·셋째 문단 --- **lambda = sum n_i (mu_i -
    mubar)^2 / sigma^2 하나가 두 기각률을 모두 설명한다** --- 만 본다.
    평균·표준편차·표본을 제각기 바꾼 여러 설정이 lambda 를 가로축에 놓으면
    모두 한 곡선 위에 모인다.
    """
    K = 3
    MU = np.array([3.0, 5.0, 6.0])
    SSQ = ((MU - MU.mean()) ** 2).sum()
    NS = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    SIG = 10.0
    lam_page = NS * SSQ / SIG ** 2

    def power(lam, n):
        crit = stats.f.isf(0.05, K - 1, K * n - K)
        return stats.ncf.sf(crit, K - 1, K * n - K, lam)

    pw_page = [power(l, n) for l, n in zip(lam_page, NS)]

    rng = np.random.default_rng(1211)
    MIX = []
    for _ in range(40):
        n = int(rng.integers(8, 400))
        sg = float(rng.uniform(3, 20))
        mu = rng.normal(0, rng.uniform(0.5, 6.0), K)
        lam = n * ((mu - mu.mean()) ** 2).sum() / sg ** 2
        MIX.append((lam, power(lam, n)))
    MIX = np.array(MIX)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(NS, [100 * v for v in pw_page], "o-", ms=9, lw=2.8,
             color="tab:red")
    for n, v in ((10, pw_page[0]), (5000, pw_page[-1])):
        i = int(np.where(NS == n)[0][0])
        ax1.plot([n], [100 * pw_page[i]], "o", ms=14, mfc="none",
                 mec="tab:red", mew=2.4)
        ax1.text(n, 100 * pw_page[i] + 4,
                 f"집단당 {n:,}\n$\\lambda = {lam_page[i]:.2f}$\n"
                 f"{100 * pw_page[i]:.1f}%", ha="center", va="bottom",
                 fontsize=10.5, color="tab:red", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xticks(NS, [f"{v:,}" for v in NS], rotation=45, fontsize=9.5)
    ax1.set_xticks([], minor=True)
    ax1.set_ylim(0, 125)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("집단당 표본 크기 (로그 눈금)", fontsize=11)
    ax1.set_ylabel("기각률 (%)", fontsize=11)
    ax1.set_title(f"평균 {tuple(MU.astype(int))} · 표준편차 {SIG:.0f} 를 "
                  f"못 박고 n 만 바꿨다", fontsize=12)
    ax1.grid(alpha=0.25)

    lam_grid = np.logspace(-1.3, 2.6, 300)
    ax2.plot(lam_grid, [100 * power(l, 100) for l in lam_grid], "-", lw=2.8,
             color="0.35", label="닫힌 꼴 (집단당 $100$)")
    ax2.plot(MIX[:, 0], 100 * MIX[:, 1], "o", ms=8, color="tab:blue",
             alpha=0.8, label="평균·표준편차·n 을 제멋대로 바꾼 설정 40 개")
    ax2.plot(lam_page, [100 * v for v in pw_page], "s", ms=9,
             color="tab:red", label="왼쪽 그림의 아홉 설정")
    ax2.set_xscale("log")
    ax2.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("비중심모수 λ (로그 눈금)", fontsize=11)
    ax2.set_ylabel("기각률 (%)", fontsize=11)
    ax2.set_title("λ 를 가로축에 놓으면 모두 한 곡선에 모인다", fontsize=12.5)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("평균 차이를 한 치도 바꾸지 않고 기각률을 바꿀 수 있다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, ONE + "lambda_explains_it.png")

    print(f"\n[F 모의실험] sum (mu-mubar)^2 = {SSQ:.4f} · sigma = {SIG}")
    for n, l, v in zip(NS, lam_page, pw_page):
        print(f"  집단당 {n:>5,}  lambda {l:9.4f}  기각률 {100*v:6.2f}%")


# ════════════════════════════════════════════════════════════════════
def fig_when_blocking_costs():
    """12.2 이원배치 — 블록이 손해가 되는 자리.

    쪽의 twoway_blocking 은 블록을 넣고 뺀 두 표를, blocking_origin
    상자는 그 생각이 나온 밭을 다룬다. 여기서는 들머리의 마지막 문단 ---
    **설명하지 못하는 변수를 넣으면 분모의 제곱합은 그대로인데 나눌
    자유도만 깎여 손해가 된다** --- 만 본다. 블록이 설명하는 몫을 0 에서
    키워 가며 두 모형의 검정력을 잰다.
    """
    A, B, REP = 3, 4, 2            # 요인 수준 · 블록 수준 · 칸당 반복
    N = A * B * REP
    SHARES = np.linspace(0.0, 0.8, 9)
    REPS = 3000
    EFFECT = np.array([-0.55, 0.0, 0.55])
    rng = np.random.default_rng(1212)

    def one(share, gen):
        blk = gen.normal(0, np.sqrt(share), B)
        a = np.repeat(np.arange(A), B * REP)
        b = np.tile(np.repeat(np.arange(B), REP), A)
        y = EFFECT[a] + blk[b] + gen.normal(0, np.sqrt(1 - share), N)

        def rss(cols):
            M = [np.ones(N)]
            if "a" in cols:
                M += [(a == k).astype(float) for k in range(1, A)]
            if "b" in cols:
                M += [(b == k).astype(float) for k in range(1, B)]
            M = np.column_stack(M)
            beta = np.linalg.lstsq(M, y, rcond=None)[0]
            return ((y - M @ beta) ** 2).sum(), M.shape[1]

        out = []
        for cols in ("a", "ab"):
            r1, p1 = rss(cols)
            r0, p0 = rss(cols.replace("a", "", 1) or "")
            f = ((r0 - r1) / (p1 - p0)) / (r1 / (N - p1))
            out.append(stats.f.sf(f, p1 - p0, N - p1) < 0.05)
        return out

    res = np.array([[one(s, rng) for _ in range(REPS)] for s in SHARES])
    pw_no = res[:, :, 0].mean(axis=1)
    pw_blk = res[:, :, 1].mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(100 * SHARES, 100 * pw_no, "o-", ms=9, lw=2.8, color="tab:red",
             label="블록을 적어 두지 않은 모형")
    ax1.plot(100 * SHARES, 100 * pw_blk, "D-", ms=8, lw=2.8,
             color="tab:blue", label="블록을 적어 둔 모형")
    ax1.set_xlabel("블록이 설명하는 변동의 몫 (%)", fontsize=11)
    ax1.set_ylabel("요인의 효과를 잡아낸 비율 (%)", fontsize=11)
    ax1.set_title(f"칸당 {REP} 개 · 요인 {A} · 블록 {B} · 각 {REPS:,} 번",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    diff = 100 * (pw_blk - pw_no)
    cols = ["tab:red" if d < 0 else "tab:blue" for d in diff]
    ax2.bar(100 * SHARES, diff, width=7, color=cols, alpha=0.88)
    ax2.axhline(0, color="0.15", lw=1.8)
    for s, d in zip(100 * SHARES, diff):
        ax2.text(s, d + np.sign(d) * 0.7, f"{d:+.1f}", ha="center",
                 va="bottom" if d > 0 else "top", fontsize=10,
                 color="0.2", fontweight="bold")
    ax2.set_xlabel("블록이 설명하는 변동의 몫 (%)", fontsize=11)
    ax2.set_ylabel("블록을 적어 두어 번 검정력 (퍼센트포인트)", fontsize=11)
    ax2.set_title("몫이 작으면 자유도만 깎여 손해다", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("블록을 적어 두는 일은 그 요인이 설명하는 만큼만 값을 한다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TWO + "when_blocking_costs.png")

    print(f"\n[이원배치] 각 {REPS:,} 번")
    for s, a, b in zip(SHARES, pw_no, pw_blk):
        print(f"  블록의 몫 {100*s:4.0f}%  안 넣음 {100*a:5.1f}%  "
              f"넣음 {100*b:5.1f}%  차이 {100*(b-a):+5.1f}")


# ════════════════════════════════════════════════════════════════════
def fig_spread_of_the_gaps():
    """12.2 교호작용 — 격차들이 서로 얼마나 흩어져 있는지.

    쪽의 interaction_learned 상자는 모형이 찾아내는 교호작용을 다루고
    본문 그림은 이 쪽의 자료를 그린다. 여기서는 들머리의 셋째 문단 ---
    **교호작용 제곱합은 격차들의 흩어짐 말고 아무것도 아니다** --- 를
    그린다. 평균 격차를 3.70 으로 못 박고 흩어짐만 바꾸면 제곱합이
    흩어짐의 제곱을 따라간다.
    """
    GAPS = np.array([5.25, 5.93, -0.08])
    MEAN_GAP = GAPS.mean()
    REP, LVL = 10, 2
    SS_PAGE = ((GAPS - MEAN_GAP) ** 2).sum() * REP / LVL

    ts = np.linspace(0, 1.6, 300)
    dev = GAPS - MEAN_GAP
    ss = np.array([((t * dev) ** 2).sum() * REP / LVL for t in ts])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(3)
    DOSES = ["낮은 용량", "가운데 용량", "높은 용량"]
    ax1.bar(xs, GAPS, width=0.5, color=["tab:blue", "tab:blue", "tab:red"],
            alpha=0.88)
    ax1.axhline(MEAN_GAP, color="0.2", ls="--", lw=2.2)
    ax1.text(2.42, MEAN_GAP + 0.25, f"평균 격차 {MEAN_GAP:.2f}", ha="right",
             va="bottom", fontsize=11, color="0.2", fontweight="bold")
    ax1.axhline(0, color="0.15", lw=1.6)
    for x_, g in zip(xs, GAPS):
        ax1.text(x_, g + np.sign(g) * 0.3, f"{g:+.2f}", ha="center",
                 va="bottom" if g > 0 else "top", fontsize=12,
                 fontweight="bold", color="0.15")
        ax1.annotate("", xy=(x_ + 0.3, g), xytext=(x_ + 0.3, MEAN_GAP),
                     arrowprops=dict(arrowstyle="<->", color="tab:orange",
                                     lw=2.0))
        ax1.text(x_ + 0.37, (g + MEAN_GAP) / 2, f"{g - MEAN_GAP:+.2f}",
                 ha="left", va="center", fontsize=10.5, color="tab:orange",
                 fontweight="bold")
    ax1.set_xticks(xs, DOSES, fontsize=11.5)
    ax1.set_xlim(-0.55, 2.95)
    ax1.set_ylim(-2.2, 8.2)
    ax1.set_ylabel("두 보충제의 격차", fontsize=11)
    ax1.set_title("주황 화살표를 제곱해 더하면 끝이다", fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.03, 0.04, f"$({dev[0]:.2f})^2 + ({dev[1]:.2f})^2 + "
             f"({dev[2]:.2f})^2 = {((GAPS - MEAN_GAP) ** 2).sum():.2f}$\n"
             f"$\\times\\ {REP}/{LVL} = {SS_PAGE:.1f}$ — 표의 그 수다",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=11,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(ts, ss, "-", lw=2.8, color="tab:purple")
    ax2.plot([1.0], [SS_PAGE], "o", ms=13, color="tab:purple", zorder=5)
    ax2.annotate(f"이 쪽의 자료\n{SS_PAGE:.1f}", xy=(1.0, SS_PAGE),
                 xytext=(0.52, 180), fontsize=11.5, color="tab:purple",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:purple",
                                 lw=1.8))
    ax2.plot([0], [0], "o", ms=13, mfc="none", mec="0.25", mew=2.4,
             zorder=5)
    ax2.text(0.05, 14, "격차가 고르면 정확히 $0$", ha="left", va="bottom",
             fontsize=11, color="0.25", fontweight="bold")
    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(0, 290)
    ax2.set_xlabel("격차가 평균에서 벗어난 정도 (이 쪽의 자료를 1 로)",
                   fontsize=11)
    ax2.set_ylabel("교호작용 제곱합", fontsize=11)
    ax2.set_title("흩어짐의 제곱만큼 커진다", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("교호작용 제곱합은 격차들이 서로 얼마나 흩어져 있는지다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TWO + "spread_of_the_gaps.png")

    print(f"\n[교호작용] 격차 {GAPS.tolist()} · 평균 {MEAN_GAP:.4f}")
    print(f"  벗어난 정도 {np.round(dev, 4).tolist()}")
    print(f"  제곱합 {((GAPS - MEAN_GAP) ** 2).sum():.4f} × {REP}/{LVL} = "
          f"{SS_PAGE:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_gaps_with_error_bars():
    """12.2 교호작용 그림 — 그림이 그리는 것과 숨기는 것.

    쪽의 interaction_plot_31 은 두 선을, asbestos_scale 상자는 다른
    이야기를 다룬다. 여기서는 들머리의 셋째·넷째 문단 --- **그 판정에는
    표준오차가 들어 있지 않다** --- 만 본다. 같은 격차 셋에 구간을 얹고,
    그 구간이 허락하는 모양을 함께 그린다.
    """
    GAPS = np.array([5.25, 5.93, -0.08])
    SE, DF = 1.6240, 54
    tcrit = stats.t.ppf(0.975, DF)
    half = tcrit * SE
    DOSES = ["낮은 용량", "가운데 용량", "높은 용량"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(3)
    cols = ["tab:blue" if g - half > 0 else "tab:red" for g in GAPS]
    ax1.errorbar(xs, GAPS, yerr=half, fmt="o", ms=13, lw=0, elinewidth=3.0,
                 capsize=10, capthick=3.0, ecolor="0.4", mfc="white",
                 mec="0.15", mew=2.4, zorder=5)
    for x_, g, c in zip(xs, GAPS, cols):
        ax1.plot([x_, x_], [g - half, g + half], "-", lw=3.0, color=c,
                 alpha=0.7)
        ax1.text(x_ + 0.13, g, f"{g:+.2f}", ha="left", va="center",
                 fontsize=12, color=c, fontweight="bold")
        ax1.text(x_, g - half - 0.4, f"$({g - half:+.2f},\\ "
                 f"{g + half:+.2f})$", ha="center", va="top", fontsize=10.5,
                 color="0.3")
    ax1.axhline(0, color="0.15", lw=2.0)
    ax1.set_xticks(xs, DOSES, fontsize=11.5)
    ax1.set_xlim(-0.5, 2.6)
    ax1.set_ylim(-5.5, 11)
    ax1.set_ylabel("두 보충제의 격차", fontsize=11)
    ax1.set_title(f"같은 격차 셋에 $95\\%$ 구간을 얹으면 "
                  f"(표준오차 {SE}, 자유도 {DF})", fontsize=12)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.97, 0.05, f"마지막 격차 ${GAPS[2]:+.2f}$ 의 구간이 폭 "
             f"${2 * half:.2f}$ 로 $0$ 을 넉넉히 품는다\n"
             f"— 어느 쪽이 위인지 이 자료는 알지 못한다",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=10.5,
             fontweight="bold", color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    base = np.array([20.0, 26.0, 26.0])
    rng = np.random.default_rng(1213)
    for _ in range(40):
        g = GAPS + rng.uniform(-half, half, 3)
        ax2.plot(xs, base, "-", lw=1.0, color="0.8", alpha=0.6)
        ax2.plot(xs, base + g, "-", lw=1.0, color="tab:orange", alpha=0.5)
    ax2.plot(xs, base, "o-", lw=3.0, ms=10, color="0.3", label="보충제 A")
    ax2.plot(xs, base + GAPS, "s-", lw=3.0, ms=10, color="tab:red",
             label="보충제 B")
    ax2.set_xticks(xs, DOSES, fontsize=11.5)
    ax2.set_xlim(-0.3, 2.3)
    ax2.set_xlabel("용량", fontsize=11)
    ax2.set_ylabel("평균 반응", fontsize=11)
    ax2.set_title("구간이 허락하는 모양을 마흔 번 겹쳐 그리면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.05, "평행한 모양도 교차하는 모양도 모두 들어 있다",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=11,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("그림은 격차의 크기만 보여 주고 그 격차가 얼마나 흔들리는지는 "
                 "숨긴다", fontsize=13.5, y=1.0)
    plt.tight_layout()
    save(fig, TWO + "gaps_with_error_bars.png")

    print(f"\n[교호작용 그림] 표준오차 {SE} · 자유도 {DF} · "
          f"임계값 {tcrit:.4f} · 반폭 {half:.4f}")
    for d, g in zip(DOSES, GAPS):
        print(f"  {d:<10s} {g:+.2f}  구간 ({g - half:+.4f}, "
              f"{g + half:+.4f})")


fig_f_is_not_effect_size()
fig_computation_vs_design()
fig_lambda_explains_it()
fig_when_blocking_costs()
fig_spread_of_the_gaps()
fig_gaps_with_error_bars()
