"""10장 들머리 그림 여섯 장 (진단 셋 · 응용 셋).

만드는 파일:
  ch10/diagnostics/img/one_point_does_not_wash_out.png  한 점은 씻겨 내려가지 않는다
  ch10/diagnostics/img/which_way_removal_goes.png       빼면 세지나 약해지나
  ch10/diagnostics/img/mean_stays_spread_grows.png      평균은 제자리, 흩어짐만
  ch10/practical_applications/img/stopped_early_inflates.png 일찍 멈추면 부푼다
  ch10/practical_applications/img/gatekeeper_blinks.png  문지기가 눈을 감는 자리
  ch10/practical_applications/img/cancellation_window.png 상쇄가 일어나는 자리

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

DIAG = "docs/ch10/diagnostics/img/"
APP = "docs/ch10/practical_applications/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_one_point_does_not_wash_out():
    """12.6 진단 — 한 점이 만든 왜도·첨도는 표본을 키워도 씻기지 않는다.

    쪽의 influence_fragile 은 한 점을 빼고 다시 학습해 보는 길을 다룬다.
    여기서는 들머리의 셋째 문단 --- **표본을 키워도 그 점이 남아 있으면
    두 수는 제자리로 오지 않는다** --- 만 본다. 꼬리가 두꺼운 자료와
    점 하나가 튄 자료를 같은 눈금에 올려 가른다.
    """
    NS = np.array([30, 60, 120, 250, 500, 1000])
    REPS = 600
    OUT = 7.0                      # 심는 점의 크기 (표준편차 단위)
    rng = np.random.default_rng(1260)

    def kurt(v):
        z = (v - v.mean()) / v.std(ddof=0)
        return (z ** 4).mean() - 3

    rows = {k: [] for k in ("깨끗한 자료", "점 하나가 튀었다",
                            "꼬리가 두껍다")}
    for n in NS:
        a = b = c = 0.0
        for _ in range(REPS):
            y = rng.normal(0, 1, n)
            a += kurt(y)
            z = y.copy()
            z[0] = OUT
            b += kurt(z)
            c += kurt(rng.standard_t(5, n))
        rows["깨끗한 자료"].append(a / REPS)
        rows["점 하나가 튀었다"].append(b / REPS)
        rows["꼬리가 두껍다"].append(c / REPS)

    sizes = np.linspace(2, 10, 200)
    n0 = 30
    share = []
    for s in sizes:
        y = np.concatenate([rng.normal(0, 1, n0 - 1), [s]])
        z = (y - y.mean()) / y.std(ddof=0)
        share.append(z[-1] ** 4 / (z ** 4).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    COLS = {"깨끗한 자료": "0.5", "점 하나가 튀었다": "tab:red",
            "꼬리가 두껍다": "tab:blue"}
    for nm, v in rows.items():
        ax1.plot(NS, v, "o-", ms=8, lw=2.6, color=COLS[nm], label=nm)
    ax1.axhline(0, color="0.2", ls="--", lw=1.6)
    ax1.set_xscale("log")
    ax1.set_xticks(NS, [str(v) for v in NS])
    ax1.set_xticks([], minor=True)
    ax1.set_xlabel("표본 크기 (로그 눈금)", fontsize=11)
    ax1.set_ylabel("초과첨도의 평균", fontsize=11)
    ax1.set_title(f"표본을 서른세 배로 키우면 ({REPS} 번 평균)",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.03, 0.06, "꼬리가 두꺼운 쪽은 n 과 무관하게 제자리이고\n"
             "점 하나가 튄 쪽만 n 을 따라 내려온다 — 그런데 느리다",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=10.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(sizes, 100 * np.array(share), "-", lw=2.8, color="tab:red")
    ax2.axhline(100 / n0, color="0.3", ls="--", lw=1.8)
    ax2.text(9.9, 100 / n0 + 1.5, f"관측 하나의 「공평한」 몫  "
             f"$1/{n0} = {100 / n0:.1f}\\%$", ha="right", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")
    ax2.set_xlim(2, 10)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel(f"튄 점의 크기 (표준편차 단위, $n = {n0}$)", fontsize=11)
    ax2.set_ylabel("그 한 점이 첨도에 내는 몫 (%)", fontsize=11)
    ax2.set_title("네제곱이라 멀리 있는 한 점이 전부를 가져간다",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("분포가 두꺼운 것인가, 점 하나가 튄 것인가", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "one_point_does_not_wash_out.png")

    print(f"\n[진단] 초과첨도의 평균 ({REPS} 번)")
    for i, n in enumerate(NS):
        print(f"  n = {n:>5d}  깨끗 {rows['깨끗한 자료'][i]:+7.3f}  "
              f"튄 점 {rows['점 하나가 튀었다'][i]:+7.3f}  "
              f"두꺼운 꼬리 {rows['꼬리가 두껍다'][i]:+7.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_which_way_removal_goes():
    """12.6 영향점 — 빼면 세지는 자리와 약해지는 자리.

    쪽의 influential_points_63·_97 은 영향 측도를, reinhart_rogoff 상자는
    한 나라의 한 해를 다룬다. 여기서는 들머리의 첫 문단 --- **재어 보면
    반대 방향인 경우가 흔하다** --- 를 이상점의 자리에 따라 그린다.
    이상점이 그 집단의 평균을 전체 평균 **쪽으로** 끌면 빼는 순간 F 가
    오르고, 바깥으로 끌면 내린다.
    """
    MU = np.array([10.0, 12.0, 14.0])
    N, SIG = 12, 1.2
    SHIFTS = np.linspace(-8, 8, 33)
    REPS = 1500
    rng = np.random.default_rng(1261)

    def f_of(groups):
        k = len(groups)
        n = np.array([len(g) for g in groups])
        m = np.array([g.mean() for g in groups])
        gm = np.concatenate(groups).mean()
        ssb = (n * (m - gm) ** 2).sum()
        ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
        return (ssb / (k - 1)) / (ssw / (n.sum() - k))

    ratio = {0: [], 2: []}
    for gi in (0, 2):
        for d in SHIFTS:
            acc = 0.0
            for _ in range(REPS):
                g = [rng.normal(m, SIG, N) for m in MU]
                g[gi] = g[gi].copy()
                g[gi][0] += d
                before = f_of(g)
                g2 = [x.copy() for x in g]
                g2[gi] = g2[gi][1:]
                acc += f_of(g2) / before
            ratio[gi].append(acc / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(SHIFTS, ratio[0], "o-", ms=7, lw=2.6, color="tab:blue",
             label="가장 낮은 집단의 한 점을 옮겼다")
    ax1.plot(SHIFTS, ratio[2], "s-", ms=7, lw=2.6, color="tab:red",
             label="가장 높은 집단의 한 점을 옮겼다")
    ax1.axhline(1, color="0.15", lw=2.0)
    ax1.axvline(0, color="0.7", lw=1.2)
    ax1.set_xlabel("그 한 점을 얼마나 옮겼나", fontsize=11)
    ax1.set_ylabel("그 점을 뺀 뒤의 $F$ ÷ 빼기 전의 $F$", fontsize=11)
    ax1.set_title(f"참 평균 {tuple(MU.astype(int))} · 집단당 {N} "
                  f"(각 {REPS:,} 번)", fontsize=12)
    ax1.legend(fontsize=10.5, loc="upper center", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.5, 0.05, "$1$ 보다 위면 빼는 순간 결론이 세진다",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    i_up = int(np.argmin(np.abs(SHIFTS - 6)))
    i_dn = int(np.argmin(np.abs(SHIFTS + 6)))
    xs = np.arange(2)
    vals = [ratio[2][i_up], ratio[2][i_dn]]
    cols = ["tab:red" if v < 1 else "tab:blue" for v in vals]
    ax2.bar(xs, vals, width=0.46, color=cols, alpha=0.9)
    ax2.axhline(1, color="0.15", lw=2.0)
    for x_, v in zip(xs, vals):
        ax2.text(x_, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="0.15")
    ax2.set_xticks(xs, ["위로 튀었다\n(집단 평균을 바깥으로 민다)",
                        "아래로 튀었다\n(집단 평균을 가운데로 끈다)"],
                   fontsize=11)
    ax2.set_ylim(0, max(vals) * 1.3)
    ax2.set_ylabel("뺀 뒤의 $F$ ÷ 빼기 전의 $F$", fontsize=11)
    ax2.set_title("가장 높은 집단에서 여섯만큼 옮겼을 때", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.06, "영향점을 찾았다는 사실만으로는\n"
             "결과를 의심할 이유가 되지 않는다", transform=ax2.transAxes,
             ha="center", va="bottom", fontsize=11.5, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("빼면 결론이 세지는 자리와 약해지는 자리가 따로 있다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "which_way_removal_goes.png")

    print(f"\n[영향점] 집단당 {N} · 각 {REPS:,} 번")
    for d, a, b in zip(SHIFTS[::4], ratio[0][::4], ratio[2][::4]):
        print(f"  옮긴 양 {d:+6.2f}  낮은 집단 {a:.4f}  높은 집단 {b:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_mean_stays_spread_grows():
    """12.6 위반 다루기 — 평균은 제자리이고 흩어짐만 커진다.

    쪽의 remedy_matrix 는 위반과 처방을 맞춰 두고, boxcox_rebutted 상자는
    이름 이야기를 따라간다. 여기서는 들머리의 둘째·셋째 문단 ---
    **한 점을 끌면 평균은 2.6% 만 움직이는데 표준편차는 1.81 배가 된다,
    그래서 효과를 놓치는 쪽으로 기운다** --- 만 본다.
    """
    N, SIG = 20, 1.145
    PULLS = np.linspace(0, 10, 200)
    rng = np.random.default_rng(1262)
    base = rng.normal(12.190922, SIG, N)
    base = (base - base.mean()) / base.std(ddof=1) * SIG + 12.190922

    mean_shift, sd_ratio = [], []
    for d in PULLS:
        y = base.copy()
        y[-1] += d
        mean_shift.append(100 * (y.mean() - base.mean()) / base.mean())
        sd_ratio.append(y.std(ddof=1) / base.std(ddof=1))

    MU = np.array([10.0, 11.5, 13.0])
    REPS = 3000
    pw = []
    for d in np.linspace(0, 10, 11):
        hit = 0
        for _ in range(REPS):
            g = [rng.normal(m, SIG, N) for m in MU]
            g[2] = g[2].copy()
            g[2][0] += d
            k, n = 3, 3 * N
            m = np.array([x.mean() for x in g])
            gm = np.concatenate(g).mean()
            ssb = N * ((m - gm) ** 2).sum()
            ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
            f = (ssb / (k - 1)) / (ssw / (n - k))
            hit += stats.f.sf(f, k - 1, n - k) < 0.05
        pw.append(hit / REPS)

    PULL_PAGE = 6.450755
    i_page = int(np.argmin(np.abs(PULLS - PULL_PAGE)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(PULLS, mean_shift, "-", lw=2.8, color="tab:blue",
             label="집단 평균이 움직인 비율 (%)")
    ax1.set_xlabel("그 한 점을 끌어당긴 양", fontsize=11)
    ax1.set_ylabel("평균이 움직인 비율 (%)", fontsize=11, color="tab:blue")
    ax1.tick_params(axis="y", colors="tab:blue")
    ax1.set_ylim(0, 5)
    ax1.grid(alpha=0.25)
    axb = ax1.twinx()
    axb.plot(PULLS, sd_ratio, "-", lw=2.8, color="tab:red")
    axb.set_ylabel("표준편차가 커진 배수", fontsize=11, color="tab:red")
    axb.tick_params(axis="y", colors="tab:red")
    axb.set_ylim(1, 3.2)
    axb.axvline(PULL_PAGE, color="0.4", ls=":", lw=1.8)
    axb.text(PULL_PAGE + 0.2, 1.08,
             f"이 쪽의 자리 {PULL_PAGE:.3f}\n"
             f"평균 {mean_shift[i_page]:.1f}% · 표준편차 "
             f"{sd_ratio[i_page]:.2f} 배", ha="left", va="bottom",
             fontsize=10.5, color="0.2", fontweight="bold")
    ax1.set_title(f"관측 {N} 개짜리 집단에서 한 점만 끌면", fontsize=12.5)

    ax2.plot(np.linspace(0, 10, 11), [100 * v for v in pw], "o-", ms=9,
             lw=2.8, color="tab:purple")
    ax2.set_xlabel("한 집단의 한 점을 끌어당긴 양", fontsize=11)
    ax2.set_ylabel("집단차를 잡아내는 비율 (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title(f"참 평균은 {tuple(MU)} 로 그대로다 (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.9, "평균은 거의 제자리인데 분모가 부풀어\n"
             "효과를 놓치는 쪽으로 기운다", transform=ax2.transAxes,
             ha="right", va="top", fontsize=11.5, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("제곱합에서는 그 양이 제곱으로 들어온다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "mean_stays_spread_grows.png")

    print(f"\n[위반 다루기] 끌어당긴 양 {PULL_PAGE:.6f} 에서")
    print(f"  평균 {mean_shift[i_page]:.3f}% 이동 · 표준편차 "
          f"{sd_ratio[i_page]:.4f} 배")
    for d, v in zip(np.linspace(0, 10, 11), pw):
        print(f"  끌어당긴 양 {d:5.1f}  검정력 {100 * v:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_stopped_early_inflates():
    """12.7 A/B 검정 — 일찍 멈춘 실험의 추정값은 부풀어 있다.

    쪽의 ab_peeking 은 엿보기가 기각률을 올리는 것을 그린다. 여기서는
    그 뒤에 남는 것 --- **엿보다 멈춘 실험이 보고하는 효과의 크기** ---
    만 본다. 참 효과가 0 이 아닐 때에도 일찍 멈춘 쪽은 제 크기보다 크게
    보고한다.
    """
    N_MAX, PEEKS, REPS = 4000, 50, 4000
    TRUE = 0.10
    rng = np.random.default_rng(1263)
    checks = np.linspace(N_MAX / PEEKS, N_MAX, PEEKS).astype(int)

    est_stop, est_full, stopped = [], [], 0
    for _ in range(REPS):
        a = rng.normal(0, 1, N_MAX)
        b = rng.normal(TRUE, 1, N_MAX)
        hit = None
        for m in checks:
            d = b[:m].mean() - a[:m].mean()
            se = np.sqrt(a[:m].var(ddof=1) / m + b[:m].var(ddof=1) / m)
            if abs(d / se) > 1.96:
                hit = d
                break
        est_full.append(b.mean() - a.mean())
        if hit is not None:
            est_stop.append(hit)
            stopped += 1
    est_stop = np.array(est_stop)
    est_full = np.array(est_full)

    PEEK_GRID = [1, 2, 5, 10, 25, 50]
    bias = []
    for P in PEEK_GRID:
        ck = np.linspace(N_MAX / P, N_MAX, P).astype(int)
        acc, cnt = 0.0, 0
        for _ in range(1500):
            a = rng.normal(0, 1, N_MAX)
            b = rng.normal(TRUE, 1, N_MAX)
            for m in ck:
                d = b[:m].mean() - a[:m].mean()
                se = np.sqrt(a[:m].var(ddof=1) / m + b[:m].var(ddof=1) / m)
                if abs(d / se) > 1.96:
                    acc += d
                    cnt += 1
                    break
        bias.append(acc / max(cnt, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    bins = np.linspace(-0.1, 0.45, 110)
    ax1.hist(est_full, bins=bins, color="0.6", alpha=0.75,
             label=f"끝까지 가서 한 번만 판정 (평균 {est_full.mean():.4f})")
    ax1.hist(est_stop, bins=bins, color="tab:red", alpha=0.6,
             label=f"엿보다 멈춘 실험 (평균 {est_stop.mean():.4f})")
    ax1.axvline(TRUE, color="0.15", lw=2.6)
    ax1.text(TRUE - 0.006, ax1.get_ylim()[1] * 0.95, f"참 효과 {TRUE}",
             ha="right", va="top", fontsize=11, color="0.15",
             fontweight="bold")
    ax1.set_xlabel("보고된 효과의 크기", fontsize=11)
    ax1.set_ylabel("횟수", fontsize=11)
    ax1.set_title(f"참 효과 {TRUE} · 최대 {N_MAX:,} 명 · {REPS:,} 번",
                  fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(PEEK_GRID, bias, "o-", ms=9, lw=2.8, color="tab:red")
    ax2.axhline(TRUE, color="0.15", ls="--", lw=2.0)
    ax2.text(50, TRUE + 0.004, f"참 효과 {TRUE}", ha="right", va="bottom",
             fontsize=10.5, color="0.15", fontweight="bold")
    for p, v in zip(PEEK_GRID, bias):
        ax2.text(p, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color="tab:red", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(PEEK_GRID, [str(v) for v in PEEK_GRID])
    ax2.set_xticks([], minor=True)
    ax2.set_xlabel("실험 동안 들여다본 횟수 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("멈춘 실험이 보고한 효과의 평균", fontsize=11)
    ax2.set_title("많이 볼수록 더 크게 보고하게 된다", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("일찍 멈춘 실험은 이겼다고도 하고 크게도 말한다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, APP + "stopped_early_inflates.png")

    print(f"\n[A/B] 참 효과 {TRUE} · 멈춘 비율 {stopped / REPS:.3f}")
    print(f"  끝까지 간 추정값의 평균 {est_full.mean():.4f}")
    print(f"  엿보다 멈춘 추정값의 평균 {est_stop.mean():.4f} "
          f"({est_stop.mean() / TRUE:.2f} 배)")
    for p, v in zip(PEEK_GRID, bias):
        print(f"  엿본 횟수 {p:>3d}  보고된 효과의 평균 {v:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_gatekeeper_blinks():
    """12.7 사례 — 가장 위험한 자리에서 문지기가 가장 자주 눈을 감는다.

    쪽의 네 그림은 사례 자체를, variance_components 는 분산성분을 다룬다.
    여기서는 들머리의 셋째 문단 --- **가장 위험한 자리에서 문지기가 가장
    자주 눈을 감는다** --- 만 본다. 등분산 위반의 세기를 키워 가며
    르빈의 적발률과 고전 F 의 실제 오류율을 같은 가로축에 올린다.
    """
    K, REPS = 3, 4000
    NS = np.array([30, 20, 10])            # 불균형 설계
    RATIOS = np.linspace(1.0, 4.0, 7)
    rng = np.random.default_rng(1264)

    def run(ratio, heavy):
        sds = np.array([1.0, np.sqrt(ratio), ratio])
        det = rej = 0
        for _ in range(REPS):
            if heavy:
                g = [s * rng.standard_t(4, n) / np.sqrt(2)
                     for s, n in zip(sds, NS)]
            else:
                g = [rng.normal(0, s, n) for s, n in zip(sds, NS)]
            det += stats.levene(*g, center="median").pvalue < 0.05
            m = np.array([x.mean() for x in g])
            gm = np.concatenate(g).mean()
            ssb = (NS * (m - gm) ** 2).sum()
            ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
            f = (ssb / (K - 1)) / (ssw / (NS.sum() - K))
            rej += stats.f.sf(f, K - 1, NS.sum() - K) < 0.05
        return det / REPS, rej / REPS

    norm = np.array([run(r, False) for r in RATIOS])
    hv = np.array([run(r, True) for r in RATIOS])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(RATIOS, 100 * norm[:, 1], "o-", ms=9, lw=2.8, color="tab:red",
             label="고전 $F$ 의 실제 오류율")
    ax1.plot(RATIOS, 100 * norm[:, 0], "s-", ms=8, lw=2.8, color="tab:blue",
             label="르빈이 적발하는 비율")
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.set_xlabel("가장 큰 집단과 가장 작은 집단의 표준편차 비", fontsize=11)
    ax1.set_ylabel("비율 (%)", fontsize=11)
    ax1.set_title(f"정규 자료 · 불균형 {tuple(NS)} (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(RATIOS, 100 * hv[:, 1], "o-", ms=9, lw=2.8, color="tab:red",
             label="고전 $F$ 의 실제 오류율")
    ax2.plot(RATIOS, 100 * hv[:, 0], "s-", ms=8, lw=2.8, color="tab:blue",
             label="르빈이 적발하는 비율")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_xlabel("가장 큰 집단과 가장 작은 집단의 표준편차 비", fontsize=11)
    ax2.set_ylabel("비율 (%)", fontsize=11)
    ax2.set_ylim(*ax1.get_ylim())
    ax2.set_title("거기에 꼬리까지 두꺼우면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.05, "오류율은 더 올라가는데\n적발률은 오히려 내려간다",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("「가정 위반의 증거가 없다」는 「없다」가 아니라 "
                 "「못 찾았다」이다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, APP + "gatekeeper_blinks.png")

    print(f"\n[사례] 불균형 {NS.tolist()} · 각 {REPS:,} 번")
    for r, a, b in zip(RATIOS, norm, hv):
        print(f"  표준편차 비 {r:.2f}  정규: 적발 {a[0]:.3f} / 오류율 "
              f"{a[1]:.3f}   두꺼운 꼬리: 적발 {b[0]:.3f} / 오류율 "
              f"{b[1]:.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_cancellation_window():
    """12.7 금융 응용 — 상쇄는 국면이 반반일 때 가장 완전하다.

    쪽의 finance_regime 은 국면별 성과를 그린다. 여기서는 들머리의 첫
    문단 --- **반대 방향의 두 효과가 상쇄되어 없는 것처럼 보인다** ---
    이 언제 일어나는지만 본다. 상승장의 비율을 바꾸어 가며 통째 평균의
    t 검정과 이원배치의 교호작용 검정을 나란히 잰다.
    """
    UP, DN = 2.53, -2.16                 # 모멘텀의 두 국면
    D_UP, D_DN = 0.38, -0.12             # 방어형
    SIG, MONTHS, REPS = 4.0, 120, 3000
    SHARES = np.linspace(0.2, 0.8, 13)
    rng = np.random.default_rng(1265)

    t_pow, inter_pow = [], []
    for s in SHARES:
        up = max(int(round(MONTHS * s)), 5)
        dn = MONTHS - up
        a = b = 0
        for _ in range(REPS):
            m_up = rng.normal(UP, SIG, up)
            m_dn = rng.normal(DN, SIG, dn)
            d_up = rng.normal(D_UP, SIG, up)
            d_dn = rng.normal(D_DN, SIG, dn)
            mom = np.concatenate([m_up, m_dn])
            dfn = np.concatenate([d_up, d_dn])
            a += stats.ttest_ind(mom, dfn, equal_var=False).pvalue < 0.05
            # 교호작용 --- 두 국면의 격차가 다른가
            g1 = m_up.mean() - d_up.mean()
            g2 = m_dn.mean() - d_dn.mean()
            se = SIG * np.sqrt(2 / up + 2 / dn)
            b += abs(g1 - g2) / se > 1.96
        t_pow.append(a / REPS)
        inter_pow.append(b / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    pooled = SHARES * UP + (1 - SHARES) * DN
    pooled_d = SHARES * D_UP + (1 - SHARES) * D_DN
    ax1.plot(100 * SHARES, pooled, "-", lw=2.8, color="tab:red",
             label="모멘텀의 통째 평균")
    ax1.plot(100 * SHARES, pooled_d, "-", lw=2.8, color="tab:blue",
             label="방어형의 통째 평균")
    ax1.fill_between(100 * SHARES, pooled, pooled_d, color="0.7", alpha=0.3)
    cross = 100 * (D_DN - DN) / ((UP - DN) - (D_UP - D_DN))
    ax1.axvline(cross, color="0.25", ls="--", lw=2.0)
    ax1.text(cross + 1, -1.6, f"상승장이 {cross:.1f}% 일 때\n"
             "두 평균이 정확히 겹친다", ha="left", va="bottom", fontsize=11,
             color="0.2", fontweight="bold")
    ax1.axhline(0, color="0.7", lw=1.2)
    ax1.set_xlabel("상승장이 차지하는 비율 (%)", fontsize=11)
    ax1.set_ylabel("한 달 평균 수익률 (%)", fontsize=11)
    ax1.set_title("국면을 지우고 통째로 평균하면", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(100 * SHARES, [100 * v for v in t_pow], "o-", ms=8, lw=2.8,
             color="tab:red", label="통째 평균의 $t$ 검정")
    ax2.plot(100 * SHARES, [100 * v for v in inter_pow], "s-", ms=8, lw=2.8,
             color="tab:blue", label="국면과의 교호작용 검정")
    ax2.axvline(cross, color="0.25", ls="--", lw=2.0)
    ax2.axhline(5, color="0.2", ls=":", lw=1.6)
    ax2.set_xlabel("상승장이 차지하는 비율 (%)", fontsize=11)
    ax2.set_ylabel("차이를 잡아내는 비율 (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title(f"{MONTHS} 개월 자료에서 (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="center left", framealpha=0.95)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.05, "교호작용 쪽은 국면 비율과 거의 무관하게\n"
             "자료에 있던 것을 찾아낸다", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             color="tab:blue",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:blue", lw=1.0))

    fig.suptitle("평균은 반대 부호의 두 수를 한 수로 접는다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, APP + "cancellation_window.png")

    print(f"\n[금융] 두 평균이 겹치는 상승장 비율 {cross:.2f}%")
    for s, a, b in zip(SHARES, t_pow, inter_pow):
        print(f"  상승장 {100*s:4.0f}%  t 검정 {100*a:5.1f}%  "
              f"교호작용 {100*b:5.1f}%")


# ════════════════════════════════════════════════════════════════════
fig_one_point_does_not_wash_out()
fig_which_way_removal_goes()
fig_mean_stays_spread_grows()
fig_stopped_early_inflates()
fig_gatekeeper_blinks()
fig_cancellation_window()
