"""15장 들어가는 이야기 그림 — 둘째 묶음 일곱.

  8.  f_test_two_variances  꼬리를 보고 고르면 오류율이 두 배가 된다
  9.  f_test/sensitivity    t 검정에는 없고 F 검정에만 있는 고장
  10. f_test_robust_alt     확인이 가장 필요한 곳에서 확인이 가장 무력하다
  11. bartlett_test         T 가 커지는 길이 둘이다
  12. bartlett_sensitivity  표본을 늘리면 더 나빠진다
  13. bartlett/limitations  첨도를 재어 보정하려 해도 못 한다
  14. robust_tests/levene   레빈이 견주는 것은 분산이 아니라 평균절대편차다

만드는 파일:
  docs/ch15/f_test/img/picking_the_tail_doubles_it.png
  docs/ch15/f_test/img/only_the_spread_question_breaks.png
  docs/ch15/f_test/img/weakest_where_most_needed.png
  docs/ch15/bartlett_test/img/two_roads_to_a_big_t.png
  docs/ch15/bartlett_test/img/more_data_makes_it_worse.png
  docs/ch15/bartlett_test/img/cannot_correct_what_you_cannot_measure.png
  docs/ch15/robust_tests/img/levene_compares_mad_not_variance.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

FT = "docs/ch15/f_test/img/"
BT = "docs/ch15/bartlett_test/img/"
RT = "docs/ch15/robust_tests/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# 모집단 묶음 — 분산을 1 로 맞추고 초과첨도만 바꾼다
def make_pops(r):
    return [
        ("균등", -1.2, lambda m, n: r.uniform(-np.sqrt(3), np.sqrt(3),
                                              (m, n))),
        ("정규", 0.0, lambda m, n: r.standard_normal((m, n))),
        ("t(10)", 1.0, lambda m, n: r.standard_t(10, (m, n))
         * np.sqrt(8 / 10)),
        ("t(6)", 3.0, lambda m, n: r.standard_t(6, (m, n))
         * np.sqrt(4 / 6)),
        ("지수", 6.0, lambda m, n: r.exponential(1.0, (m, n)) - 1.0),
        ("대수정규", 110.9, lambda m, n: (np.exp(r.normal(0, 1, (m, n)))
                                      / np.sqrt(np.exp(1)
                                                * (np.exp(1) - 1)))),
    ]


# ════════════════════════════════════════════════════════════════════
# 8. f_test_two_variances — 꼬리를 보고 고르면 오류율이 두 배
#
# 이 쪽은 그림이 열이다. f_rejection_region 이 「비대칭 기각역과 기각에
# 필요한 분산비」를, f_required_n 이 「검정력 0.8 에 필요한 표본크기」를
# 이미 그려 들머리의 첫·셋째 문단을 담는다. 남은 것은 **둘째 문단의
# 경고**다 — 「관측값을 보고 꼬리를 고른 뒤 단측으로 적으면 실제 오류율이
# 명목의 두 배가 된다.」 그 수가 어느 그림에도 없으므로 그것만 그린다.
# ════════════════════════════════════════════════════════════════════
N8 = [5, 8, 15, 30, 60, 120]
REPS8 = 60_000
r8 = np.random.default_rng(15_108)
proper, convention, cheat = [], [], []
for n in N8:
    a = r8.standard_normal((REPS8, n))
    b = r8.standard_normal((REPS8, n))
    s1, s2 = a.var(1, ddof=1), b.var(1, ddof=1)
    f = s1 / s2
    d = n - 1
    # ① 제대로 — 분자·분모를 미리 정하고 등꼬리 양측
    p_proper = 2 * np.minimum(stats.f.cdf(f, d, d), stats.f.sf(f, d, d))
    proper.append((p_proper < 0.05).mean())
    # ② 관례 — 큰 분산을 분자에 두고 한쪽 꼬리확률의 두 배
    fb = np.maximum(f, 1 / f)
    convention.append((2 * stats.f.sf(fb, d, d) < 0.05).mean())
    # ③ 꼬리를 보고 고른 뒤 단측으로 적기
    cheat.append((stats.f.sf(fb, d, d) < 0.05).mean())
print("[f_test_two_variances] 같은 자료를 세 가지로 적어 본다")
for n, a, b, c in zip(N8, proper, convention, cheat):
    print(f"  n = {n:4d}:  제대로 {a:.4f} · 관례(두 배) {b:.4f} · "
          f"꼬리를 보고 단측 {c:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
bx = np.arange(len(N8))
w = 0.27
ax.bar(bx - w, proper, width=w, color="tab:blue",
       label="① 분자를 미리 정하고 등꼬리 양측")
ax.bar(bx, convention, width=w, color="0.55",
       label="② 큰 분산을 분자에 두고 한쪽 꼬리의 두 배")
ax.bar(bx + w, cheat, width=w, color="tab:red",
       label="③ 큰 분산을 분자에 두고 그대로 단측")
for i in range(len(N8)):
    ax.text(bx[i] + w, cheat[i] + 0.0035, f"{cheat[i]:.3f}", ha="center",
            va="bottom", fontsize=10.5, fontweight="bold", color="tab:red")
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.text(len(N8) - 0.45, 0.053, "약속한 0.05", ha="right", va="bottom",
        fontsize=11.5, color="0.2", fontweight="bold")
ax.set_xticks(bx, [f"집단당 {n}" for n in N8], fontsize=11.5)
ax.set_xlabel("표본크기", fontsize=12)
ax.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=12)
ax.set_ylim(0, 0.135)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.annotate("관측값을 보고 꼬리를 고른 뒤\n단측으로 적으면 정확히 두 배다",
            xy=(bx[3] + w, cheat[3]), xytext=(bx[1] + 0.35, 0.118),
            fontsize=11.8, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.set_title("같은 자료, 같은 통계량 — 적는 법만 바꾼 것이다", fontsize=13)
fig.suptitle("큰 분산을 분자에 두는 관례에는 조건이 하나 붙어 있다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, FT + "picking_the_tail_doubles_it.png")


# ════════════════════════════════════════════════════════════════════
# 9. f_test/sensitivity — t 검정에는 없고 F 검정에만 있는 고장
#
# 쪽의 f_size_by_shape 는 「모집단 모양이 바뀌면 F 통계량의 **실제 분포**가
# 표집분포와 어긋난다」를 그린다. 들머리가 마지막 문단에서 던지는 물음은
# 그 다음이다 — 「왜 평균에 대한 t 검정에는 없는 문제가 여기에만 있는가.」
#
# 그 물음에 답하려면 **같은 모집단에 두 검정을 나란히** 걸어야 한다.
# 모양을 바꿔 가며 두 크기를 함께 그리면 한쪽은 선 위에 붙어 있고
# 다른 쪽만 크게 움직인다. 쪽의 그림에는 t 검정이 없다.
# ════════════════════════════════════════════════════════════════════
N9, REPS9 = 20, 60_000
r9 = np.random.default_rng(15_109)
POPS = make_pops(r9)
t_size, f_size, g2s = [], [], []
for nm, g2, f in POPS:
    a = f(REPS9, N9)
    b = f(REPS9, N9)
    # 두 평균에 대한 웰치 t 검정
    tt = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    t_size.append((tt < 0.05).mean())
    # 두 분산에 대한 F 검정
    fr = a.var(1, ddof=1) / b.var(1, ddof=1)
    d = N9 - 1
    pf = 2 * np.minimum(stats.f.cdf(fr, d, d), stats.f.sf(fr, d, d))
    f_size.append((pf < 0.05).mean())
    g2s.append(g2)
print(f"\n[f_test/sensitivity] 같은 모집단에 두 물음을 건다 (n = {N9})")
for (nm, g2, _), a, b in zip(POPS, t_size, f_size):
    print(f"  {nm:8s} 초과첨도 {g2:7.1f}:  두 평균 {a:.4f} · "
          f"두 분산 {b:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
xx = np.arange(len(POPS))
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.plot(xx, t_size, "o-", color="tab:blue", lw=3.2, ms=13,
        label="두 평균이 같은가 (웰치 t 검정)")
ax.plot(xx, f_size, "s-", color="tab:red", lw=3.2, ms=12,
        label="두 분산이 같은가 (F 검정)")
for i in range(len(POPS)):
    ax.text(xx[i], f_size[i] + 0.016, f"{f_size[i]:.3f}", ha="center",
            va="bottom", fontsize=11, fontweight="bold", color="tab:red")
ax.set_xticks(xx, [f"{nm}\n({g2:+.1f})" for nm, g2, _ in POPS],
              fontsize=11.5)
ax.set_xlabel("모집단 (괄호는 초과첨도) — 어느 쪽이든 두 표본은 같은 "
              "분포에서 왔다", fontsize=11.5)
ax.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=12)
ax.set_ylim(0, 0.52)
ax.legend(fontsize=11.8, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.text(0.03, 0.56, "약속한 0.05", transform=ax.transAxes, ha="left",
        va="bottom", fontsize=11.5, color="0.2", fontweight="bold")
ax.annotate("평균을 묻는 쪽은 모양이 바뀌어도\n선 위에 붙어 있다",
            xy=(xx[-2], t_size[-2] + 0.012), xytext=(1.6, 0.26),
            fontsize=12, color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.set_title(f"같은 자료, 같은 표본크기 (n = {N9}) — 갈라지는 것은 물음이다",
             fontsize=13)
fig.suptitle("왜 t 검정에는 없는 문제가 F 검정에만 있는가", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, FT + "only_the_spread_question_breaks.png")


# ════════════════════════════════════════════════════════════════════
# 10. f_test_robust_alt — 확인이 가장 필요한 곳에서 확인이 가장 무력하다
#
# 쪽의 shapiro_power 는 「표본크기에 따른 샤피로–윌크의 검정력」 하나를
# 그린다. 들머리의 머리글은 그 곡선 **하나로는 말할 수 없는 것**이다 —
# 「표본이 작으면 정규성 검정에 검정력이 거의 없고, 하필 그 자리가
# F 검정이 비정규성에 가장 취약한 자리다.」
#
# 두 곡선을 같은 가로축에 올려 보니 들머리가 적은 것과 다르다. F 검정의
# 왜곡은 작은 n 에서 가장 큰 것이 아니라 **n 과 함께 커진다**(0.114 에서
# 0.237). 그래서 그림이 말하는 것은 「작은 자리가 가장 나쁘다」가 아니라
# **「안전한 자리가 없다」**이다 — 작으면 못 보고, 커져서 보일 무렵에는
# 왜곡이 더 커져 있다. 잰 쪽으로 제목을 맞췄다.
# ════════════════════════════════════════════════════════════════════
NS10 = [8, 12, 20, 30, 50, 100, 200]
REPS10 = 20_000
r10 = np.random.default_rng(15_110)
sw_power, f_dist = [], []
for n in NS10:
    d = r10.standard_t(5, (REPS10, n)) * np.sqrt(3 / 5)
    pv = np.array([stats.shapiro(row).pvalue for row in d[:4000]])
    sw_power.append((pv < 0.05).mean())
    a = r10.standard_t(5, (REPS10, n)) * np.sqrt(3 / 5)
    b = r10.standard_t(5, (REPS10, n)) * np.sqrt(3 / 5)
    fr = a.var(1, ddof=1) / b.var(1, ddof=1)
    dd = n - 1
    pf = 2 * np.minimum(stats.f.cdf(fr, dd, dd), stats.f.sf(fr, dd, dd))
    f_dist.append((pf < 0.05).mean())
print("\n[f_test_robust_alt] 같은 t(5) 자료에서 둘을 함께 잰다")
for n, a, b in zip(NS10, sw_power, f_dist):
    print(f"  n = {n:4d}:  샤피로가 비정규를 잡아낼 확률 {a:.4f} · "
          f"F 검정의 실제 크기 {b:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(NS10, sw_power, "o-", color="tab:blue", lw=3.2, ms=12,
        label="정규성 검정이 이 자료를 잡아낼 확률")
ax.plot(NS10, f_dist, "s-", color="tab:red", lw=3.2, ms=11,
        label="같은 자료에서 F 검정의 실제 오류율 (약속은 0.05)")
ax.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax.set_xscale("log")
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에서 두부가 된다
ax.set_xticks(NS10, [str(n) for n in NS10])
ax.set_xlabel("집단당 표본크기 (로그 눈금)", fontsize=12)
ax.set_ylabel("비율", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11.5, loc="center left", bbox_to_anchor=(0.03, 0.72))
ax.grid(alpha=0.25)
ax.axvspan(8, 20, color="tab:red", alpha=0.07)
ax.text(11, 0.62, "확인이 가장 필요한 자리", ha="center", va="center",
        fontsize=12.5, color="tab:red", fontweight="bold", rotation=90)
ax.annotate(f"n = 8 에서 {sw_power[0]:.3f} 밖에 못 잡는데",
            xy=(8.6, sw_power[0]), xytext=(23, 0.17), fontsize=11.8,
            color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.annotate(f"그 자리에서 F 검정은 이미 {f_dist[0]:.3f} 이고,\n"
            f"n 을 키우면 {f_dist[-1]:.3f} 로 더 틀어진다",
            xy=(NS10[-2], f_dist[-2]), xytext=(23, 0.36), fontsize=11.8,
            color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.set_title("작으면 못 보고, 보일 무렵에는 더 틀어져 있다",
             fontsize=13)
fig.suptitle("확인이 가장 필요한 곳에서 확인이 가장 무력하다 — 그리고 "
             "안전한 자리가 없다", fontsize=15, y=0.975)
plt.tight_layout()
save(fig, FT + "weakest_where_most_needed.png")


# ════════════════════════════════════════════════════════════════════
# 11. bartlett_test — T 가 커지는 길이 둘이다
#
# 쪽의 bartlett_amgm 이 산술·기하평균의 간격을, correction_order 가
# 보정인자를, bartlett_equals_f 가 T 와 F 의 대응을 그린다. 들머리의
# **셋째 문단**은 그 셋이 건드리지 않는다 — 「T 가 커지는 길이 둘이다.
# 분산이 정말 다를 때 커지고, 자료가 정규가 아닐 때도 커진다. 기각을
# 보았을 때 둘 중 어느 쪽인지 통계량만으로는 구별할 수 없다.」
#
# 길이 둘이라는 말은 **평면**을 그려야 보인다. 가로를 분산의 이질성,
# 세로를 모집단의 비정규성으로 놓고 기각률을 칠하면, 같은 기각률에
# 이르는 길이 두 축 어느 쪽으로도 나 있는 것이 드러난다.
# ════════════════════════════════════════════════════════════════════
RATIOS = np.array([1.0, 1.3, 1.6, 2.0, 2.5, 3.0])
# t 자유도만으로는 세로 길이 0.22 까지밖에 못 가 「길이 둘」이 보이지
# 않는다. 들머리가 드는 대수정규(초과첨도 111)를 맨 위에 얹어야 두 길의
# 끝이 맞는다. 0 은 「정규」를 뜻한다.
DFS = [1e9, 20, 10, 7, 5, "lognorm"]
G2S = [0.0, 0.375, 1.0, 2.0, 6.0, 110.9]
NG, K11, REPS11 = 20, 3, 6000
r11 = np.random.default_rng(15_111)
grid = np.zeros((len(DFS), len(RATIOS)))
for i, df in enumerate(DFS):
    for j, ratio in enumerate(RATIOS):
        sds = np.array([1.0, np.sqrt(ratio), ratio])
        sds = sds / np.exp(np.log(sds).mean())
        rej = 0
        for _ in range(REPS11):
            groups = []
            for g in range(K11):
                if df == "lognorm":
                    e = np.exp(1.0)
                    x = (np.exp(r11.normal(0, 1, NG))
                         / np.sqrt(e * (e - 1)))
                elif df > 1e6:
                    x = r11.standard_normal(NG)
                else:
                    x = r11.standard_t(df, NG) * np.sqrt((df - 2) / df)
                groups.append(x * sds[g])
            rej += stats.bartlett(*groups).pvalue < 0.05
        grid[i, j] = rej / REPS11
print(f"\n[bartlett_test] 기각률 평면 (집단 {K11}, 집단당 {NG})")
print("   비정규성\\분산비  " + "  ".join(f"{r_:4.1f}" for r_ in RATIOS))
for i, g2 in enumerate(G2S):
    print(f"   초과첨도 {g2:5.2f}      " +
          "  ".join(f"{v:.2f}" for v in grid[i]))

fig, ax = plt.subplots(figsize=(10.4, 5.9))
im = ax.imshow(grid, origin="lower", cmap="Reds", vmin=0, vmax=1,
               aspect="auto")
for i in range(len(DFS)):
    for j in range(len(RATIOS)):
        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if grid[i, j] > 0.55 else "0.15")
ax.set_xticks(range(len(RATIOS)), [f"{r_:.1f}" for r_ in RATIOS],
              fontsize=12)
ax.set_yticks(range(len(DFS)),
              [f"{g2:.2f}" if g2 else "0 (정규)" for g2 in G2S],
              fontsize=12)
ax.set_xlabel("집단 사이 분산의 비 (1 이면 분산이 정말 같다)", fontsize=12)
ax.set_ylabel("모집단의 초과첨도", fontsize=12)
cb = fig.colorbar(im, ax=ax)
cb.set_label("기각하는 비율", fontsize=11.5)
ax.plot([0, 0], [0, len(DFS) - 1], color="tab:blue", lw=4.0, alpha=0.8)
ax.plot([0, len(RATIOS) - 1], [0, 0], color="tab:green", lw=4.0,
        alpha=0.8)
ax.text(0.12, 4.3, "이 길은 **분산이 같다**".replace("**", ""),
        ha="left", va="center", fontsize=12, color="tab:blue",
        fontweight="bold")
ax.text(2.6, 0.52, "이 길은 **자료가 정규다**".replace("**", ""),
        ha="center", va="center", fontsize=12, color="tab:green",
        fontweight="bold")
ax.set_title("같은 기각률에 이르는 길이 두 축 어느 쪽으로도 나 있다",
             fontsize=13)
fig.suptitle("T 가 커지는 길이 둘이고, 통계량만으로는 구별할 수 없다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, BT + "two_roads_to_a_big_t.png")


# ════════════════════════════════════════════════════════════════════
# 12. bartlett_sensitivity — 표본을 늘리면 더 나빠진다
#
# 쪽의 bartlett_lognormal_null 은 「실제 분포가 표집분포에서 통째로
# 벗어난다」를 그린다 — 한 자리에서의 어긋남이다. 들머리의 머리글은
# n 이 하는 일이다 — 「표본을 키우면 기각률이 0.6655 에서 0.9270 으로
# 오히려 올라간다. 자료를 더 모으는 것이 답이 되지 않는 드문 자리다.」
# 그 곡선이 그려져 있지 않다.
# ════════════════════════════════════════════════════════════════════
NS12 = [10, 20, 40, 80, 160, 320]
REPS12 = 4000
r12 = np.random.default_rng(15_112)
rates12 = {}
for nm, gen in [("대수정규", lambda n: np.exp(r12.normal(0, 1, n))),
                ("지수", lambda n: r12.exponential(1.0, n)),
                ("정규", lambda n: r12.standard_normal(n))]:
    row = []
    for n in NS12:
        rej = 0
        for _ in range(REPS12):
            g = [gen(n) for _ in range(3)]
            rej += stats.bartlett(*g).pvalue < 0.05
        row.append(rej / REPS12)
    rates12[nm] = row
print("\n[bartlett_sensitivity] 표본을 늘리면 어떻게 되는가")
for nm, row in rates12.items():
    print(f"  {nm:6s}: " + " ".join(f"{v:.4f}" for v in row))

fig, ax = plt.subplots(figsize=(10.8, 5.8))
for nm, c, mk in [("대수정규", "tab:red", "o-"), ("지수", "tab:orange", "s-"),
                  ("정규", "0.35", "^-")]:
    ax.plot(NS12, rates12[nm], mk, color=c, lw=3.0, ms=11, label=nm)
ax.axhline(0.05, color="tab:blue", lw=2.2, ls="--")
ax.text(330, 0.075, "약속한 5%", ha="right", va="bottom", fontsize=11.5,
        color="tab:blue", fontweight="bold")
ax.set_xscale("log")
ax.set_xticks(NS12, [str(n) for n in NS12])
ax.set_xlabel("집단당 표본크기 (로그 눈금) — 세 집단의 분산은 언제나 같다",
              fontsize=12)
ax.set_ylabel("「분산이 다르다」고 답하는 비율", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11.8, loc="center left", bbox_to_anchor=(0.03, 0.62))
ax.grid(alpha=0.25)
ax.annotate(f"{rates12['대수정규'][0]:.3f} 에서 "
            f"{rates12['대수정규'][-1]:.3f} 로 **올라간다**".replace("**", ""),
            xy=(NS12[-2], rates12["대수정규"][-2]), xytext=(13, 0.86),
            fontsize=12, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.text(0.97, 0.06, "문제가 근사의 수렴이 아니라 모형의 전제라서,\n"
        "표본을 늘리면 **틀린 극한**으로 더 정확하게 간다".replace("**", ""),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5,
        color="0.2", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))
ax.set_title("자료를 더 모으는 것이 답이 되지 않는 드문 자리", fontsize=13)
fig.suptitle("표본을 늘리면 더 나빠진다", fontsize=15, y=0.975)
plt.tight_layout()
save(fig, BT + "more_data_makes_it_worse.png")


# ════════════════════════════════════════════════════════════════════
# 13. bartlett_test/limitations — 재어 보정하려 해도 못 한다
#
# 쪽의 bartlett_size_vs_kurtosis 가 「초과첨도에 따른 실제 1 종 오류율」을
# 그려 들머리의 첫·둘째 문단을 담는다. 남은 것은 **셋째 문단**이고 그것이
# 가장 아픈 대목이다 — 「그 첨도가 4 차 적률이라 바로 그 표본에서 정밀하게
# 추정할 수가 없으니 어긋남의 크기를 짐작할 길이 없고, 보정해 줄 수도 없다.」
#
# 「못 한다」를 말로만 두지 않고 **해 본다.** 표본에서 첨도를 재어
# 통계량을 (2+g2hat)/2 로 나눠 보정하면 어떤 줄은 고쳐지고(균등
# 0.0026 -> 0.0526) 어떤 줄은 반쯤만 고쳐지는데(지수 0.3835 -> 0.2136),
# **정규 줄이 망가진다**(0.0509 -> 0.0895). 멀쩡하던 자리를 고장내는
# 보정이니 쓸 수 없다. 뿌리는 g2hat 의 폭이고, 그 폭이 고칠 대상만큼
# 크다는 것이 오른쪽 칸이다.
# ════════════════════════════════════════════════════════════════════
NG13, K13, REPS13 = 20, 3, 8000
r13 = np.random.default_rng(15_113)
POP13 = [("균등", -1.2, lambda n: r13.uniform(-np.sqrt(3), np.sqrt(3), n)),
         ("정규", 0.0, lambda n: r13.standard_normal(n)),
         ("t(10)", 1.0, lambda n: r13.standard_t(10, n) * np.sqrt(8 / 10)),
         ("t(6)", 3.0, lambda n: r13.standard_t(6, n) * np.sqrt(4 / 6)),
         ("지수", 6.0, lambda n: r13.exponential(1.0, n))]
raw13, adj13, g2hat_sd, g2_true = [], [], [], []
for nm, g2, gen in POP13:
    rej_raw = rej_adj = 0
    ghats = []
    for _ in range(REPS13):
        g = [gen(NG13) for _ in range(K13)]
        T = stats.bartlett(*g).statistic
        rej_raw += stats.chi2.sf(T, K13 - 1) < 0.05
        gh = np.mean([stats.kurtosis(x) for x in g])
        ghats.append(gh)
        scale = max((2 + gh) / 2, 0.15)
        rej_adj += stats.chi2.sf(T / scale, K13 - 1) < 0.05
    raw13.append(rej_raw / REPS13)
    adj13.append(rej_adj / REPS13)
    g2hat_sd.append(np.std(ghats))
    g2_true.append(g2)
print(f"\n[bartlett/limitations] 표본에서 첨도를 재어 보정해 본다 "
      f"(집단당 {NG13})")
for (nm, g2, _), a, b, s_ in zip(POP13, raw13, adj13, g2hat_sd):
    print(f"  {nm:6s} 참 초과첨도 {g2:+5.1f}:  보정 전 {a:.4f} → "
          f"보정 뒤 {b:.4f}   (재어 본 첨도의 표준편차 {s_:.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
xx = np.arange(len(POP13))
w = 0.36
ax1.bar(xx - w / 2, raw13, width=w, color="tab:red", label="보정하지 않고")
ax1.bar(xx + w / 2, adj13, width=w, color="tab:blue",
        label="표본에서 잰 첨도로 보정하고")
for i in range(len(POP13)):
    ax1.text(xx[i] + w / 2, adj13[i] + 0.008, f"{adj13[i]:.3f}",
             ha="center", va="bottom", fontsize=10.8, fontweight="bold",
             color="tab:blue")
ax1.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax1.set_xticks(xx, [f"{nm}\n({g2:+.1f})" for nm, g2, _ in POP13],
               fontsize=11)
ax1.set_xlabel("모집단 (괄호는 참 초과첨도)", fontsize=11.5)
ax1.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=11.5)
ax1.set_ylim(0, 0.46)
ax1.legend(fontsize=10.8, loc="upper left")
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("어떤 줄은 고쳐지고 **정규 줄은 망가진다**".replace("**", ""),
              fontsize=12.5)

ax2.bar(xx, g2hat_sd, color="0.5", width=0.55)
for i, (nm, g2, _) in enumerate(POP13):
    ax2.plot([i], [abs(g2)], "o", ms=13, color="tab:red", zorder=5)
    ax2.text(i, g2hat_sd[i] + 0.09, f"{g2hat_sd[i]:.2f}", ha="center",
             va="bottom", fontsize=11, fontweight="bold", color="0.25")
ax2.plot([], [], "o", ms=11, color="tab:red", label="참 초과첨도의 크기")
ax2.bar([], [], color="0.5", label="표본에서 잰 첨도가 흔들리는 폭")
ax2.set_xticks(xx, [nm for nm, _, _ in POP13], fontsize=11.5)
ax2.set_xlabel("모집단", fontsize=11.5)
ax2.set_ylabel("초과첨도", fontsize=11.5)
ax2.set_ylim(0, max(max(g2hat_sd), 6.5) * 1.22)
ax2.legend(fontsize=10.8, loc="upper left")
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("고치려고 쓰는 그 수가 고칠 대상만큼 흔들린다", fontsize=12.5)

fig.suptitle("어긋남의 크기를 짐작할 길이 없고, 보정해 줄 수도 없다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, BT + "cannot_correct_what_you_cannot_measure.png")


# ════════════════════════════════════════════════════════════════════
# 14. robust_tests/levene — 견주는 것이 분산이 아니다
#
# 쪽의 그림 셋이 변환·종속·검정력 손실을 모두 담는다. levene_dependence 의
# 오른쪽 칸은 크기 곡선까지 들고 있다. 남은 자리를 찾다 보니 들머리가
# **말하지 않고 지나간 것**이 하나 보인다 — 레빈이 실제로 견주는 것은
# 분산이 아니라 **평균절대편차**다. 정규에서 그 둘은 상수배로 묶여 있어
# 같은 말이 되는데, 모집단의 **모양이 서로 다르면** 상수가 달라져
# 분산이 같아도 평균절대편차가 다르다.
#
# 쪽의 어느 그림도 이것을 다루지 않고, 다음 쪽(중심을 바꾸는 쪽)으로도
# 고쳐지지 않는 종류의 한계라 적어 둘 값이 있다.
# ════════════════════════════════════════════════════════════════════
def mad_ratio(name):
    """분산 1 로 맞춘 분포의 E|X - mu|."""
    if name == "정규":
        return np.sqrt(2 / np.pi)
    if name == "균등":
        return np.sqrt(3) / 2
    if name == "라플라스":
        return 1 / np.sqrt(2)
    raise ValueError(name)


N14, REPS14 = 40, 20_000
r14 = np.random.default_rng(15_114)
GEN14 = {"정규": lambda m, n: r14.standard_normal((m, n)),
         "균등": lambda m, n: r14.uniform(-np.sqrt(3), np.sqrt(3), (m, n)),
         "라플라스": lambda m, n: r14.laplace(0, 1 / np.sqrt(2), (m, n))}
PAIRS = [("정규", "정규"), ("정규", "균등"), ("정규", "라플라스"),
         ("균등", "라플라스")]
lev_rate, f_rate, labels, gaps = [], [], [], []
for a_nm, b_nm in PAIRS:
    A = GEN14[a_nm](REPS14, N14)
    B = GEN14[b_nm](REPS14, N14)
    za = np.abs(A - A.mean(1, keepdims=True))
    zb = np.abs(B - B.mean(1, keepdims=True))
    tt = stats.ttest_ind(za, zb, axis=1, equal_var=False).pvalue
    lev_rate.append((tt < 0.05).mean())
    fr = A.var(1, ddof=1) / B.var(1, ddof=1)
    d = N14 - 1
    pf = 2 * np.minimum(stats.f.cdf(fr, d, d), stats.f.sf(fr, d, d))
    f_rate.append((pf < 0.05).mean())
    labels.append(f"{a_nm} 대 {b_nm}")
    gaps.append(mad_ratio(a_nm) - mad_ratio(b_nm))
print(f"\n[levene] 분산은 두 집단 모두 1 이다 (n = {N14})")
for lab, g, a, b in zip(labels, gaps, lev_rate, f_rate):
    print(f"  {lab:14s} 평균절대편차 차이 {g:+.4f}:  "
          f"레빈 {a:.4f} · F 검정 {b:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.15]})
names = ["균등", "정규", "라플라스"]
vals = [mad_ratio(n_) for n_ in names]
ax1.bar(np.arange(3), vals, color=["tab:green", "0.5", "tab:red"],
        width=0.55)
for i, v in enumerate(vals):
    ax1.text(i, v + 0.012, f"{v:.4f}", ha="center", va="bottom",
             fontsize=13, fontweight="bold")
ax1.set_xticks(np.arange(3), names, fontsize=12.5)
ax1.set_ylabel("평균절대편차 (분산은 셋 다 1 이다)", fontsize=11.5)
ax1.set_ylim(0, 1.02)
ax1.set_title("분산이 같아도 평균절대편차는 다르다", fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)
ax1.text(0.5, 0.18, "레빈이 실제로 견주는 것은\n이쪽이다",
         transform=ax1.transAxes, ha="center", va="center", fontsize=12.5,
         color="0.15", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.65"))

xx = np.arange(len(PAIRS))
w = 0.36
ax2.bar(xx - w / 2, lev_rate, width=w, color="tab:blue", label="레빈 검정")
ax2.bar(xx + w / 2, f_rate, width=w, color="0.55", label="F 검정")
for i in range(len(PAIRS)):
    ax2.text(xx[i] - w / 2, lev_rate[i] + 0.012, f"{lev_rate[i]:.3f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             color="tab:blue")
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xticks(xx, labels, fontsize=11)
ax2.set_xlabel("두 집단의 모집단 (분산은 언제나 둘 다 1)", fontsize=11.5)
ax2.set_ylabel("「분산이 다르다」고 답하는 비율", fontsize=11.5)
ax2.set_ylim(0, max(lev_rate) * 1.3)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("모양이 다르면 분산이 같아도 기각한다", fontsize=12.5)

fig.suptitle("문제를 갈아 끼우면 묻는 것도 갈린다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, RT + "levene_compares_mad_not_variance.png")
