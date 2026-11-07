"""15장 들어가는 이야기 그림 — 셋째 묶음 여섯 (마지막).

  15. levene_simulation         「흔들리지 않는다」와 「명목값과 같다」는 다르다
  16. brown_forsythe            중심을 바꾸면 신호가 절반 — 그 절반의 값
  17. fligner_killeen           꼬리는 막고 왜도는 못 막는다
  18. robust_tests              이상점 하나를 키워 가면 네 p 값이 갈라진다
  19. comparison                크기를 먼저 통과시킨 뒤에 검정력을 따진다
  20. robust_tests_comparison   왜도를 키우면 무너지는 순서가 있다

만드는 파일:
  docs/ch15/robust_tests/img/stable_is_not_calibrated.png
  docs/ch15/robust_tests/img/half_the_signal_costs_this.png
  docs/ch15/robust_tests/img/stops_tails_not_skew.png
  docs/ch15/robust_tests/img/one_outlier_pulls_them_apart.png
  docs/ch15/robust_tests/img/size_first_then_power.png
  docs/ch15/robust_tests/img/the_order_they_break.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

RT = "docs/ch15/robust_tests/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def bf(*g):
    """브라운–포사이드 — 중앙값 중심 레빈."""
    return stats.levene(*g, center="median")


def lev(*g):
    """평균 중심 레빈."""
    return stats.levene(*g, center="mean")


def run(gen, k, n, reps, rng, ratios=None, tests=None):
    """등분산(또는 주어진 비)에서 검정마다 기각 비율을 센다."""
    ratios = ratios if ratios is not None else np.ones(k)
    sds = np.sqrt(ratios)
    out = {nm: 0 for nm in tests}
    for _ in range(reps):
        g = [gen(n) * sds[i] for i in range(k)]
        for nm, fn in tests.items():
            try:
                out[nm] += fn(*g).pvalue < 0.05
            except Exception:
                pass
    return {nm: v / reps for nm, v in out.items()}


TESTS = {"바틀렛": stats.bartlett, "레빈 (평균 중심)": lev,
         "브라운–포사이드": bf, "플리그너–킬린": stats.fligner}

# ════════════════════════════════════════════════════════════════════
# 15. levene_simulation — 흔들리지 않는 것과 눈금이 맞는 것은 다르다
#
# 쪽의 size_vs_n_divergence 가 「n 이 커질 때 두 검정이 반대 방향으로」를,
# levene_centering 이 세 중심화를 그린다. 들머리의 **셋째 문단**은 그
# 둘이 건드리지 않는다 — 「로버스트한 쪽의 0.037 은 명목값 0.05 의
# 74% 이기도 하다. 「모집단에 흔들리지 않는다」와 「명목값과 같다」는
# 다른 주장이고, 뒤엣것은 성립하지 않는다.」
#
# 그래서 왼쪽에 **흔들리지 않음**(다섯 모집단에서 평평한 선)을, 오른쪽에
# 그 보수성이 치르는 값(검정력)을 둔다. 안전해 보이는 것이 공짜가 아니다.
# ════════════════════════════════════════════════════════════════════
r15 = np.random.default_rng(15_215)
POP15 = [
    ("균등", lambda n: r15.uniform(-np.sqrt(3), np.sqrt(3), n)),
    ("정규", lambda n: r15.standard_normal(n)),
    ("t(5)", lambda n: r15.standard_t(5, n) * np.sqrt(3 / 5)),
    ("지수", lambda n: r15.exponential(1.0, n)),
    ("대수정규", lambda n: np.exp(r15.normal(0, 1, n))),
]
K15, N15, REPS15 = 3, 30, 5000
size_bf, size_bart = [], []
for nm, gen in POP15:
    res = run(gen, K15, N15, REPS15, r15,
              tests={"bf": bf, "bart": stats.bartlett})
    size_bf.append(res["bf"])
    size_bart.append(res["bart"])
print("[levene_simulation] 다섯 모집단에서의 크기")
for (nm, _), a, b in zip(POP15, size_bf, size_bart):
    print(f"  {nm:8s} 브라운–포사이드 {a:.4f} · 바틀렛 {b:.4f}")
print(f"  폭: {min(size_bf):.3f}~{max(size_bf):.3f} 대 "
      f"{min(size_bart):.3f}~{max(size_bart):.3f}")

RATIOS15 = [1.0, 1.5, 2.0, 3.0, 4.0]
pw_bf, pw_bart = [], []
for ratio in RATIOS15:
    res = run(lambda n: r15.standard_normal(n), K15, N15, 4000, r15,
              ratios=np.array([1.0, 1.0, ratio]),
              tests={"bf": bf, "bart": stats.bartlett})
    pw_bf.append(res["bf"])
    pw_bart.append(res["bart"])
print("  정규 자료에서의 검정력")
for ratio, a, b in zip(RATIOS15, pw_bf, pw_bart):
    print(f"    분산비 {ratio}:  브라운–포사이드 {a:.4f} · 바틀렛 {b:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
xx = np.arange(len(POP15))
ax1.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax1.plot(xx, size_bart, "s-", color="tab:red", lw=3.0, ms=12,
         label="바틀렛")
ax1.plot(xx, size_bf, "o-", color="tab:blue", lw=3.0, ms=12,
         label="브라운–포사이드 (중앙값 중심)")
for i in xx:
    ax1.text(i, size_bf[i] - 0.035, f"{size_bf[i]:.3f}", ha="center",
             va="top", fontsize=10.8, fontweight="bold", color="tab:blue")
ax1.set_xticks(xx, [nm for nm, _ in POP15], fontsize=11.5)
ax1.set_xlabel("모집단 (분산은 어느 자리에서나 세 집단이 같다)",
               fontsize=11.5)
ax1.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=11.5)
ax1.set_ylim(0, 0.82)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_title("흔들리지 않는다 — 그런데 늘 명목선 **아래**다".replace("**", ""),
              fontsize=12.5)
ax1.annotate("0.05 가 아니라 그 아래에 평평하다",
             xy=(2, size_bf[2]), xytext=(1.1, 0.30), fontsize=11.3,
             color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

ax2.plot(RATIOS15, pw_bart, "s-", color="tab:red", lw=3.0, ms=12,
         label="바틀렛")
ax2.plot(RATIOS15, pw_bf, "o-", color="tab:blue", lw=3.0, ms=12,
         label="브라운–포사이드")
for i, ratio in enumerate(RATIOS15):
    if ratio in (2.0, 3.0):
        ax2.annotate("", xy=(ratio, pw_bf[i]), xytext=(ratio, pw_bart[i]),
                     arrowprops=dict(arrowstyle="<->", color="0.3", lw=2.0))
        ax2.text(ratio + 0.07, (pw_bf[i] + pw_bart[i]) / 2,
                 f"{pw_bart[i] - pw_bf[i]:.2f} 만큼\n덜 잡는다", ha="left",
                 va="center", fontsize=10.8, color="0.25",
                 fontweight="bold")
ax2.set_xlabel("세 집단 가운데 하나의 분산비 (정규 자료)", fontsize=11.5)
ax2.set_ylabel("참으로 다를 때 잡아내는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(alpha=0.25)
ax2.set_title("명목선 아래에 있는 만큼을 검정력으로 치른다", fontsize=12.5)

fig.suptitle("「흔들리지 않는다」와 「명목값과 같다」는 다른 주장이다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, RT + "stable_is_not_calibrated.png")


# ════════════════════════════════════════════════════════════════════
# 16. brown_forsythe — 중심을 바꾸면 신호가 절반, 그 절반의 값
#
# 쪽의 trimmed_size_disaster 가 세 중심화의 크기를, median_centering_
# mechanism 이 이상점 하나의 영향을 그린다. 들머리의 **둘째 문단**은
# 수로만 적혀 있다 — 「잃는 것이 그 절반이고, 그것은 정규 자료에서
# 검정력 10–25% 로 나타난다. 적지 않은 대가다.」 그 대가를 그린다.
# ════════════════════════════════════════════════════════════════════
r16 = np.random.default_rng(15_216)
NS16 = [10, 20, 30, 50]
RATIO16 = 3.0
pw_mean, pw_med = [], []
for n in NS16:
    res = run(lambda m: r16.standard_normal(m), 3, n, 5000, r16,
              ratios=np.array([1.0, 1.0, RATIO16]),
              tests={"mean": lev, "med": bf})
    pw_mean.append(res["mean"])
    pw_med.append(res["med"])
print(f"\n[brown_forsythe] 정규 자료에서 분산비 {RATIO16} 을 잡는 검정력")
for n, a, b in zip(NS16, pw_mean, pw_med):
    print(f"  n = {n:3d}:  평균 중심 {a:.4f} · 중앙값 중심 {b:.4f}  "
          f"(잃는 몫 {100 * (1 - b / a):.1f}%)")

r16b = np.random.default_rng(15_217)
size_mean, size_med = [], []
for n in NS16:
    res = run(lambda m: r16b.standard_normal(m), 3, n, 5000, r16b,
              tests={"mean": lev, "med": bf})
    size_mean.append(res["mean"])
    size_med.append(res["med"])
print("  정규 자료에서의 크기")
for n, a, b in zip(NS16, size_mean, size_med):
    print(f"  n = {n:3d}:  평균 중심 {a:.4f} · 중앙값 중심 {b:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
xx = np.arange(len(NS16))
w = 0.36
ax1.bar(xx - w / 2, pw_mean, width=w, color="0.55", label="평균 중심")
ax1.bar(xx + w / 2, pw_med, width=w, color="tab:blue",
        label="중앙값 중심 (브라운–포사이드)")
for i in xx:
    ax1.text(xx[i], max(pw_mean[i], pw_med[i]) + 0.022,
             f"{100 * (1 - pw_med[i] / pw_mean[i]):.0f}% 손실", ha="center",
             va="bottom", fontsize=11, fontweight="bold", color="0.2")
ax1.set_xticks(xx, [f"집단당 {n}" for n in NS16], fontsize=11.5)
ax1.set_xlabel(f"표본크기 (정규 자료, 분산비 {RATIO16:.0f})", fontsize=11.5)
ax1.set_ylabel("참으로 다를 때 잡아내는 비율", fontsize=11.5)
ax1.set_ylim(0, 1.12)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("치르는 값 — 정규 자료에서의 검정력", fontsize=12.5)

ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.plot(NS16, size_mean, "s-", color="0.55", lw=3.0, ms=12,
         label="평균 중심")
ax2.plot(NS16, size_med, "o-", color="tab:blue", lw=3.0, ms=12,
         label="중앙값 중심")
ax2.set_xlabel("표본크기 (정규 자료, 분산은 세 집단이 같다)",
               fontsize=11.5)
ax2.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, 0.1)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title("사 오는 것 — 명목선 아래에 머무는 성질", fontsize=12.5)
ax2.text(0.5, 0.08, "평균 중심은 명목선을 맴돌고\n"
         "중앙값 중심은 그 아래에 머문다", transform=ax2.transAxes,
         ha="center", va="bottom", fontsize=11.3, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("중심을 바꾸는 한 줄이 신호를 절반으로 줄인다 — 그 절반의 값",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, RT + "half_the_signal_costs_this.png")


# ════════════════════════════════════════════════════════════════════
# 17. fligner_killeen — 꼬리는 막고 왜도는 못 막는다
#
# 쪽의 그림 셋은 이상점 면역·중심 불일치·세 단계 변환을 다룬다. 들머리의
# **셋째 문단**이 남는다 — 「값은 모양이 바뀌어도 크기가 버틴다는 것이
# 이 검정의 성질이지만 무한정 버티지는 않는다. 왜도가 세지면 이쪽도
# 자유주의로 기울어 카이제곱 4 에서 0.055, 지수분포에서 0.087 이 된다.」
#
# 「꼬리는 막고 왜도는 못 막는다」는 **두 축**을 나란히 놓아야 보인다.
# 왼쪽은 대칭인 채 꼬리만 두껍게, 오른쪽은 왜도만 세게 한 것이다.
# ════════════════════════════════════════════════════════════════════
r17 = np.random.default_rng(15_218)
TAILS = [("정규", lambda n: r17.standard_normal(n)),
         ("t(10)", lambda n: r17.standard_t(10, n)),
         ("t(6)", lambda n: r17.standard_t(6, n)),
         ("t(4)", lambda n: r17.standard_t(4, n)),
         ("t(3)", lambda n: r17.standard_t(3, n))]
SKEWS = [("정규", lambda n: r17.standard_normal(n)),
         ("카이제곱 8", lambda n: r17.chisquare(8, n)),
         ("카이제곱 4", lambda n: r17.chisquare(4, n)),
         ("지수", lambda n: r17.exponential(1.0, n)),
         ("대수정규", lambda n: np.exp(r17.normal(0, 1, n)))]
K17, N17, REPS17 = 3, 30, 5000
tail_res = {nm: [] for nm in ("플리그너–킬린", "브라운–포사이드", "바틀렛")}
skew_res = {nm: [] for nm in tail_res}
for nm, gen in TAILS:
    res = run(gen, K17, N17, REPS17, r17,
              tests={"플리그너–킬린": stats.fligner,
                     "브라운–포사이드": bf, "바틀렛": stats.bartlett})
    for k_ in tail_res:
        tail_res[k_].append(res[k_])
for nm, gen in SKEWS:
    res = run(gen, K17, N17, REPS17, r17,
              tests={"플리그너–킬린": stats.fligner,
                     "브라운–포사이드": bf, "바틀렛": stats.bartlett})
    for k_ in skew_res:
        skew_res[k_].append(res[k_])
print("\n[fligner_killeen] 꼬리 축과 왜도 축")
print("  꼬리: " + " ".join(f"{v:.3f}" for v in tail_res["플리그너–킬린"]))
print("  왜도: " + " ".join(f"{v:.3f}" for v in skew_res["플리그너–킬린"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
STY = [("플리그너–킬린", "tab:blue", "o-"), ("브라운–포사이드", "tab:green", "s-"),
       ("바틀렛", "tab:red", "^-")]
for ax, data, names, title in [
        (ax1, tail_res, [nm for nm, _ in TAILS],
         "대칭인 채 꼬리만 두껍게"),
        (ax2, skew_res, [nm for nm, _ in SKEWS], "왜도만 세게")]:
    xx = np.arange(len(names))
    ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
    for nm, c, mk in STY:
        ax.plot(xx, data[nm], mk, color=c, lw=3.0, ms=11, label=nm)
    ax.set_xticks(xx, names, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 0.82)
    ax.grid(alpha=0.25)
ax1.set_ylabel("아무 차이가 없는데도 기각하는 비율", fontsize=11.5)
ax1.legend(fontsize=11, loc="upper left")
ax1.annotate("꼬리가 아무리 두꺼워져도 평평하다",
             xy=(3, tail_res["플리그너–킬린"][3]), xytext=(0.15, 0.32),
             fontsize=11.5, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax2.annotate(f"왜도에는 끌려간다\n"
             f"({skew_res['플리그너–킬린'][0]:.3f} → "
             f"{skew_res['플리그너–킬린'][-1]:.3f})",
             xy=(3.9, skew_res["플리그너–킬린"][-1]), xytext=(1.0, 0.38),
             fontsize=11.5, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

fig.suptitle("「가장 로버스트」라는 꼬리표는 조건부로 읽어야 한다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, RT + "stops_tails_not_skew.png")


# ════════════════════════════════════════════════════════════════════
# 18. robust_tests — 이상점 하나를 키워 가면 네 p 값이 갈라진다
#
# 쪽의 four_lenses 는 「같은 자료를 네 변환으로 본 모습」을 그린다 —
# 한 자리에서의 사진이다. 들머리는 그 사진이 **움직이면** 어떻게 되는지를
# 묻는다 — 「이상점 하나 때문에 p 값이 0.1339 에서 0.7273 까지 나온다.」
# 심은 이상점의 크기를 손잡이로 돌려 보면 네 곡선이 갈라지는 자리가
# 보이고, 그것이 「어느 요약을 쓰는가」의 값이다.
# ════════════════════════════════════════════════════════════════════
r18 = np.random.default_rng(15_219)
OUT = np.linspace(0, 8, 17)
K18, N18, REPS18 = 3, 12, 3000
curves = {nm: [] for nm in TESTS}
for o in OUT:
    cnt = {nm: 0 for nm in TESTS}
    for _ in range(REPS18):
        g = [r18.standard_normal(N18) for _ in range(K18)]
        g[0] = g[0].copy()
        g[0][0] = o          # 첫 집단에 이상점 하나를 심는다
        for nm, fn in TESTS.items():
            cnt[nm] += fn(*g).pvalue < 0.05
    for nm in TESTS:
        curves[nm].append(cnt[nm] / REPS18)
print("\n[robust_tests] 심은 이상점의 크기를 키워 간다 "
      f"(집단 {K18}, 집단당 {N18}, 등분산이 참)")
for nm in TESTS:
    print(f"  {nm:14s}: " + " ".join(f"{v:.2f}" for v in curves[nm][::4]))

fig, ax = plt.subplots(figsize=(11.0, 5.8))
STY18 = [("바틀렛", "tab:red", "-"), ("레빈 (평균 중심)", "tab:orange", "-"),
         ("브라운–포사이드", "tab:blue", "-"), ("플리그너–킬린", "tab:green", "-")]
for nm, c, ls in STY18:
    ax.plot(OUT, curves[nm], ls, color=c, lw=3.2, label=nm)
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.text(0.1, 0.075, "약속한 5% — 등분산이 참이므로 네 선 모두 여기 있어야 한다",
        ha="left", va="bottom", fontsize=11.3, color="0.2",
        fontweight="bold")
ax.set_xlabel("첫 집단에 심은 이상점의 크기 (표준편차 단위)", fontsize=12)
ax.set_ylabel("「분산이 다르다」고 답하는 비율", fontsize=12)
ax.set_ylim(0, 1.02)
ax.set_xlim(0, 8)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
i6 = int(np.argmin(np.abs(OUT - 6)))
ax.annotate(f"6 표준편차 하나에\n바틀렛은 {curves['바틀렛'][i6]:.2f}",
            xy=(6, curves["바틀렛"][i6]), xytext=(3.4, 0.86), fontsize=11.5,
            color="tab:red", fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.annotate(f"같은 자리에서 순위로 바꾼 쪽은 "
            f"{curves['플리그너–킬린'][i6]:.2f}",
            xy=(6, curves["플리그너–킬린"][i6]), xytext=(3.4, 0.22),
            fontsize=11.5, color="tab:green", fontweight="bold",
            ha="right",
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.9))
ax.set_title("자료도 하나이고 귀무가설도 참인데 결론이 갈린다", fontsize=13)
fig.suptitle("이상점 하나가 — 달라진 것은 요약하는 방식뿐이다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, RT + "one_outlier_pulls_them_apart.png")


# ════════════════════════════════════════════════════════════════════
# 19. comparison — 크기를 먼저 통과시킨 뒤에 검정력을 따진다
#
# 쪽의 size_power_tradeoff 가 「네 검정의 크기와 검정력을 함께 놓고」
# 그려 들머리의 첫·둘째 문단을 담는다. 셋째 문단은 **순서**를 말한다 —
# 「무엇을 먼저 볼 것인가. 크기다. 「기각률이 20% 인 5% 검정」은 검정력이
# 높아도 검정이 아니다. 명목 수준을 지키는 것이 통과 조건이고, 검정력은
# 그 조건을 통과한 것들 사이에서 따질 이야기다.」
#
# 그래서 흩뿌린 그림이 아니라 **두 단계 문**으로 그린다. 첫 칸이 통과를
# 가르고, 둘째 칸에는 통과한 것만 남는다.
# ════════════════════════════════════════════════════════════════════
r19 = np.random.default_rng(15_220)
POP19 = [("정규", lambda n: r19.standard_normal(n)),
         ("t(5)", lambda n: r19.standard_t(5, n)),
         ("지수", lambda n: r19.exponential(1.0, n)),
         ("대수정규", lambda n: np.exp(r19.normal(0, 1, n)))]
K19, N19, REPS19 = 3, 30, 5000
worst = {nm: 0.0 for nm in TESTS}
for pnm, gen in POP19:
    res = run(gen, K19, N19, REPS19, r19, tests=TESTS)
    for nm in TESTS:
        worst[nm] = max(worst[nm], res[nm])
pw19 = run(lambda n: r19.standard_normal(n), K19, N19, 5000, r19,
           ratios=np.array([1.0, 1.0, 3.0]), tests=TESTS)
print("\n[comparison] 네 모집단에서의 **가장 나쁜** 크기와 정규에서의 검정력")
for nm in TESTS:
    print(f"  {nm:14s}: 최악 크기 {worst[nm]:.4f} · 검정력 {pw19[nm]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
names19 = list(TESTS)
w_vals = [worst[nm] for nm in names19]
p_vals = [pw19[nm] for nm in names19]
PASS = 0.075
cols = ["tab:green" if v <= PASS else "tab:red" for v in w_vals]
xx = np.arange(len(names19))
ax1.bar(xx, w_vals, color=cols, width=0.6)
for i, v in enumerate(w_vals):
    ax1.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
             fontsize=12, fontweight="bold")
ax1.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax1.axhline(PASS, color="tab:red", lw=2.2, ls=":")
ax1.text(2.0, PASS + 0.022, f"통과선 {PASS} (점선) · 명목 0.05 (파선)",
         ha="center", va="bottom", fontsize=11.3, color="tab:red",
         fontweight="bold")
ax1.set_xticks(xx, [nm.replace(" ", "\n") for nm in names19], fontsize=10.5)
ax1.set_ylabel("네 모집단에서 가장 나빴던 크기", fontsize=11.5)
ax1.set_ylim(0, 0.85)
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("① 먼저 크기를 본다 — 여기서 떨어지면 검정이 아니다",
              fontsize=12)

keep = [i for i, v in enumerate(w_vals) if v <= PASS]
drop = [i for i, v in enumerate(w_vals) if v > PASS]
ax2.bar([i for i in range(len(names19))],
        [p_vals[i] if i in keep else 0 for i in range(len(names19))],
        color="tab:green", width=0.6)
for i in range(len(names19)):
    if i in keep:
        ax2.text(i, p_vals[i] + 0.02, f"{p_vals[i]:.3f}", ha="center",
                 va="bottom", fontsize=12, fontweight="bold")
    else:
        ax2.text(i, 0.04, f"크기에서 탈락\n(검정력 {p_vals[i]:.3f} 이지만)",
                 ha="center", va="bottom", fontsize=10.8, color="tab:red",
                 fontweight="bold")
ax2.set_xticks(xx, [nm.replace(" ", "\n") for nm in names19], fontsize=10.5)
ax2.set_ylabel("정규 자료에서 분산비 3 을 잡아내는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.02)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("② 통과한 것들 사이에서만 검정력을 따진다", fontsize=12)

fig.suptitle("검정력이 높은 쪽이 크기를 못 지킨다 — 그래서 순서가 있다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, RT + "size_first_then_power.png")


# ════════════════════════════════════════════════════════════════════
# 20. robust_tests_comparison — 무너지는 순서가 있다
#
# 쪽의 lognormal_center_stability 는 **한** 대수정규에서 네 검정의 거짓
# 양성률을 그린다. 들머리의 머리글은 그 한 자리의 수지만, 「넷 가운데
# 셋이 무너진다」를 보려면 왜도를 **연속으로** 키워야 무너지는 **순서**가
# 드러난다. 어느 검정이 먼저 꺾이고 어디까지 버티는가가 이 쪽의 값이다.
# ════════════════════════════════════════════════════════════════════
r20 = np.random.default_rng(15_221)
SIGS = [0.1, 0.3, 0.5, 0.75, 1.0, 1.25]
skews20 = [(np.exp(s ** 2) + 2) * np.sqrt(np.exp(s ** 2) - 1) for s in SIGS]
K20, N20, REPS20 = 3, 30, 5000
tracks20 = {nm: [] for nm in TESTS}
for s in SIGS:
    res = run(lambda n, s=s: np.exp(r20.normal(0, s, n)), K20, N20,
              REPS20, r20, tests=TESTS)
    for nm in TESTS:
        tracks20[nm].append(res[nm])
print("\n[robust_tests_comparison] 왜도를 키우면 무너지는 순서")
print("  왜도: " + " ".join(f"{v:7.2f}" for v in skews20))
for nm in TESTS:
    print(f"  {nm:14s}: " + " ".join(f"{v:7.3f}" for v in tracks20[nm]))

fig, ax = plt.subplots(figsize=(11.2, 5.9))
for nm, c, ls in STY18:
    ax.plot(skews20, tracks20[nm], "o-", color=c, lw=3.2, ms=11, label=nm)
ax.axhline(0.05, color="0.2", lw=2.4, ls="--")
ax.axhline(0.10, color="0.45", lw=1.8, ls=":")
ax.set_xscale("log")
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에서 두부가 된다
ax.set_xticks(skews20, [f"{v:.2f}" if v < 10 else f"{v:.0f}"
                        for v in skews20])
ax.set_xlabel("모집단의 왜도 (로그 척도) — 분산은 어느 자리에서나 "
              "세 집단이 같다", fontsize=11.5)
ax.set_ylabel("「분산이 다르다」고 답하는 비율", fontsize=12)
ax.set_ylim(0, 0.92)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
ax.text(skews20[0], 0.115, "두 배 (0.10)", ha="left", va="bottom",
        fontsize=10.8, color="0.4", fontweight="bold")
for nm, c in [("바틀렛", "tab:red"), ("레빈 (평균 중심)", "tab:orange"),
              ("플리그너–킬린", "tab:green")]:
    tr = tracks20[nm]
    j = next((i for i, v in enumerate(tr) if v > 0.10), None)
    if j:
        ax.plot([skews20[j]], [tr[j]], "o", ms=20, mfc="none", mec=c,
                mew=2.6)
ax.annotate("셋이 차례로 두 배 선을 넘는 동안\n"
            "중앙값 중심 하나만 아래에 남는다",
            xy=(skews20[-2], tracks20["브라운–포사이드"][-2] + 0.015),
            xytext=(1.2, 0.52), fontsize=12, color="tab:blue",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.set_title("왜도를 연속으로 키우면 무너지는 차례가 드러난다",
             fontsize=13)
fig.suptitle("로버스트하다는 넷 가운데 셋이 무너진다", fontsize=15, y=0.975)
plt.tight_layout()
save(fig, RT + "the_order_they_break.png")
