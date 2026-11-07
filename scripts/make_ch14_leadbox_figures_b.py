"""14장 들어가는 이야기 그림 — 둘째 그룹 여섯.

  1. qq_plots            정상의 폭이 자리마다 다르다 (꼬리에서 여섯 배)
  2. qq_plots_financial  과소평가는 꼬리로 갈수록 빠르게 커진다
  3. skewness_kurtosis   차수가 올라갈수록 같은 정밀도가 비싸진다
  4. jarque_bera         작은 p 값으로 갈수록 불일치가 커진다
  5. dagostino           사십이 년이 걸린 변환이 한 일
  6. kurtosistest        중심도 제자리가 아니다 — 그리고 펴기 전과 뒤

만드는 파일:
  docs/ch14/graphical_methods/img/normal_wiggle_by_position.png
  docs/ch14/graphical_methods/img/underestimate_grows_in_the_tail.png
  docs/ch14/descriptive_statistics/img/higher_moments_cost_more.png
  docs/ch14/descriptive_statistics/img/the_small_p_is_the_wrong_one.png
  docs/ch14/descriptive_statistics/img/what_the_transform_bought.png
  docs/ch14/descriptive_statistics/img/center_is_off_too.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

GRAPH = "docs/ch14/graphical_methods/img/"
DESC = "docs/ch14/descriptive_statistics/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. qq_plots — 정상의 폭이 자리마다 다르다
#
# 쪽의 qq_band_* 넷은 **한 그림 위에** 띠를 둘러 보인다. 들머리가 드는
# 수는 그것이 아니라 비(比)다 — 가운데 682 점의 최대 잔차가 0.0544 인데
# 꼬리 46 점은 0.3587 로 6.6 배. 곧 「정상의 폭이 자리마다 몇 배나
# 다른가」가 요점이고, 그것은 띠를 두른 한 그림이 아니라 **위치에 대한
# 곡선**으로 그려야 보인다. 띠의 모양이 왜 나팔꼴인지도 여기서 나온다.
# ════════════════════════════════════════════════════════════════════
NQ, REPS1 = 1000, 3000
r1 = np.random.default_rng(14_101)
pp = (np.arange(1, NQ + 1) - 0.375) / (NQ + 0.25)     # 블롬 자리
theo = stats.norm.ppf(pp)
res = np.sort(r1.standard_normal((REPS1, NQ)), axis=1) - theo
band = np.percentile(np.abs(res), 95, axis=0)

mid = (pp >= 0.1587) & (pp <= 0.8413)                  # 가운데 1 표준편차
tail = (pp < 0.025) | (pp > 0.975)
print("[qq_plots] 참으로 정규인 자료에서 자리마다의 정상 폭")
print(f"  가운데 {mid.sum()} 점의 평균 폭 {band[mid].mean():.4f}")
print(f"  양 끝 {tail.sum()} 점의 평균 폭 {band[tail].mean():.4f}")
print(f"  가장 바깥 한 점의 폭 {max(band[0], band[-1]):.4f}  "
      f"(가운데의 {max(band[0], band[-1]) / band[mid].mean():.1f} 배)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

ax1.fill_between(theo, -band, band, color="tab:blue", alpha=0.22,
                 label="참으로 정규일 때 점이 흔들리는 폭 (95%)")
one = np.sort(r1.standard_normal(NQ)) - theo
ax1.plot(theo, one, "o", ms=2.8, color="0.3",
         label="정규에서 뽑은 표본 하나")
ax1.axhline(0, color="tab:red", lw=2.0)
ax1.set_xlim(-3.6, 3.6)
ax1.set_ylim(-0.75, 0.75)
ax1.set_xlabel("정규 분위수 (그림에서 점이 놓이는 자리)", fontsize=11.5)
ax1.set_ylabel("직선에서 벗어난 양", fontsize=11.5)
ax1.set_title("폭이 가운데는 좁고 양 끝은 넓다", fontsize=13)
ax1.legend(fontsize=10.3, loc="upper center")
ax1.grid(alpha=0.25)

ax2.plot(theo, band, "-", color="tab:blue", lw=3.0)
ax2.axhline(band[mid].mean(), color="0.35", lw=1.8, ls=":")
ax2.text(0, band[mid].mean() + 0.012, f"가운데의 폭 {band[mid].mean():.4f}",
         ha="center", va="bottom", fontsize=11, color="0.3",
         fontweight="bold")
imax = int(np.argmax(band))
ax2.annotate(f"가장 바깥 점은 {band[imax]:.4f}\n"
             f"가운데의 {band[imax] / band[mid].mean():.1f} 배",
             xy=(theo[imax] - 0.12, band[imax]), xytext=(0.15, 0.60),
             fontsize=11.5, color="tab:blue", fontweight="bold",
             ha="center",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax2.set_xlim(-3.6, 3.6)
ax2.set_ylim(0, 0.86)
ax2.set_xlabel("정규 분위수", fontsize=11.5)
ax2.set_ylabel("그 자리에서 정상인 이탈의 폭", fontsize=11.5)
ax2.set_title("「양 끝이 벌어졌다」는 그 자체로는 아무 말도 아니다",
              fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("참으로 정규인데도 양 끝은 가운데보다 몇 배나 흔들린다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, GRAPH + "normal_wiggle_by_position.png")


# ════════════════════════════════════════════════════════════════════
# 2. qq_plots_financial — 과소평가는 꼬리로 갈수록 빠르게 커진다
#
# 쪽의 qq_normal_vs_t6 와 qq_plots_financial_* 는 **그림의 모양**을
# 다룬다. black_monday 는 하루의 드묾을 다룬다. 들머리에서 그 셋이
# 건드리지 않는 것은 「이 차이가 돈으로 바뀌는 자리」다 — 99% 한도가
# 명목의 1.39 배로 깨지고 99.9% 에서는 4.04 배가 된다는 그 두 수.
# 그래서 **분위수 깊이에 따른 과소평가**만 그린다.
# ════════════════════════════════════════════════════════════════════
DF2 = 8
sc2 = np.sqrt((DF2 - 2) / DF2)        # 분산을 1 로 맞춘다
LEVELS = np.array([0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995])
z = stats.norm.ppf(LEVELS)            # 정규 가정으로 그은 한도
breach = stats.t.sf(z / sc2, DF2)     # 실제로 그 한도를 넘는 비율
nominal = 1 - LEVELS
ratio = breach / nominal
print("\n[qq_plots_financial] 정규로 그은 한도가 실제로 얼마나 자주 깨지는가")
for L, nb, b, r in zip(LEVELS, nominal, breach, ratio):
    print(f"  {100 * L:6.2f}% 한도:  약속한 초과율 {nb:.5f} · "
          f"실제 {b:.5f}  → {r:.2f} 배")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
bx = np.arange(len(LEVELS))
ax1.bar(bx, ratio, color=["0.55" if r < 2 else "tab:red" for r in ratio],
        width=0.6)
for i, r in enumerate(ratio):
    ax1.text(i, r + 0.09, f"{r:.2f} 배", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax1.axhline(1.0, color="tab:blue", lw=2.2, ls="--")
ax1.text(0.03, 0.96, "가로 점선 — 약속한 만큼만 깨지면 1 배",
         transform=ax1.transAxes, ha="left", va="top", fontsize=11,
         color="tab:blue", fontweight="bold")
ax1.set_xticks(bx, [f"{100 * L:g}%" for L in LEVELS], fontsize=11)
ax1.set_xlabel("정규 가정으로 그은 손실 한도의 수준", fontsize=11.5)
ax1.set_ylabel("실제로 깨지는 비율 ÷ 약속한 비율", fontsize=11.5)
ax1.set_ylim(0, max(ratio) * 1.3)
ax1.set_title("깊이 들어갈수록 더 자주 깨진다", fontsize=13)
ax1.grid(axis="y", alpha=0.25)

qn = stats.norm.ppf(LEVELS)
qt = stats.t.ppf(LEVELS, DF2) * sc2
# 가로를 퍼센트로 놓으면 99 위쪽이 한 점에 뭉친다 — 왼쪽과 같은 칸 눈금으로
ax2.plot(bx, qn, "o-", color="tab:blue", lw=2.8, ms=11,
         label="정규가 적는 한도")
ax2.plot(bx, qt, "s-", color="tab:red", lw=2.8, ms=10,
         label="꼬리를 두껍게 잡은 한도")
for i in range(len(LEVELS)):
    d = 100 * (qt[i] / qn[i] - 1)
    ax2.annotate("", xy=(i, qt[i]), xytext=(i, qn[i]),
                 arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.8))
    ax2.text(i, max(qt[i], qn[i]) + 0.10, f"{d:+.0f}%", ha="center",
             va="bottom", fontsize=10.8,
             color="tab:red" if d > 0 else "0.45", fontweight="bold")
ax2.set_xticks(bx, [f"{100 * L:g}%" for L in LEVELS], fontsize=11)
ax2.set_xlabel("손실 한도의 수준", fontsize=11.5)
ax2.set_ylabel("한도를 그어야 하는 깊이 (표준편차 단위)", fontsize=11.5)
ax2.set_ylim(0.9, 5.3)
ax2.set_title("같은 확률을 적어 둔 한도가 더 깊은 곳에 있어야 했다",
              fontsize=12.5)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(alpha=0.25)
ax2.text(0.02, 0.60, "얕은 곳에서는 오히려 덜 깊다 —\n"
         "두꺼운 꼬리는 어깨가 얇기 때문이다", transform=ax2.transAxes,
         ha="left", va="top", fontsize=10.6, color="0.35",
         fontweight="bold")

plt.tight_layout()
fig.suptitle("「양 끝이 조금 벌어졌다」가 돈으로 바뀌는 자리", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, GRAPH + "underestimate_grows_in_the_tail.png")


# ════════════════════════════════════════════════════════════════════
# 3. skewness_kurtosis — 차수가 올라갈수록 같은 정밀도가 비싸진다
#
# 쪽의 moment_weights_tails 는 「차수가 높아질수록 꼬리의 관측값이
# 지배적이 된다」를 그린다 — 곧 **왜 흔들리는가**다. 들머리가 드는 수는
# 그것이 아니라 **얼마나**다(왜도 0.0339 가 표준오차의 0.44 배, 지수의
# 2.0526 은 27 배). 그래서 네 적률의 표준오차를 n 에 대해 나란히 놓고,
# 그 눈금 위에 두 관측값을 얹는다. 「수 하나만 보고는 판정이 되지 않고
# 그 수를 표집변동으로 나눈 뒤에야 판정이 된다」가 그림이 된다.
# ════════════════════════════════════════════════════════════════════
NS3 = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
REPS3 = 8000
r3 = np.random.default_rng(14_103)
se = {k: [] for k in ("평균", "표준편차", "왜도", "첨도")}
for n in NS3:
    d = r3.standard_normal((REPS3, n))
    se["평균"].append(d.mean(1).std())
    se["표준편차"].append(d.std(1, ddof=1).std())
    se["왜도"].append(stats.skew(d, axis=1).std())
    se["첨도"].append(stats.kurtosis(d, axis=1).std())

n1000 = int(np.where(NS3 == 1000)[0][0])
se_skew_1000 = se["왜도"][n1000]
OBS_NORMAL, OBS_EXP = 0.0339, 2.0526
print("\n[skewness_kurtosis] 네 적률의 표준오차 (정규 자료)")
for n, a, b, c, d_ in zip(NS3, se["평균"], se["표준편차"], se["왜도"],
                          se["첨도"]):
    print(f"  n = {n:5d}:  평균 {a:.4f} · 표준편차 {b:.4f} · "
          f"왜도 {c:.4f} · 첨도 {d_:.4f}")
print(f"  n = 1000 에서 왜도의 표준오차 {se_skew_1000:.5f} "
      f"(쪽은 0.07734)")
print(f"  관측 {OBS_NORMAL} 은 그 {OBS_NORMAL / se_skew_1000:.2f} 배 · "
      f"지수의 {OBS_EXP} 는 {OBS_EXP / se_skew_1000:.1f} 배")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
STYLE = [("평균", "0.45", "o-"), ("표준편차", "tab:green", "s-"),
         ("왜도", "tab:blue", "^-"), ("첨도", "tab:red", "D-")]
for nm, col, mk in STYLE:
    ax1.plot(NS3, se[nm], mk, color=col, lw=2.6, ms=9, label=nm)
ax1.set_xscale("log")
ax1.set_yscale("log")
# 로그 눈금 이름표를 맨 문자열로 — 기본 이름표는 mathtext 라 두부가 된다
ax1.set_xticks(NS3, [f"{n:,}" for n in NS3])
ax1.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3],
               ["0.01", "0.03", "0.1", "0.3", "1", "3"])
ax1.set_xlabel("표본크기 n (로그 척도)", fontsize=11.5)
ax1.set_ylabel("그 통계량이 흔들리는 폭 (표준오차)", fontsize=11.5)
ax1.set_title("차수가 올라갈수록 같은 정밀도가 비싸진다", fontsize=13)
ax1.legend(fontsize=11, loc="lower left")
ax1.grid(alpha=0.25, which="both")
ax1.annotate("첨도는 왜도보다 늘 위에 있다 —\n"
             "「꼬리가 두껍다」가 가장 늦게 믿을 만해진다",
             xy=(NS3[-2], se["첨도"][-2] * 1.12), xytext=(NS3[1], 1.9),
             fontsize=10.8, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

g = np.linspace(-3.2 * se_skew_1000, 3.2 * se_skew_1000, 400)
ax2.fill_between(g, 0, stats.norm.pdf(g, 0, se_skew_1000), color="0.80")
ax2.axvline(0, color="0.3", lw=1.8, ls="--")
ax2.axvline(OBS_NORMAL, color="tab:blue", lw=3.0)
ax2.annotate(f"정규 자료에서 잰 {OBS_NORMAL}\n"
             f"표준오차의 {OBS_NORMAL / se_skew_1000:.2f} 배 — 아무 말도 "
             f"아니다", xy=(OBS_NORMAL, 2.4), xytext=(0.075, 4.1),
             fontsize=11.3, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax2.annotate(f"지수분포에서 잰 {OBS_EXP}\n"
             f"표준오차의 {OBS_EXP / se_skew_1000:.0f} 배 — 눈금 밖이다",
             xy=(0.285, 1.0), xytext=(0.325, 3.1), fontsize=11.3,
             color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax2.annotate("", xy=(0.30, 0.9), xytext=(0.20, 0.9),
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=3.0))
ax2.set_xlim(-0.30, 0.33)
ax2.set_ylim(0, 6.2)
ax2.set_yticks([])
ax2.set_xlabel("n = 1000 에서 잰 왜도", fontsize=11.5)
ax2.set_title("정규 자료에서 왜도가 흔들리는 폭 위에 두 관측값을 얹으면",
              fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("수 하나만 보고는 판정이 되지 않는다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, DESC + "higher_moments_cost_more.png")


# ════════════════════════════════════════════════════════════════════
# 4. jarque_bera — 작은 p 값으로 갈수록 불일치가 커진다
#
# 쪽의 jb_chi2_calibration 이 「n = 50 에서 JB 의 귀무분포와 카이제곱
# 근사의 불일치」을 이미 그린다. 곧 들머리의 첫 문단(95 백분위가 5.991 이
# 아니라 4.97)은 그 그림의 것이다. 여기서는 **둘째 문단**만 그린다 —
# 「꾸러미가 p = 0.001 이라 적어 줄 때 실제 확률은 0.0080 으로 여덟 배다.
# 논문에 그대로 옮겨 적히는 수가 바로 이 자리의 수다.」
#
# 그래서 꾸러미가 적어 주는 p 를 가로로, 실제 확률을 세로로 놓는다.
# 45 도선에서 멀어지는 방향이 **증거를 강하게 보이게 하는 쪽**이다.
# ════════════════════════════════════════════════════════════════════
NS4 = [30, 50, 100, 500]
REPS4 = 400_000
PGRID = np.array([0.10, 0.05, 0.01, 0.005, 0.001, 0.0005])
r4 = np.random.default_rng(14_104)
actual = {}
for n in NS4:
    d = r4.standard_normal((REPS4, n))
    s, k = stats.skew(d, axis=1), stats.kurtosis(d, axis=1)
    jb = n / 6 * (s ** 2 + k ** 2 / 4)
    crit = stats.chi2.isf(PGRID, 2)            # 꾸러미가 쓰는 임계값
    actual[n] = np.array([(jb >= c).mean() for c in crit])

print("\n[jarque_bera] 꾸러미가 적어 주는 p 와 실제 확률")
for n in NS4:
    print(f"  n = {n}")
    for pn, pa in zip(PGRID, actual[n]):
        print(f"    적어 주는 p = {pn:<7g} 실제 {pa:.5f}  "
              f"({pa / pn:5.2f} 배)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
COLS = ["tab:red", "tab:orange", "tab:blue", "0.45"]
lim = [3e-4, 0.2]
ax1.plot(lim, lim, color="0.3", lw=2.2, ls="--")
for n, c in zip(NS4, COLS):
    ax1.plot(PGRID, actual[n], "o-", color=c, lw=2.6, ms=9, label=f"n = {n}")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
               ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"])
ax1.set_yticks([0.001, 0.005, 0.01, 0.05, 0.1],
               ["0.001", "0.005", "0.01", "0.05", "0.1"])
ax1.set_xlim(*lim)
ax1.set_ylim(5e-4, 0.2)
ax1.set_xlabel("꾸러미가 적어 주는 p 값", fontsize=11.5)
ax1.set_ylabel("실제로 그만큼 커질 확률", fontsize=11.5)
ax1.set_title("점선 위라면 적힌 대로다 — 큰 p 에서는 아래, 작은 p 에서는 위",
              fontsize=12)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(alpha=0.25, which="both")

for n, c in zip(NS4, COLS):
    ax2.plot(PGRID, actual[n] / PGRID, "o-", color=c, lw=2.6, ms=9,
             label=f"n = {n}")
ax2.axhline(1.0, color="0.3", lw=2.2, ls="--")
ax2.set_xscale("log")
ax2.set_xticks([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
               ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"])
ax2.invert_xaxis()
ax2.set_xlabel("꾸러미가 적어 주는 p 값 (오른쪽으로 갈수록 작다)",
               fontsize=11.5)
ax2.set_ylabel("실제 확률 ÷ 적어 주는 p 값", fontsize=11.5)
ax2.set_ylim(0, max(actual[50] / PGRID) * 1.28)
ax2.set_title("5% 에서는 조심스럽고 0.001 에서는 과장한다 — 방향이 뒤집힌다",
              fontsize=12)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(alpha=0.25)
i005 = int(np.where(PGRID == 0.05)[0][0])
ax2.annotate(f"흔히 쓰는 5% 에서는\n{actual[50][i005]:.4f} — 오히려 "
             f"조심스럽다", xy=(0.05, actual[50][i005] / 0.05),
             xytext=(0.098, 2.6), fontsize=11, color="0.3",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8))
i001 = int(np.where(PGRID == 0.001)[0][0])
ax2.annotate(f"n = 50 에서 p = 0.001 이라 적어 줄 때\n"
             f"실제는 {actual[50][i001]:.4f} — "
             f"{actual[50][i001] / 0.001:.1f} 배",
             xy=(0.001, actual[50][i001] / 0.001),
             xytext=(0.012, max(actual[50] / PGRID) * 1.1), fontsize=11.3,
             color="tab:orange", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.9))

fig.suptitle("적어 둔 임계값이 실제와 다르다 — 어긋나는 방향이 p 에 따라 "
             "뒤집힌다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, DESC + "the_small_p_is_the_wrong_one.png")


# ════════════════════════════════════════════════════════════════════
# 5. dagostino — 사십이 년이 걸린 변환이 한 일
#
# 쪽의 dagostino_z1_z2_plane 이 「Z1·Z2 평면과 원형 기각역」을 그려
# 들머리의 첫 문단(따로는 못 넘고 합치면 넘는다)을 이미 담고 있고,
# skew_vs_kurt_blindspot 이 둘째 문단(맹점)을 담는다.
# dagostino_simple_vs_scipy_z 는 두 **귀무분포**를 견준다.
#
# 남은 것은 셋째 문단이다 — 「n = 30 에서 단순 z 점수로 첨도를 재면 실제
# 기각률이 0.0225 로 명목의 절반도 되지 않는데, 변환을 거친 쪽은
# 0.0487 로 제자리에 온다. 그 변환을 손에 넣는 데 사십이 년이 걸렸다.」
# 귀무분포가 아니라 **기각률**을 n 에 대해 그린다.
# ════════════════════════════════════════════════════════════════════
NS5 = [15, 20, 30, 50, 100, 200, 500, 1000]
REPS5 = 60_000
r5 = np.random.default_rng(14_105)
simple_k, scipy_k, simple_s, scipy_s = [], [], [], []
for n in NS5:
    d = r5.standard_normal((REPS5, n))
    g1, g2 = stats.skew(d, axis=1), stats.kurtosis(d, axis=1)
    # 단순 표준화 — 점근 표준오차로 나누기만 한 것
    se1 = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    se2 = np.sqrt(24 * n * (n - 1) ** 2 /
                  ((n - 3) * (n - 2) * (n + 3) * (n + 5)))
    simple_s.append((np.abs(g1 / se1) > 1.959964).mean())
    simple_k.append((np.abs(g2 / se2) > 1.959964).mean())
    # scipy 의 변환된 z 점수
    scipy_s.append((stats.skewtest(d, axis=1).pvalue < 0.05).mean())
    scipy_k.append((stats.kurtosistest(d, axis=1).pvalue < 0.05).mean())

print("\n[dagostino] 변환 전과 뒤의 실제 1 종 오류율 (명목 0.05)")
print("     n     왜도(단순)  왜도(변환)  첨도(단순)  첨도(변환)")
for n, a, b, c, d_ in zip(NS5, simple_s, scipy_s, simple_k, scipy_k):
    print(f"  {n:5d}   {a:9.4f}  {b:9.4f}  {c:9.4f}  {d_:9.4f}")

fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
for ax, (nm, sim, tr, col) in zip(axs, [
        ("왜도", simple_s, scipy_s, "tab:blue"),
        ("첨도", simple_k, scipy_k, "tab:red")]):
    ax.axhline(0.05, color="0.3", lw=2.2, ls="--")
    ax.plot(NS5, sim, "o--", color="0.55", lw=2.6, ms=10,
            label="표준오차로 나누기만 한 점수")
    ax.plot(NS5, tr, "s-", color=col, lw=2.8, ms=10,
            label="변환을 거친 점수")
    ax.set_xscale("log")
    ax.set_xticks(NS5, [f"{n:,}" for n in NS5])
    ax.set_xlabel("표본크기 n (로그 척도)", fontsize=11.5)
    ax.set_title(f"{nm} 검정", fontsize=13.5)
    ax.set_ylim(0, 0.072)
    ax.legend(fontsize=10.6, loc="lower right")
    ax.grid(alpha=0.25)
axs[0].set_ylabel("실제 1 종 오류율 (명목은 0.05)", fontsize=11.5)
i30 = NS5.index(30)
axs[1].annotate(f"n = 30 에서 {simple_k[i30]:.4f} — 명목의 절반도 안 된다",
                xy=(30, simple_k[i30]), xytext=(60, 0.013), fontsize=11,
                color="0.4", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=1.8))
axs[1].annotate(f"변환을 거치면 {scipy_k[i30]:.4f}", xy=(30, scipy_k[i30]),
                xytext=(14, 0.0655), fontsize=11, color="tab:red",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("사십이 년이 걸린 변환이 한 일 — 그냥 더하면 되는 것이 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, DESC + "what_the_transform_bought.png")


# ════════════════════════════════════════════════════════════════════
# 6. kurtosistest — 중심도 제자리가 아니다
#
# 쪽의 kurt_g2_null_skew 가 「표본 초과첨도의 귀무분포가 n = 50, 200,
# 1000 에서 어떤 모양인지」를 그린다. 곧 들머리의 첫 문단(치우쳐 있다)은
# 그 그림의 것이다. 남은 beat 가 둘이고 둘 다 그려져 있지 않다 —
# 「중심도 0 이 아니다. 정규성 아래에서 비보정 첨도의 기댓값이
# 3(n-1)/(n+1) 이라 n = 300 에서 2.980066 이다」와 「단순 표준화는
# u = 3.7403 인데 변환하면 2.7603 이고 p 로는 31 배 차이다」.
#
# 왼쪽이 **중심의 편향**, 오른쪽이 **두 표준화가 같은 통계량에 매기는
# p 값**이다. 둘 다 펴기 전과 뒤를 가른다.
# ════════════════════════════════════════════════════════════════════
NS6 = np.array([20, 30, 50, 100, 200, 300, 500, 1000, 2000])
bias = 3 * (NS6 - 1) / (NS6 + 1) - 3.0       # 비보정 첨도의 기댓값 - 3
REPS6 = 60_000
r6 = np.random.default_rng(14_106)
meas = []
for n in NS6:
    d = r6.standard_normal((REPS6, n))
    m2 = ((d - d.mean(1, keepdims=True)) ** 2).mean(1)
    m4 = ((d - d.mean(1, keepdims=True)) ** 4).mean(1)
    meas.append((m4 / m2 ** 2).mean() - 3.0)

i300 = int(np.where(NS6 == 300)[0][0])
print("\n[kurtosistest] 비보정 첨도의 기댓값은 3 이 아니다")
for n, b, m in zip(NS6, bias, meas):
    print(f"  n = {n:5d}:  이론 {3 + b:.6f} · 재어 본 값 {3 + m:.6f}")
print(f"  n = 300 에서 이론이 {3 + bias[i300]:.6f} — 쪽이 적는 2.980066")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.axhline(0, color="tab:red", lw=2.4, ls="--")
ax1.text(2100, 0.0045, "3 이어야 할 자리", ha="right", va="bottom",
         fontsize=11.5, color="tab:red", fontweight="bold")
ax1.plot(NS6, bias, "-", color="0.72", lw=6.0, alpha=0.75,
         label="이론값")
ax1.plot(NS6, meas, "o", color="tab:blue", ms=11,
         label="되풀이해 재어 본 값")
ax1.set_xscale("log")
ax1.set_xticks(NS6, [f"{n:,}" for n in NS6])
ax1.set_xlabel("표본크기 n (로그 척도)", fontsize=11.5)
ax1.set_ylabel("비보정 첨도의 기댓값 빼기 3", fontsize=11.5)
ax1.set_ylim(-0.33, 0.045)
ax1.set_title("정규 자료에서도 3 을 밑돌게 나온다", fontsize=13)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25)
ax1.annotate(f"n = 300 에서 {3 + bias[i300]:.6f}", xy=(300, bias[i300]),
             xytext=(60, -0.17), fontsize=11.3, color="tab:blue",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

# 오른쪽 — 같은 통계량에 두 표준화가 매기는 p 값
N6B = 300
d6 = r6.standard_normal((200_000, N6B))
g2 = stats.kurtosis(d6, axis=1)
se2b = np.sqrt(24 * N6B * (N6B - 1) ** 2 /
               ((N6B - 3) * (N6B - 2) * (N6B + 3) * (N6B + 5)))
grid = np.linspace(0.0, 1.4, 160)
p_simple = np.array([2 * stats.norm.sf(v / se2b) for v in grid])
# 변환된 점수는 표본에서 직접 얻어 g2 순으로 세운 뒤 격자에 끼워 넣는다
order = np.argsort(g2)
g2s = g2[order]
ztr_all = stats.kurtosistest(d6, axis=1).statistic[order]
p_trans = np.interp(grid, g2s, 2 * stats.norm.sf(np.abs(ztr_all)))
ax2.plot(grid, p_simple, "-", color="0.55", lw=3.0,
         label="표준오차로 나누기만 하면")
ax2.plot(grid, p_trans, "-", color="tab:red", lw=3.0,
         label="변환을 거치면")
ax2.set_yscale("log")
ax2.set_yticks([1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.5],
               ["0.00001", "0.0001", "0.001", "0.01", "0.05", "0.5"])
ax2.axhline(0.05, color="0.3", lw=1.8, ls=":")
ax2.set_ylim(2e-6, 1.6)
ax2.set_xlim(0, 1.4)
ax2.set_xlabel(f"표본 초과첨도 (n = {N6B})", fontsize=11.5)
ax2.set_ylabel("매겨지는 p 값 (로그 척도)", fontsize=11.5)
ax2.set_title("같은 통계량에 두 표준화가 다른 p 를 매긴다", fontsize=12.5)
ax2.legend(fontsize=11, loc="lower left")
ax2.grid(alpha=0.25, which="both")
gmark = 0.95
ps = 2 * stats.norm.sf(gmark / se2b)
pt = float(np.interp(gmark, grid, p_trans))
ax2.annotate(f"초과첨도가 {gmark} 인 자리\n{ps:.1e} 대 {pt:.1e} — "
             f"{pt / ps:.0f} 배 다르다", xy=(gmark, ps * 1.6),
             xytext=(1.37, 0.14), fontsize=11.3, color="0.25",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.9))

fig.suptitle("아래로는 벽이 있고 위로는 없다 — 중심도 제자리가 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, DESC + "center_is_off_too.png")
