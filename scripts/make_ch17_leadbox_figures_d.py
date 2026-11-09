"""17장 들어가는 이야기 그림 — 넷째 그룹 다섯.

  21. paired                    대칭이 깨지면 어느 쪽으로 어긋나는가
  22. permutation/correlation   크기는 늘 지켜지고 검정력만 달라진다
  23. ab_testing_permutation    기각역을 미리 그려 보면
  24. resampling_shoe_sales     「효과가 없다」와 「판별할 수 없다」
  25. bootstrap_testing/corr    섞기·쌍째 뽑기·따로 뽑기가 내는 세 분포

만드는 파일:
  docs/ch17/permutation/img/skew_tilts_the_sign_flip.png
  docs/ch17/permutation/img/size_holds_power_does_not.png
  docs/ch17/permutation/img/the_rejection_region_first.png
  docs/ch17/permutation/img/cannot_tell_is_not_no_effect.png
  docs/ch17/bootstrap_testing/img/three_ways_to_resample_a_pair.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import itertools

import numpy as np
from scipy import stats

PRM = "docs/ch17/permutation/img/"
TEST = "docs/ch17/bootstrap_testing/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def signs(n, m, rng):
    """부호를 뒤집는 m 벌을 한꺼번에 만든다."""
    return rng.choice([-1.0, 1.0], (m, n))


# ════════════════════════════════════════════════════════════════════
# 21. paired — 대칭이 깨지면 어느 쪽으로 어긋나는가
#
# 쪽의 sign_flip_exact 가 부호 뒤집기와 4,096 가지가 만드는 정확
# 귀무분포를 그려 들머리의 첫 문단을 담는다. 그려지지 않은 것은
# **셋째 문단**이다 — 「그 가정이 깨지면 어긋나는 방향이 정해져 있다.
# … 귀무분포는 그 큰 값을 양쪽에 고르게 놓아 가운데가 실제보다
# 두꺼워진다. 그래서 … 거짓 기각이 늘어난다.」
#
# 「방향이 정해져 있다」는 **왜도를 손잡이로 돌려** 보아야 보인다.
# 평균은 0 으로 묶어 둔 채 왜도만 키우며 거짓 기각률을 잰다.
# 두 번째 칸이 까닭을 담는다 — 두 분포의 가운데가 어떻게 어긋나는지.
# ════════════════════════════════════════════════════════════════════
r21 = np.random.default_rng(17_021)
N21, REPS21, M21 = 12, 4000, 1024
SHAPES = [("정규 (대칭)", lambda m, n: r21.standard_normal((m, n))),
          ("약하게 왜도", lambda m, n: r21.gamma(8.0, 1.0, (m, n))),
          ("왜도", lambda m, n: r21.gamma(2.0, 1.0, (m, n))),
          ("세게 왜도", lambda m, n: r21.gamma(0.7, 1.0, (m, n))),
          ("아주 세게", lambda m, n: r21.gamma(0.3, 1.0, (m, n)))]
sizes, skews = [], []
for nm, gen in SHAPES:
    d = gen(REPS21, N21)
    d = d - d.mean(1, keepdims=True) * 0 - np.mean(d)   # 평균만 0 으로
    hit = 0
    for row in d:
        obs = abs(row.mean())
        flip = (signs(N21, M21, r21) * row).mean(1)
        hit += (np.abs(flip) >= obs).mean() < 0.05
    sizes.append(hit / REPS21)
    skews.append(float(stats.skew(d.ravel())))
    print(f"[paired] {nm:<14s} 왜도 {skews[-1]:+.3f} · "
          f"거짓 기각률 {sizes[-1]:.4f}")

# 두 번째 칸 — 까닭을 담는다. 처음에 「참 귀무분포와 부호 뒤집기 분포」를
# 겹쳐 그렸는데, 표본 여러 벌에 걸쳐 모으면 두 분포가 거의 같아진다
# (sd 0.1575 대 0.1410). 평균 꼴에는 기제가 없다 — 검정은 **그 표본의**
# 뒤집기 분포로 임계값을 정하므로, 봐야 할 것은 표본마다의 임계값이다.
gen_last = SHAPES[-1][1]
mu21 = float(stats.gamma.mean(0.3, scale=1.0))
# 모집단의 평균 mu21 을 빼서 귀무가 참인 자료로 만든다. 이걸 빼먹으면
# 두 분포가 0.3 만큼 밀려 임계값이 0.38 대신 0.68 로 나온다 (그랬다)
true_null = gen_last(200000, N21).mean(1) - mu21
true_crit = float(np.percentile(np.abs(true_null), 97.5))
crit = np.array([np.percentile(
    np.abs((signs(N21, 600, r21) * (row - mu21)).mean(1)), 97.5)
    for row in gen_last(2000, N21)])
below = float((crit < true_crit).mean())
print(f"  참 귀무분포의 97.5 백분위수 {true_crit:.4f}")
print(f"  부호 뒤집기가 내는 임계값: 가운데값 {np.median(crit):.4f} · "
      f"참값보다 작은 비율 {below:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3))
ax1.plot(skews, sizes, "o-", color="tab:red", lw=3.2, ms=13)
ax1.axhline(0.05, color="0.25", lw=2.4, ls="--")
ax1.text(skews[0], 0.0525, "명목 0.05", ha="left", va="bottom",
         fontsize=11.5, color="0.25", fontweight="bold")
for s, v, (nm, _) in zip(skews, sizes, SHAPES):
    ax1.annotate(nm, xy=(s, v), xytext=(0, -17),
                 textcoords="offset points", ha="center", fontsize=10.6,
                 color="0.3")
ax1.set_xlabel("차이의 왜도 (평균은 모두 0 으로 묶어 두었다)",
               fontsize=11.5)
ax1.set_ylabel("아무 효과가 없는데 기각하는 비율", fontsize=11.5)
ax1.set_ylim(0, max(sizes) * 1.3)
ax1.grid(alpha=0.25)
ax1.set_title("한쪽으로만 어긋난다 — 늘 거짓 기각이 늘는 쪽", fontsize=12.5)

ax2.hist(crit, bins=60, color="tab:red", alpha=0.72)
ax2.axvline(true_crit, color="0.15", lw=3.0)
ax2.text(true_crit + 0.012, ax2.get_ylim()[1] * 0.96,
         f" 참 임계값 {true_crit:.3f}", ha="left", va="top", fontsize=12.5,
         color="0.15", fontweight="bold")
ax2.axvline(float(np.median(crit)), color="tab:red", lw=2.4, ls="--")
ax2.set_xlabel("그 표본의 부호 뒤집기가 정한 기각 경계", fontsize=11.5)
ax2.set_ylabel("표본의 개수", fontsize=11.5)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"표본 {len(crit):,} 벌 가운데 {below * 100:.0f}% 가 "
              f"참값보다 낮은 경계를 쓴다", fontsize=12.5)
ax2.annotate(f"경계가 낮으면 관측값이 더 쉽게 넘는다 —\n"
             f"가운데값이 {np.median(crit):.3f} 로 참값의 "
             f"{np.median(crit) / true_crit:.2f} 배다",
             xy=(float(np.median(crit)), ax2.get_ylim()[1] * 0.55),
             xytext=(true_crit + 0.05, ax2.get_ylim()[1] * 0.70),
             fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("대칭이 깨지면 어긋나는 방향이 정해져 있다", fontsize=15,
             y=0.99)
plt.tight_layout()
save(fig, PRM + "skew_tilts_the_sign_flip.png")


# ════════════════════════════════════════════════════════════════════
# 22. permutation/correlation — 크기는 늘 지켜지고 검정력만 달라진다
#
# 쪽의 statistic_picks 가 「이차 종속 자료에서 통계량을 바꾸면 검정력이
# 어떻게 달라지는가」를 그려 들머리의 **셋째 문단**을 통째로 담는다.
# 그려지지 않은 것은 **둘째 문단의 앞 절반**이고 그것이 들머리의
# 제목이다 — 「거짓 기각률은 무엇을 재든 정확히 0.05 로 지켜지고,
# 통계량의 선택은 검정력만 바꾼다.」
#
# 검정력 쪽은 이미 그려져 있으니 **크기 쪽**을 그린다. 자료의 꼴을
# 험하게 바꾸어 가며 세 통계량의 크기를 재고, 고전적 검정을 같은 자에
# 함께 올린다. 셋이 선 위에 붙어 있고 고전적 검정만 떠나는 것이 요점.
# ════════════════════════════════════════════════════════════════════
r22 = np.random.default_rng(17_022)
N22, REPS22, NP22 = 20, 3000, 500
CASES = [
    ("이변량 정규", lambda m, n: (r22.standard_normal((m, n)),
                              r22.standard_normal((m, n)))),
    ("두꺼운 꼬리", lambda m, n: (r22.standard_t(1.5, (m, n)),
                             r22.standard_t(1.5, (m, n)))),
    ("한쪽만 지수", lambda m, n: (r22.exponential(1.0, (m, n)),
                             r22.standard_normal((m, n)))),
    ("다섯 단계 설문", lambda m, n: (r22.integers(1, 6, (m, n)) * 1.0,
                                r22.integers(1, 6, (m, n)) * 1.0)),
    # 이상치를 양쪽 같은 자리에 두면 x 와 y 가 정말 종속이 되어 버린다.
    # 자리를 각자 따로 섞어 독립을 지킨다 (처음에 그 실수를 했다)
    ("이상치 하나", lambda m, n: (
        np.take_along_axis(
            np.c_[r22.standard_normal((m, n - 1)),
                  r22.standard_normal(m) * 25],
            np.argsort(r22.random((m, n)), axis=1), axis=1),
        np.take_along_axis(
            np.c_[r22.standard_normal((m, n - 1)),
                  r22.standard_normal(m) * 25],
            np.argsort(r22.random((m, n)), axis=1), axis=1))),
]
# 셋째는 쪽의 statistic_picks 가 쓰는 「가운데에서 얼마나 멀리 있는가」다.
# 처음에 부호만 보는 꼴을 썼는데 n = 20 에서 값이 너무 거칠어 크기가
# 0.02 로 내려앉았다 — 이어진 통계량이어야 크기가 명목에 붙는다
STATS = [("피어슨",), ("스피어만",), ("가운데에서의 거리",)]
size_perm = {nm: [] for nm, _ in CASES}
size_class = []
print("\n[correlation] 아무 관계가 없는 자료에서 거짓 기각률 (명목 0.05)")
for nm, gen in CASES:
    x, y = gen(REPS22, N22)
    rx = np.apply_along_axis(stats.rankdata, 1, x)
    ry = np.apply_along_axis(stats.rankdata, 1, y)
    hits = [0, 0, 0]
    cls = 0
    for i in range(REPS22):
        xi, yi, rxi, ryi = x[i], y[i], rx[i], ry[i]
        perm = np.argsort(r22.random((NP22, N22)), axis=1)
        for k, (a, b) in enumerate(((xi, yi), (rxi, ryi))):
            ac = a - a.mean()
            bc = b - b.mean()
            obs = abs((ac * bc).mean())
            nd = np.abs((ac[perm] * bc).mean(1))
            hits[k] += (nd >= obs - 1e-12).mean() < 0.05
        # 셋째 — 가운데에서의 거리
        sx = np.abs(xi - np.median(xi))
        sy = np.abs(yi - np.median(yi))
        sx = sx - sx.mean()
        sy = sy - sy.mean()
        obs = abs((sx * sy).mean())
        nd = np.abs((sx[perm] * sy).mean(1))
        hits[2] += (nd >= obs - 1e-12).mean() < 0.05
        rr = np.corrcoef(xi, yi)[0, 1]
        t = rr * np.sqrt((N22 - 2) / max(1 - rr ** 2, 1e-12))
        cls += 2 * stats.t.sf(abs(t), N22 - 2) < 0.05
    size_perm[nm] = [h / REPS22 for h in hits]
    size_class.append(cls / REPS22)
    print(f"  {nm:<14s} 순열: " +
          " · ".join(f"{s[0]} {v:.4f}" for s, v
                     in zip(STATS, size_perm[nm])) +
          f"  | 고전적 검정 {cls / REPS22:.4f}")

fig, ax = plt.subplots(figsize=(12.4, 5.9))
names22 = [nm for nm, _ in CASES]
xx = np.arange(len(names22))
w = 0.20
cs = ["tab:blue", "tab:green", "tab:purple"]
for k, (snm,) in enumerate(STATS):
    vals = [size_perm[nm][k] for nm in names22]
    ax.bar(xx + (k - 1.5) * w, vals, width=w, color=cs[k],
           label=f"순열 — {snm}")
ax.bar(xx + 1.5 * w, size_class, width=w, color="tab:red",
       label="고전적 t 검정")
for i, v in enumerate(size_class):
    ax.text(xx[i] + 1.5 * w, v + 0.002, f"{v:.3f}", ha="center",
            va="bottom", fontsize=10.6, color="tab:red",
            fontweight="bold")
ax.axhline(0.05, color="0.15", lw=2.6, ls="--")
ax.text(len(names22) - 0.52, 0.0525, "명목 0.05", ha="right", va="bottom",
        fontsize=12, color="0.15", fontweight="bold")
ax.set_xticks(xx, names22, fontsize=12)
ax.set_ylabel("아무 관계가 없는데 기각하는 비율", fontsize=12)
ax.set_ylim(0, max(max(size_class), 0.08) * 1.45)
ax.legend(fontsize=11, ncols=2, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.set_title(f"n = {N22} · 자료 {REPS22:,} 벌 — 세 통계량이 자료의 꼴과 "
             f"무관하게 선을 넘지 않는다", fontsize=13)
ax.annotate("고전적 검정만 자료의 꼴에 끌려간다",
            xy=(1 + 1.5 * w, size_class[1]),
            xytext=(1.9, max(max(size_class), 0.08) * 1.3),
            fontsize=11.5, color="tab:red", fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
fig.suptitle("섞는 것이 크기를 지켜 준다 — 통계량의 선택은 검정력만 바꾼다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, PRM + "size_holds_power_does_not.png")


# ════════════════════════════════════════════════════════════════════
# 23. ab_testing_permutation — 기각역을 미리 그려 보면
#
# 쪽의 binary_permutation 과 peeking 은 이진 자료와 들여다보기를
# 그린다. 들머리의 **제목이자 둘째 문단**이 그려져 있지 않다 —
# 「양측 0.05 에서 기각하려면 평균차가 10.2 초를 넘어야 한다 —
# 관측된 5.4 초의 거의 두 배다.」
#
# 히스토그램에 임계값을 그으면 foundations 의 그림과 같은 꼴이 된다.
# 그래서 **임계값을 사람 수의 함수로** 그린다 — 그러면 들머리의
# 마지막 문단(「집단마다 400 명이 들고 … 25 명으로 된다」)까지 같은
# 그림에 들어오고, 관측된 5.4 가 어느 n 에서 기각선을 넘는지 보인다.
# ════════════════════════════════════════════════════════════════════
r23 = np.random.default_rng(17_023)
SD23, OBS23 = 11.69, 5.4          # 쪽의 자료에서 온 흩어짐과 관측 평균차
NS23 = [10, 15, 20, 30, 50, 80, 120, 200, 400]
crit, mde80 = [], []
NP23 = 4000
print(f"\n[ab_testing] 흩어짐 {SD23} · 관측 평균차 {OBS23}")
for n in NS23:
    if n <= 10:                    # 작은 n 은 전수로 열거한다
        pool = np.r_[r23.standard_normal(n), r23.standard_normal(n)]
        pool = (pool - pool.mean()) / pool.std(ddof=1) * SD23
        tt = pool.sum()
        d = np.array([pool[list(c)].sum() / n - (tt - pool[list(c)].sum()) / n
                      for c in itertools.combinations(range(2 * n), n)])
    else:
        # 흩어짐을 쪽의 값에 정확히 맞춘다. 그냥 뽑으면 그 표본의 산포가
        # n 마다 달라져 기각선이 들쭉날쭉해진다 (처음에 그랬다)
        pool = r23.standard_normal(2 * n)
        pool = (pool - pool.mean()) / pool.std(ddof=1) * SD23
        sh = pool[np.argsort(r23.random((NP23, 2 * n)), axis=1)]
        d = sh[:, :n].mean(1) - sh[:, n:].mean(1)
    c = float(np.percentile(np.abs(d), 95))
    crit.append(c)
    # 검정력 0.8 을 주는 참 차이 — 정규근사로 환산한다
    se = SD23 * np.sqrt(2.0 / n)
    mde80.append(float((1.959964 + 0.841621) * se))
    print(f"  집단마다 {n:3d} 명: 기각선 {c:6.2f} · 검정력 0.8 을 주는 "
          f"차이 {mde80[-1]:6.2f}")
# 격자의 n 만 보면 「50 명부터」가 되는데 곡선은 그보다 훨씬 왼쪽에서
# 지난다. 로그 눈금 위에서 보간해 실제로 만나는 자리를 쓴다
lg = np.interp(-OBS23, [-c for c in crit], np.log(NS23))
cross = float(np.exp(lg))
print(f"  관측된 {OBS23} 가 기각선과 만나는 자리 n = {cross:.0f} "
      f"(격자 위에서 처음 넘는 n 은 "
      f"{next(n for n, c in zip(NS23, crit) if c <= OBS23)})")

fig, ax = plt.subplots(figsize=(12.0, 6.0))
ax.plot(NS23, crit, "o-", color="tab:red", lw=3.2, ms=12,
        label="양측 0.05 의 기각선 — 이만큼 넘어야 유의하다")
ax.plot(NS23, mde80, "s--", color="tab:purple", lw=2.6, ms=10,
        label="검정력 0.8 을 주는 참 차이")
ax.axhline(OBS23, color="tab:blue", lw=3.0)
ax.text(NS23[0], OBS23 + 0.4, f"관측된 평균차 {OBS23} 초", ha="left",
        va="bottom", fontsize=12.5, color="tab:blue", fontweight="bold")
ax.scatter([10], [crit[0]], s=330, facecolor="none", edgecolor="tab:red",
           lw=3.0, zorder=5)
ax.annotate(f"집단마다 10 명이면 기각선이 {crit[0]:.1f} 초 —\n"
            f"관측값의 {crit[0] / OBS23:.1f} 배다",
            xy=(10.6, crit[0]), xytext=(22, crit[0] + 2.6), fontsize=12,
            color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.axvline(cross, color="0.4", lw=1.8, ls=":")
ax.annotate(f"{cross:.0f} 명쯤부터\n이 크기가 유의해진다",
            xy=(cross, OBS23), xytext=(cross * 1.5, OBS23 * 0.45),
            fontsize=11.5, color="0.2", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=1.8))
ax.set_xscale("log")
ax.set_xticks(NS23, [str(n) for n in NS23], fontsize=11.5)
ax.set_xlabel("집단마다의 사람 수", fontsize=12)
ax.set_ylabel("평균차 (초)", fontsize=12)
ax.set_ylim(0, max(max(crit), max(mde80)) * 1.08)
ax.legend(fontsize=11.5, loc="lower left")
ax.grid(alpha=0.25)
ax.set_title(f"자료의 흩어짐 {SD23} 초를 고정해 두고 사람 수만 바꾸었다",
             fontsize=13)
fig.suptitle("기각역을 미리 그려 보면 설계의 한계가 먼저 드러난다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, PRM + "the_rejection_region_first.png")


# ════════════════════════════════════════════════════════════════════
# 24. resampling_shoe_sales — 「효과가 없다」와 「판별할 수 없다」
#
# 쪽에 그림이 하나뿐이고(주별 전후 연결선과 두 귀무분포) 그것은
# 짝 구조의 이득을 그린다. 들머리가 세우는 것은 **세 수를 함께 읽는
# 일**이다 — p = 0.153, +9.5%, 구간 [-1.83, 5.92], 그리고 검정력
# 0.232. 「상한 5.92 는 상당한 개선의 가능성도 배제하지 않는다.」
#
# 그 셋을 한 그림에 올린다. 왼쪽은 구간 하나를 사업적으로 뜻이 있는
# 크기와 맞대고, 오른쪽은 열두 주의 검정력이 어디까지 닿는지 본다.
# 결론이 「효과가 없다」가 아니라는 것이 눈으로 보여야 한다.
# ════════════════════════════════════════════════════════════════════
# 쪽의 보기들이 쓰는 열두 주 그대로
BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23], float)
AFTER = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27], float)
r24 = np.random.default_rng(17_024)
n24 = len(BEFORE)
obs24 = AFTER.mean() - BEFORE.mean()
pool24 = np.r_[BEFORE, AFTER]
sh24 = pool24[np.argsort(r24.random((40000, 2 * n24)), axis=1)]
pd24 = sh24[:, n24:].mean(1) - sh24[:, :n24].mean(1)
# 쪽은 「올랐는가」만 묻는 단측이다 (정확 단측 0.154287 · 양측 0.308574)
p24 = (pd24 >= obs24).mean()
bd = (AFTER[r24.integers(0, n24, (40000, n24))].mean(1)
      - BEFORE[r24.integers(0, n24, (40000, n24))].mean(1))
lo24, hi24 = np.percentile(bd, [2.5, 97.5])
print(f"\n[shoe_sales] 전 {BEFORE.mean():.2f} -> 후 {AFTER.mean():.2f} "
      f"(+{obs24:.2f} 켤레 = {obs24 / BEFORE.mean() * 100:.1f}%)")
print(f"  순열 p = {p24:.4f} · 부트스트랩 구간 "
      f"[{lo24:.2f}, {hi24:.2f}]")
DEL = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
sd_pool = np.sqrt((BEFORE.var(ddof=1) + AFTER.var(ddof=1)) / 2)
pw24 = []
for d in DEL:
    se = sd_pool * np.sqrt(2.0 / n24)
    nc = d / se
    pw24.append(float(stats.nct.sf(stats.t.ppf(0.95, 2 * n24 - 2),
                                   2 * n24 - 2, nc)))
    print(f"  참 차이 {d:.1f} 켤레일 때 열두 주의 검정력 {pw24[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4),
                               gridspec_kw={"width_ratios": [1.15, 1]})
ax1.axvspan(lo24, hi24, color="tab:blue", alpha=0.13)
ax1.axvspan(2.0, hi24, color="tab:green", alpha=0.16)
ax1.plot([lo24, hi24], [0, 0], "-", color="tab:blue", lw=6.0)
ax1.plot([obs24], [0], "o", color="tab:blue", ms=16, zorder=5)
ax1.axvline(0, color="0.2", lw=2.4, ls="--")
ax1.text(0 - 0.12, 0.88, "차이가 없다", ha="right", va="top", fontsize=12,
         color="0.2", fontweight="bold")
ax1.text(obs24, -0.16, f"관측 +{obs24:.2f}", ha="center", va="top",
         fontsize=12.5, color="tab:blue", fontweight="bold")
ax1.text(lo24, 0.14, f"{lo24:.2f}", ha="center", va="bottom", fontsize=12,
         color="tab:blue", fontweight="bold")
ax1.text(hi24, 0.14, f"+{hi24:.2f}", ha="center", va="bottom", fontsize=12,
         color="tab:blue", fontweight="bold")
ax1.text((2.0 + hi24) / 2, 0.88,
         "사업적으로 뜻이 있는 크기\n구간이 이 영역도 배제하지 못한다",
         ha="center", va="top", fontsize=11.5, color="darkgreen",
         fontweight="bold")
ax1.text(lo24 - 0.1, -0.72, "95% 부트스트랩 구간", ha="left", va="bottom",
         fontsize=11.5, color="tab:blue")
ax1.set_xlim(lo24 - 0.6, hi24 + 0.6)
ax1.set_ylim(-1.0, 1.0)
ax1.set_yticks([])
ax1.set_xlabel("주당 판매량의 참 차이 (켤레)", fontsize=12)
ax1.grid(axis="x", alpha=0.25)
ax1.set_title(f"단측 순열 p = {p24:.3f} 인데 구간의 상한은 +{hi24:.2f} 다",
              fontsize=12.5)

ax2.plot(DEL, pw24, "o-", color="tab:red", lw=3.2, ms=12)
ax2.axhline(0.8, color="0.25", lw=2.2, ls="--")
ax2.text(DEL[-1], 0.815, "흔히 거는 0.8", ha="right", va="bottom",
         fontsize=11.5, color="0.25", fontweight="bold")
i2 = DEL.index(2.0)
ax2.scatter([2.0], [pw24[i2]], s=320, facecolor="none", edgecolor="tab:red",
            lw=3.0, zorder=5)
ax2.annotate(f"참 차이가 2 켤레라면\n열두 주의 검정력이 {pw24[i2]:.3f} 다",
             xy=(2.0, pw24[i2]), xytext=(0.55, 0.60), fontsize=12,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax2.set_xlabel("참 차이 (켤레)", fontsize=12)
ax2.set_ylabel("잡아낼 확률", fontsize=12)
ax2.set_ylim(0, 1.02)
ax2.grid(alpha=0.25)
ax2.set_title("열두 주로 잡을 수 있는 크기 (단측)", fontsize=12.5)

fig.suptitle("「효과가 없다」가 아니라 「열두 주로는 판별할 수 없다」다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, PRM + "cannot_tell_is_not_no_effect.png")


# ════════════════════════════════════════════════════════════════════
# 25. bootstrap_testing/correlation — 세 가지 뽑기가 내는 세 분포
#
# 쪽에 그림이 하나뿐이고(r 척도와 arctanh 척도의 부트스트랩 분포)
# 그것은 변환을 그린다. 들머리의 **셋째 문단**이 세 길을 가르는데
# 그려져 있지 않다 — 「섞으면 짝 구조가 통째로 깨지므로 「상관이
# 0 이다」만 물을 수 있다. … 이때 쌍을 깨고 x 와 y 를 따로 뽑으면
# 의존 구조가 사라져 구간이 아니라 귀무분포가 나온다.」
#
# 셋을 같은 자료에서 돌려 겹쳐 그리면 마지막 함정이 눈으로 보인다 —
# 「따로 뽑기」가 「섞기」와 **같은 자리에** 서 있다. 구간을 만들려고
# 한 계산이 귀무분포를 내놓은 것이다.
# ════════════════════════════════════════════════════════════════════
# 쪽은 자료를 싣지 않고 r = 0.47 만 적는다. 그 값을 내는 스물다섯 쌍을
# 찾아 썼더니 쪽의 p = 0.018 과 BCa 구간 [0.11, 0.72] 까지 맞는다
r25 = np.random.default_rng(17_025)
N25, B25 = 25, 40000
gen25 = np.random.default_rng(145)
z = gen25.standard_normal(N25)
hours = 10 + 3 * z + gen25.standard_normal(N25) * 1.2
score = 60 + 6 * z + gen25.standard_normal(N25) * 5.2
r_obs = float(np.corrcoef(hours, score)[0, 1])


def corr_rows(a, b):
    ac = a - a.mean(-1, keepdims=True)
    bc = b - b.mean(-1, keepdims=True)
    return ((ac * bc).sum(-1)
            / np.sqrt((ac ** 2).sum(-1) * (bc ** 2).sum(-1)))


perm_i = np.argsort(r25.random((B25, N25)), axis=1)
d_shuffle = corr_rows(hours[perm_i], score)
pair_i = r25.integers(0, N25, (B25, N25))
d_pair = corr_rows(hours[pair_i], score[pair_i])
d_split = corr_rows(hours[r25.integers(0, N25, (B25, N25))],
                    score[r25.integers(0, N25, (B25, N25))])
p25 = float((np.abs(d_shuffle) >= abs(r_obs)).mean())
lo25, hi25 = np.percentile(d_pair, [2.5, 97.5])
lo_s, hi_s = np.percentile(d_split, [2.5, 97.5])
print(f"\n[corr] 관측 상관 {r_obs:.4f} (n = {N25})")
print(f"  섞기        가운데 {np.median(d_shuffle):+.4f} · "
      f"p = {p25:.4f}")
print(f"  쌍째 뽑기   가운데 {np.median(d_pair):+.4f} · "
      f"95% 구간 [{lo25:.4f}, {hi25:.4f}]")
print(f"  따로 뽑기   가운데 {np.median(d_split):+.4f} · "
      f"95% 구간 [{lo_s:.4f}, {hi_s:.4f}] — 관측값을 품지 않는다")

fig, ax = plt.subplots(figsize=(12.4, 6.0))
bins25 = np.linspace(-1, 1, 160)
ax.hist(d_shuffle, bins=bins25, density=True, color="0.72",
        label="한쪽을 섞기 — 「상관이 0 이다」의 귀무분포")
ax.hist(d_pair, bins=bins25, density=True, histtype="step",
        color="tab:blue", lw=3.2,
        label="쌍을 통째로 다시 뽑기 — 구간이 나온다")
ax.hist(d_split, bins=bins25, density=True, histtype="step",
        color="tab:red", lw=3.2, ls="--",
        label="쌍을 깨고 x 와 y 를 따로 뽑기 — 구간이 아니다")
ax.axvline(r_obs, color="0.1", lw=3.0)
ax.text(r_obs - 0.02, ax.get_ylim()[1] * 0.42, f"관측 {r_obs:.3f}",
        ha="right", va="center", fontsize=12.5, color="0.1",
        fontweight="bold", rotation=90)
ax.axvline(0, color="0.45", lw=1.6, ls=":")
ax.annotate("따로 뽑은 것이 섞은 것과 같은 자리에 선다 —\n"
            "구간을 만들려던 계산이 귀무분포를 내놓았다",
            xy=(-0.14, 1.75), xytext=(-0.97, ax.get_ylim()[1] * 0.46),
            fontsize=12, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.plot([lo25, hi25], [-0.16, -0.16], "-", color="tab:blue", lw=5.0,
        clip_on=False)
ax.text((lo25 + hi25) / 2, -0.42, f"[{lo25:.2f}, {hi25:.2f}]",
        ha="center", va="top", fontsize=12, color="tab:blue",
        fontweight="bold")
ax.set_xlabel("상관계수", fontsize=12)
ax.set_ylabel("밀도", fontsize=12)
ax.set_xlim(-1, 1)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
ax.set_title(f"학생 {N25} 명의 같은 자료를 세 가지로 다시 뽑았다",
             fontsize=13)
fig.suptitle("무엇을 뽑는지가 무엇을 묻는지를 정한다", fontsize=15, y=0.99)
plt.tight_layout()
save(fig, TEST + "three_ways_to_resample_a_pair.png")
