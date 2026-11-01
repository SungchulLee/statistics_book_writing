"""17장 들어가는 이야기 그림 — 셋째 묶음 여섯.

  15. bootstrap_vs_permutation  두 결론이 갈릴 때의 방향
  16. convergence               무엇을 재는가가 횟수를 정한다
  17. limitations               무너지는 자리 넷을 한 자에 대고
  18. foundations               부등호 하나가 들고 나가는 몫
  19. permutation_tests         손으로 셀 수 있는 스무 가지
  20. two_sample                보수적인 검정이 내다 버린 검정력

만드는 파일:
  docs/ch17/comparison/img/which_way_they_split.png
  docs/ch17/comparison/img/what_you_measure_sets_the_count.png
  docs/ch17/comparison/img/four_places_it_breaks.png
  docs/ch17/permutation/img/one_inequality_carries_mass.png
  docs/ch17/permutation/img/twenty_arrangements_by_hand.png
  docs/ch17/permutation/img/conservative_throws_power_away.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import itertools

import numpy as np
from scipy import stats
from scipy.signal import lfilter

CMP = "docs/ch17/comparison/img/"
PRM = "docs/ch17/permutation/img/"


def shuffles(pool, m, rng):
    """m 벌의 순열을 한꺼번에 — 파이썬 반복문으로 뽑으면 백만 번이 든다."""
    return pool[np.argsort(rng.random((m, len(pool))), axis=1)]


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 15. bootstrap_vs_permutation — 두 결론이 갈릴 때의 방향
#
# 쪽의 two_questions 가 두 재표본의 생김새와 그것이 만드는 두 분포를
# 그려 들머리의 첫·둘째 문단을 담고, small_sample_gap 이 n = 5 의 정확
# 순열분포와 세 구간을 그린다. 그려지지 않은 것은 **넷째 문단의 물음**
# 이다 — 「두 결론이 갈릴 때는 어느 쪽으로 갈리는가.」
#
# 같은 자료에 둘을 함께 걸어 네 칸으로 세면 그 방향이 수로 나온다.
# 갈림이 대칭이 아니라는 것이 요점이다.
# ════════════════════════════════════════════════════════════════════
r15 = np.random.default_rng(17_015)
N15, REPS15, NP15, B15 = 12, 3000, 2000, 2000
cell = np.zeros((2, 2), int)      # [순열 기각?][구간이 0 을 뺐나?]
for _ in range(REPS15):
    a = r15.exponential(1.0, N15)          # 치우친 자료 — 갈림이 잘 난다
    b = r15.exponential(1.0, N15) + 0.55   # 참 차이가 있다
    obs = a.mean() - b.mean()
    pool = np.r_[a, b]
    sh = shuffles(pool, NP15, r15)
    perm = sh[:, :N15].mean(1) - sh[:, N15:].mean(1)
    p = (np.abs(perm) >= abs(obs)).mean()
    da = a[r15.integers(0, N15, (B15, N15))].mean(1)
    db = b[r15.integers(0, N15, (B15, N15))].mean(1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    cell[int(p < 0.05), int(not (lo <= 0 <= hi))] += 1
tot = cell.sum()
only_perm = cell[1, 0]
only_boot = cell[0, 1]
print(f"[bootstrap_vs_permutation] 치우친 자료 {tot:,} 벌 "
      f"(n = {N15} 씩, 참 차이 0.55)")
print(f"  둘 다 유의      {cell[1, 1] / tot:.4f}")
print(f"  둘 다 아니다    {cell[0, 0] / tot:.4f}")
print(f"  순열만 유의     {only_perm / tot:.4f}")
print(f"  구간만 0 을 뺌  {only_boot / tot:.4f}")
print(f"  갈린 자료 가운데 순열 쪽이 더 센 비율 "
      f"{only_perm / max(only_perm + only_boot, 1):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                               gridspec_kw={"width_ratios": [1.1, 1]})
g = cell / tot
im = ax1.imshow(g, origin="lower", cmap="Blues", vmin=0, vmax=g.max())
lab = [["둘 다 아니다", "구간만 0 을 뺐다"],
       ["순열만 유의하다", "둘 다 그렇다"]]
for i in range(2):
    for j in range(2):
        ax1.text(j, i + 0.12, f"{g[i, j]:.3f}", ha="center", va="center",
                 fontsize=17, fontweight="bold",
                 color="white" if g[i, j] > 0.55 * g.max() else "0.15")
        ax1.text(j, i - 0.22, lab[i][j], ha="center", va="center",
                 fontsize=11.5,
                 color="white" if g[i, j] > 0.55 * g.max() else "0.3")
ax1.set_xticks([0, 1], ["0 을 품었다", "0 을 뺐다"], fontsize=12.5)
ax1.set_yticks([0, 1], ["기각 못 함", "기각함"], fontsize=12.5)
ax1.set_xlabel("붓스트랩 95% 구간", fontsize=12)
ax1.set_ylabel("순열검정 (0.05)", fontsize=12)
ax1.set_title("같은 자료에 둘을 함께 걸면", fontsize=13)

ax2.bar([0], [only_perm / tot], color="tab:red", width=0.52,
        label="순열만 유의하다")
ax2.bar([1], [only_boot / tot], color="tab:blue", width=0.52,
        label="구간만 0 을 뺐다")
for i, v in enumerate([only_perm / tot, only_boot / tot]):
    ax2.text(i, v + 0.0015, f"{v:.3f}\n({int(v * tot):,} 벌)", ha="center",
             va="bottom", fontsize=12.5, fontweight="bold")
ax2.set_xticks([0, 1], ["순열만", "구간만"], fontsize=13)
ax2.set_ylabel("자료 전체에서의 비율", fontsize=12)
ax2.set_ylim(0, max(only_perm, only_boot) / tot * 1.5)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("갈림은 대칭이 아니다", fontsize=13)
ax2.annotate("치우친 자료에서는 구간 쪽이\n먼저 0 을 뺀다",
             xy=(0.72, only_boot / tot), xytext=(-0.46, max(
                 only_perm, only_boot) / tot * 1.14), fontsize=11.5,
             color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))

fig.suptitle("두 답이 갈릴 때 어느 쪽으로 갈리는지는 정해져 있다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, CMP + "which_way_they_split.png")


# ════════════════════════════════════════════════════════════════════
# 16. convergence — 무엇을 재는가가 횟수를 정한다
#
# 쪽의 monte_carlo_convergence 가 누적 추정값의 안정화와 두 양의
# 몬테카를로 변동을 그려 들머리의 첫·둘째 문단을 담는다. 그려지지
# 않은 것은 **셋째 문단**이고 그것이 수를 셋 들고 있다 — 「95% 구간의
# 하한은 표준오차보다 늘 3.8 배 더 흔들린다. 같은 정밀도를 얻으려면
# 그 제곱인 14 배의 되풀이가 필요하다」, 그리고 「0.05 근처에서 천
# 번만 돌리면 추정값이 [0.036, 0.064] 사이를 오가며 결론이 뒤집힌다.」
#
# 한 눈금으로 넷을 세우려 해 보았는데 그러면 안 된다 — 표준오차와
# 분위수는 **자료의 단위**로 재고 p 값은 **확률의 단위**로 재므로,
# 상대 정밀도 하나로 묶으면 구간의 하한이 오히려 싸게 나와 들머리의
# 3.8 배가 뒤집힌다. 그래서 칸을 둘로 나누어 각자의 눈금을 쓴다.
# ════════════════════════════════════════════════════════════════════
r16 = np.random.default_rng(17_016)
N16, REPS16 = 40, 600
x16 = r16.exponential(1.0, N16)
B_REF = 1000
GOAL = 0.002          # 자료 단위에서의 목표 — 보고하는 수가 ±0.002 안에서


def wobble(fn, B, reps=REPS16):
    """같은 자료에서 B 번씩 재표집을 되풀이해 추정값 자체의 흔들림을 잰다."""
    v = np.array([fn(x16[r16.integers(0, N16, (B, N16))])
                  for _ in range(reps)])
    return float(v.std(ddof=1)), float(v.mean())


TASKS = [
    ("표준오차", lambda s: s.mean(1).std(ddof=1), "tab:blue"),
    ("95% 구간의 하한", lambda s: np.percentile(s.mean(1), 2.5),
     "tab:green"),
    ("95% 구간의 상한", lambda s: np.percentile(s.mean(1), 97.5),
     "tab:olive"),
]
sds, need, cols, names = [], [], [], []
print("\n[convergence] 자료 단위의 양 — 같은 절대 정밀도에 드는 횟수")
for nm, fn, c in TASKS:
    sd, mean = wobble(fn, B_REF)
    b_need = B_REF * (sd / GOAL) ** 2
    sds.append(sd)
    need.append(b_need)
    cols.append(c)
    names.append(nm)
    print(f"  {nm:<16s} 값 {mean:.4f} · B = {B_REF:,} 에서 흔들림 "
          f"{sd:.5f}  ->  ±{GOAL} 에 {b_need:,.0f} 번")
print(f"  구간의 하한이 표준오차보다 {sds[1] / sds[0]:.2f} 배 더 흔들리고, "
      f"그 제곱인 {need[1] / need[0]:.1f} 배의 되풀이가 든다")

# 오른쪽 칸 — p 값은 확률의 단위다. 참 p 가 0.05 일 때 B 가 내놓는 폭
P_TRUE = 0.05
BS16 = [200, 1000, 5000, 20000, 100000]
bands = []
print(f"  참 p 가 {P_TRUE} 일 때 B 가 내놓는 95% 폭")
for B in BS16:
    se = np.sqrt(P_TRUE * (1 - P_TRUE) / B)
    lo, hi = P_TRUE - 1.959964 * se, P_TRUE + 1.959964 * se
    bands.append((lo, hi))
    print(f"    B = {B:7,d}:  [{lo:.4f}, {hi:.4f}]  (폭 {hi - lo:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))
ax1.barh(range(len(need)), need, color=cols, height=0.5)
for i, v in enumerate(need):
    ax1.text(v * 1.05, i, f"{v:,.0f} 번", va="center", ha="left",
             fontsize=12.5, fontweight="bold", color=cols[i])
ax1.set_yticks(range(len(need)), names, fontsize=12.5)
ax1.invert_yaxis()
ax1.set_xlim(0, max(need) * 1.42)
ax1.set_xticks([0, 20000, 40000, 60000, 80000],
               ["0", "2 만", "4 만", "6 만", "8 만"], fontsize=11)
ax1.set_xlabel(f"보고하는 수를 ±{GOAL} 안에 묶는 데 드는 되풀이 횟수",
               fontsize=11.5)
ax1.grid(axis="x", alpha=0.25)
ax1.set_title(f"구간의 끝점이 표준오차보다 {need[1] / need[0]:.0f} 배 비싸다",
              fontsize=12.5)
ax1.annotate(f"흔들림이 {sds[1] / sds[0]:.1f} 배라서\n횟수는 그 제곱이 든다",
             xy=(need[1], 1), xytext=(max(need) * 0.42, 2.0),
             fontsize=11.5, color="tab:green", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.9))

for i, (B, (lo, hi)) in enumerate(zip(BS16, bands)):
    c = "tab:red" if lo < 0.05 < hi and (lo < 0.04 or hi > 0.06) else "0.45"
    ax2.plot([lo, hi], [i, i], "-", color=c, lw=6.0, solid_capstyle="butt")
    ax2.plot([P_TRUE], [i], "o", color="0.15", ms=9, zorder=5)
    ax2.text(hi + 0.0025, i, f"[{lo:.3f}, {hi:.3f}]", va="center",
             ha="left", fontsize=11.5, color=c, fontweight="bold")
ax2.axvline(0.05, color="0.15", lw=2.0, ls="--")
ax2.set_yticks(range(len(BS16)), [f"B = {B:,}" for B in BS16],
               fontsize=12.5)
ax2.invert_yaxis()
ax2.set_xlim(0.015, 0.108)
ax2.set_xlabel("참 p 가 0.05 일 때 되풀이가 내놓는 값", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)
ax2.set_title("p 값은 확률의 단위로 재야 한다", fontsize=12.5)
ax2.annotate("천 번이면 0.05 의 양쪽으로 넘나들어\n결론이 뒤집힌다",
             xy=(bands[1][1], 1.08), xytext=(0.0585, 2.6), fontsize=11,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("「많이 하면 좋다」가 아니라 「무엇을 재는가」가 횟수를 정한다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, CMP + "what_you_measure_sets_the_count.png")


# ════════════════════════════════════════════════════════════════════
# 17. limitations — 무너지는 자리 넷을 한 자에 대고
#
# 쪽의 dependence_breaks_bootstrap 이 AR(1) 의 네 칸을, m_out_of_n 이
# 최댓값의 셋을 그려 들머리의 첫 세 문단과 넷째 문단의 앞부분을 담는다.
# 들머리가 **마지막에 묻는 것**은 그려져 있지 않다 — 「그러면 재표집이
# 무너지는 자리는 **모두** 어디인가.」
#
# 쪽의 본론은 그 답을 여섯 줄 표로 준다. 들머리의 그림은 표가 아니라
# **한 자에 대고 나란히 세운 것**이어야 한다 — 네 자리를 포함확률
# 하나로 재어 0.95 선 아래 어디에 떨어지는지 보인다.
# ════════════════════════════════════════════════════════════════════
r17 = np.random.default_rng(17_017)
REPS17, B17 = 1200, 1200
cov = {}

# (가) 시간에 따라 이어진 자료 — AR(1)
PHI, N17 = 0.7, 200
sd_stat = 1 / np.sqrt(1 - PHI ** 2)
hit = 0
for _ in range(REPS17):
    e = r17.standard_normal(N17 + 200)
    y = lfilter([1.0], [1.0, -PHI], e)[200:]
    bs = y[r17.integers(0, N17, (B17, N17))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hit += lo <= 0 <= hi
cov["시간에 따라 이어진 자료\n(평균)"] = hit / REPS17

# (나) 분포의 끝 — 균일분포의 상한을 최댓값으로
N17b, TH = 50, 1.0
hit = 0
for _ in range(REPS17):
    y = r17.uniform(0, TH, N17b)
    bs = y[r17.integers(0, N17b, (B17, N17b))].max(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hit += lo <= TH <= hi
cov["분포의 끝\n(균일분포의 상한)"] = hit / REPS17

# (다) 관측이 몇 값뿐인 자료 — 중앙값
N17c = 30
hit = 0
for _ in range(REPS17):
    y = r17.integers(1, 6, N17c).astype(float)   # 다섯 단계 설문
    truth = 3.0
    bs = np.median(y[r17.integers(0, N17c, (B17, N17c))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hit += lo <= truth <= hi
cov["몇 값뿐인 자료\n(중앙값)"] = hit / REPS17
wid_disc = []
for _ in range(400):
    y = r17.integers(1, 6, N17c).astype(float)
    bs = np.median(y[r17.integers(0, N17c, (B17, N17c))], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    wid_disc.append(hi - lo)
wid_disc = float(np.mean(wid_disc))

# (라) 몇 사람에게 쏠린 합계 — 꼬리가 두꺼운 모집단의 평균
N17d, AL = 60, 1.3     # 파레토 꼬리지수 1.3 — 분산이 없다
truth_d = AL / (AL - 1)
hit = 0
for _ in range(REPS17):
    y = (1 + r17.pareto(AL, N17d))
    bs = y[r17.integers(0, N17d, (B17, N17d))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hit += lo <= truth_d <= hi
cov["분산이 없는 모집단\n(평균)"] = hit / REPS17

print("\n[limitations] 명목 95% 를 약속한 구간이 실제로 담은 비율")
for k, v in cov.items():
    print(f"  {k.replace(chr(10), ' '):<28s} {v:.3f}")

fig, ax = plt.subplots(figsize=(12.0, 5.8))
ks = list(cov)
vs = [cov[k] for k in ks]
cl = ["tab:red" if v < 0.90 else "tab:purple" for v in vs]
ax.bar(range(len(ks)), vs, color=cl, width=0.56)
ax.axhline(0.95, color="0.15", lw=2.6, ls="--")
ax.text(len(ks) - 0.45, 0.957, "약속한 0.95", ha="right", va="bottom",
        fontsize=12.5, color="0.15", fontweight="bold")
for i, v in enumerate(vs):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=cl[i])
    gap = 0.95 - v
    if v > 0.2:        # 막대 안에 글자가 들어갈 때만
        ax.text(i, v - 0.03,
                (f"모자란 양\n{gap:.3f}" if gap > 0
                 else f"넘치는 양\n{-gap:.3f}"),
                ha="center", va="top", fontsize=11, color="white",
                fontweight="bold")
    else:
        ax.text(i, 0.10, f"모자란 양 {gap:.3f}", ha="center", va="bottom",
                fontsize=11, color=cl[i], fontweight="bold")
ax.set_xticks(range(len(ks)), ks, fontsize=11.5)
ax.set_ylabel("명목 95% 구간이 참값을 담은 비율", fontsize=12)
ax.set_ylim(0, 1.06)
ax.grid(axis="y", alpha=0.25)
ax.annotate(f"이 자리만 반대로 어긋난다 —\n구간이 한 점으로 쪼그라들고"
            f"\n(평균 폭 {wid_disc:.2f})\n그 점이 참값이라 늘 「담는다」",
            xy=(2, vs[2] * 0.72), xytext=(0.30, 0.60), fontsize=11,
            color="tab:purple", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.9))
ax.set_title("보통의 붓스트랩을 그대로 걸었을 때", fontsize=13)
fig.suptitle("무너지는 자리는 넷이고, 세 곳은 과신하는 쪽으로 어긋난다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, CMP + "four_places_it_breaks.png")


# ════════════════════════════════════════════════════════════════════
# 18. foundations — 부등호 하나가 들고 나가는 몫
#
# 쪽의 label_shuffle 이 라벨 섞기와 그것이 만드는 정확 귀무분포를
# 통째로 그려 들머리의 첫·둘째 문단을 담는다. 그려지지 않은 것은
# **셋째 문단**이다 — 「±5.40 자리의 막대가 통째로 걸려 있어서, 그
# 자리를 세면 0.3256 이고 빼면 0.3068 로 0.019 가 사라진다.」
#
# label_shuffle 과 같은 분포를 또 그리지 않으려면 **꼬리만 확대**해
# 그 한 자리를 보이고, 그 몫이 표본이 작아질 때 어떻게 커지는지를
# 함께 재야 한다. 쪽의 자료(각 집단 열 개, 평균차 5.40)를 그대로 쓴다.
# ════════════════════════════════════════════════════════════════════
# 쪽의 보기 1 이 쓰는 스무 개 그대로 — 앞 열이 A, 뒤 열이 B 다
Z18 = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
                173, 181, 182, 170, 169, 177, 168, 183, 169, 164], float)
A18, B18 = Z18[:10], Z18[10:]
pool18 = np.r_[A18, B18]
n18 = len(A18)
obs18 = B18.mean() - A18.mean()   # 쪽과 같은 방향
tot18 = pool18.sum()
diffs = []
for c in itertools.combinations(range(2 * n18), n18):
    s = pool18[list(c)].sum()
    diffs.append(s / n18 - (tot18 - s) / n18)
diffs = np.array(diffs)
p_ge = (np.abs(diffs) >= abs(obs18) - 1e-9).mean()
p_gt = (np.abs(diffs) > abs(obs18) + 1e-9).mean()
on_edge = np.isclose(np.abs(diffs), abs(obs18)).sum()
print(f"\n[foundations] 평균차 {obs18:.2f} · 배치 {len(diffs):,} 가지")
print(f"  만큼 또는 그보다 (>=)  p = {p_ge:.4f}")
print(f"  그보다만 (>)           p = {p_gt:.4f}")
print(f"  경계에 걸린 배치 {on_edge:,} 가지 = {p_ge - p_gt:.4f}")

# 표본이 작아지면 그 몫이 커진다. 자료의 앞 m 쌍만 떼어 재면 관측값까지
# 함께 바뀌어 들쭉날쭉하므로(재어 보았다), 자료에 매이지 않는 양으로 센다 —
# 순위합에서 **되풀이해 자료를 모았을 때 경계에 걸리는 몫의 기댓값** sum w^2.
# 순위는 자료가 무엇이든 1..2m 이므로 m 마다 정확히 셀 수 있다.
MS = [3, 4, 5, 6, 7, 8, 9, 10, 12]
share = []
for m in MS:
    cnt = {}
    for c in itertools.combinations(range(1, 2 * m + 1), m):
        s = sum(c)
        cnt[s] = cnt.get(s, 0) + 1
    totc = sum(cnt.values())
    mid = m * (2 * m + 1) / 2
    # |S - 중심| 이 같은 두 자리가 한 쌍으로 걸린다
    absm = {}
    for s, c in cnt.items():
        absm[round(abs(s - mid), 6)] = absm.get(round(abs(s - mid), 6), 0) + c
    w = np.array(list(absm.values()), float) / totc
    share.append(float((w ** 2).sum()))
    print(f"  집단마다 {m:2d} 개: 경계에 걸리는 몫의 기댓값 {share[-1]:.4f} "
          f"(순위합이 가질 수 있는 값 {len(cnt)} 개)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})
edge = abs(obs18)
tail = diffs[diffs >= edge - 1.2]
bins = np.histogram_bin_edges(diffs, bins=140)
h, e = np.histogram(diffs, bins=bins)
h = h / h.sum()
ctr = (e[:-1] + e[1:]) / 2
keep = ctr >= edge - 1.6
is_edge = np.isclose(ctr, edge, atol=(e[1] - e[0]) / 1.9)
ax1.bar(ctr[keep & ~is_edge], h[keep & ~is_edge], width=e[1] - e[0],
        color="0.7")
ax1.bar(ctr[keep & is_edge], h[keep & is_edge], width=e[1] - e[0],
        color="tab:red")
ax1.axvline(edge, color="tab:red", lw=2.2, ls=":")
ax1.text(edge, ax1.get_ylim()[1] * 0.95, f"  관측값 {obs18:.2f}", ha="left",
         va="top", fontsize=12.5, color="tab:red", fontweight="bold")
ax1.annotate(f"이 한 자리가 {(p_ge - p_gt) / 2:.4f}\n(양쪽 합해 "
             f"{p_ge - p_gt:.4f})",
             xy=(edge, h[is_edge].max()),
             xytext=(edge + 0.5, ax1.get_ylim()[1] * 0.62), fontsize=12,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax1.set_xlim(edge - 1.7, 16.5)
ax1.set_xlabel("배치가 만드는 평균차 (오른쪽 꼬리만 확대했다)",
               fontsize=11.5)
ax1.set_ylabel("배치의 비율", fontsize=11.5)
ax1.set_title(f"배치 {len(diffs):,} 가지 가운데 경계에 {on_edge:,} 가지",
              fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)

ax2.plot(MS, share, "o-", color="tab:red", lw=3.0, ms=12)
ax2.set_xticks(MS, [str(m) for m in MS])
ax2.set_xlabel("집단마다의 관측 수 (순위로 셌다)", fontsize=11.5)
ax2.set_ylabel("경계에 걸리는 몫의 기댓값", fontsize=11.5)
ax2.set_ylim(0, max(share) * 1.2)
ax2.grid(alpha=0.25)
ax2.set_title("표본이 작아지면 그 몫이 커진다", fontsize=12.5)
ax2.annotate(f"셋씩이면 {share[0]:.3f} — p 를 통째로 가른다",
             xy=(MS[0], share[0]), xytext=(MS[1] + 0.1, max(share) * 1.05),
             fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("「만큼 또는 그보다」의 「만큼」이 p 값의 한 조각을 들고 있다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, PRM + "one_inequality_carries_mass.png")


# ════════════════════════════════════════════════════════════════════
# 19. permutation_tests — 손으로 셀 수 있는 스무 가지
#
# 쪽에 그림이 넷이지만 모두 **귀무분포 히스토그램**이거나 p 값 분포다.
# 들머리의 첫·둘째 문단이 세우는 것은 히스토그램이 아니다 — 「x = (1,
# 2, 3) 과 y = (4, 5, 6) … 이름표를 붙이는 방법이 20 가지뿐이고 관측된
# 배정만큼 극단인 것은 둘이어서 p = 2/20 = 0.1 이다」, 그리고 「쓸
# 함수를 믿으려면 먼저 답을 아는 자료로 검산해야 한다.」
#
# 검산은 **스무 가지를 다 적어 놓고 세는 것**이다. 그러니 분포가 아니라
# 배치 스무 개를 그린다. 완전히 갈라진 자료인데도 기각하지 못하는 것이
# 눈으로 보여야 한다.
# ════════════════════════════════════════════════════════════════════
V19 = np.array([1, 2, 3, 4, 5, 6], float)
obs19 = V19[:3].mean() - V19[3:].mean()
rows = []
for c in itertools.combinations(range(6), 3):
    xs = V19[list(c)]
    ys = V19[[i for i in range(6) if i not in c]]
    rows.append((c, xs.mean() - ys.mean()))
rows.sort(key=lambda t: t[1])
d19 = np.array([t[1] for t in rows])
p19 = (np.abs(d19) >= abs(obs19) - 1e-9).mean()
cnt19 = int((np.abs(d19) >= abs(obs19) - 1e-9).sum())
print(f"\n[permutation_tests] 배치 {len(rows)} 가지 · 관측 평균차 "
      f"{obs19:.1f}")
print(f"  만큼 극단인 배치 {cnt19} 가지 -> p = {cnt19}/{len(rows)} = "
      f"{p19:.4f}")
print(f"  귀무분포의 표준편차 {d19.std(ddof=0):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.2),
                               gridspec_kw={"width_ratios": [1.35, 1]})
for k, (c, d) in enumerate(rows):
    extreme = abs(d) >= abs(obs19) - 1e-9
    for v in range(6):
        inx = v in c
        ax1.scatter(v + 1, k, s=185,
                    marker="s" if inx else "o",
                    color="tab:red" if inx else "tab:blue",
                    edgecolor="0.25" if extreme else "none",
                    linewidth=2.0 if extreme else 0, zorder=3)
    ax1.text(7.0, k, f"{d:+.1f}", va="center", ha="left", fontsize=11,
             fontweight="bold" if extreme else "normal",
             color="0.1" if extreme else "0.45")
    if extreme:
        ax1.axhspan(k - 0.45, k + 0.45, color="0.9", zorder=0)
ax1.set_xticks(range(1, 7), [str(int(v)) for v in V19], fontsize=12.5)
ax1.set_yticks([])
ax1.set_xlim(0.4, 8.2)
ax1.set_ylim(-0.8, len(rows) - 0.2)
ax1.set_xlabel("여섯 수", fontsize=12)
ax1.set_ylabel(f"이름표를 붙이는 {len(rows)} 가지 (평균차 순)",
               fontsize=11.5)
ax1.set_title("빨간 네모가 x 집단 · 평균차는 오른쪽에", fontsize=12.5)
ax1.scatter([], [], s=150, marker="s", color="tab:red", label="x 집단")
ax1.scatter([], [], s=150, marker="o", color="tab:blue", label="y 집단")
ax1.legend(fontsize=11, loc="upper center", ncols=2,
           bbox_to_anchor=(0.5, -0.11), frameon=False)

# 히스토그램으로 그리면 칸이 붙어 이어진 분포처럼 읽힌다. 가질 수 있는
# 값이 열 개뿐이라는 것이 요점이므로 그 열 개를 막대로 세운다
vals19, cnt19_ = np.unique(np.round(d19, 6), return_counts=True)
bar_c = ["tab:red" if abs(v) >= abs(obs19) - 1e-9 else "0.72"
         for v in vals19]
ax2.bar(vals19, cnt19_, width=0.34, color=bar_c)
for v, c in zip(vals19, cnt19_):
    ax2.text(v, c + 0.07, str(int(c)), ha="center", va="bottom",
             fontsize=11, color="0.3")
for v in (obs19, -obs19):
    ax2.axvline(v, color="tab:red", lw=2.6, ls=":")
ax2.text(obs19 - 0.18, ax2.get_ylim()[1] * 0.55, f"관측 {obs19:+.0f}",
         ha="right", va="center", fontsize=12.5, color="tab:red",
         fontweight="bold", rotation=90)
ax2.set_xlabel("평균차", fontsize=12)
ax2.set_ylabel("배치의 개수", fontsize=12)
ax2.set_ylim(0, 3.9)
ax2.set_yticks([0, 1, 2, 3])
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"만큼 극단인 것이 {cnt19} 가지 — p = {cnt19}/{len(rows)} = "
              f"{p19:.1f}", fontsize=12.5)
ax2.annotate("완전히 갈라진 자료인데도\n0.05 에서 기각할 수 없다",
             xy=(obs19, 1.1), xytext=(-2.3, ax2.get_ylim()[1] * 0.80),
             fontsize=12, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))

fig.suptitle("세 개씩으로는 어떤 자료가 와도 기각할 수 없다 — "
             "스무 가지를 다 적어 보면 안다", fontsize=15, y=0.99)
plt.tight_layout()
save(fig, PRM + "twenty_arrangements_by_hand.png")


# ════════════════════════════════════════════════════════════════════
# 20. two_sample — 보수적인 검정이 내다 버린 검정력
#
# 쪽의 statistic_choice 가 평균차와 Welch t 를 각각 통계량으로 썼을
# 때의 **귀무분포와 결과**를 그려 들머리의 둘째 문단과 셋째 문단의
# 앞부분을 담는다. 그려지지 않은 것은 셋째 문단의 **결론**이다 —
# 「지나치게 보수적인 검정은 무해한 것이 아니라 검정력을 내다 버린
# 것이다.」 귀무분포 둘로는 그 「버린 양」이 보이지 않는다.
#
# 버린 양은 **검정력 곡선 사이의 간격**이다. 쪽의 배치를 그대로 쓴다 —
# 작은 쪽이 스물에 산포 1, 큰 쪽이 쉰에 산포 3.
# ════════════════════════════════════════════════════════════════════
r20 = np.random.default_rng(17_020)
N20A, N20B, S20A, S20B = 20, 50, 1.0, 3.0
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
REPS20, NP20 = 1200, 600
pw_raw, pw_std = [], []
for d in DELTAS:
    h_raw = h_std = 0
    for _ in range(REPS20):
        a = r20.standard_normal(N20A) * S20A + d
        b = r20.standard_normal(N20B) * S20B
        pool = np.r_[a, b]
        o_raw = abs(a.mean() - b.mean())
        o_std = o_raw / np.sqrt(a.var(ddof=1) / N20A + b.var(ddof=1) / N20B)
        sh = shuffles(pool, NP20, r20)
        pa, pb = sh[:, :N20A], sh[:, N20A:]
        dm = np.abs(pa.mean(1) - pb.mean(1))
        ds = dm / np.sqrt(pa.var(ddof=1, axis=1) / N20A
                          + pb.var(ddof=1, axis=1) / N20B)
        h_raw += (dm >= o_raw).mean() < 0.05
        h_std += (ds >= o_std).mean() < 0.05
    pw_raw.append(h_raw / REPS20)
    pw_std.append(h_std / REPS20)
    print(f"[two_sample] 차이 {d:.1f}: 평균차 {h_raw / REPS20:.3f} · "
          f"표준오차로 나눈 꼴 {h_std / REPS20:.3f}")

fig, ax = plt.subplots(figsize=(11.8, 6.0))
ax.plot(DELTAS, pw_std, "o-", color="tab:blue", lw=3.4, ms=13,
        label="각 순열의 표준오차로 나눈 꼴")
ax.plot(DELTAS, pw_raw, "s-", color="tab:red", lw=3.4, ms=12,
        label="평균차를 그대로")
ax.fill_between(DELTAS, pw_raw, pw_std, color="tab:blue", alpha=0.14)
ax.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax.text(DELTAS[-1], 0.062, "명목 0.05", ha="right", va="bottom",
        fontsize=11.5, color="0.3", fontweight="bold")
imax = int(np.argmax(np.array(pw_std) - np.array(pw_raw)))
ax.annotate(f"내다 버린 검정력 — 여기서 "
            f"{pw_std[imax] - pw_raw[imax]:.3f}",
            xy=(DELTAS[imax], (pw_raw[imax] + pw_std[imax]) / 2),
            xytext=(1.62, 0.40), fontsize=12.5, color="tab:blue",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.annotate(f"차이가 없을 때도 "
            f"{pw_raw[0]:.3f} 밖에 기각하지 않는다 — 명목의 "
            f"{0.05 / max(pw_raw[0], 1e-4):.0f} 분의 1",
            xy=(0.0, pw_raw[0]), xytext=(0.06, 0.86), fontsize=11.5,
            color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.set_xlabel("두 집단 평균의 참 차이", fontsize=12)
ax.set_ylabel("기각하는 비율", fontsize=12)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
ax.legend(fontsize=12, loc="center right")
ax.set_title(f"작은 쪽 {N20A} 명 산포 {S20A:.0f} · 큰 쪽 {N20B} 명 산포 "
             f"{S20B:.0f}", fontsize=13)
fig.suptitle("보수적인 검정은 무해한 것이 아니라 검정력을 내다 버린 것이다",
             fontsize=15, y=0.99)
plt.tight_layout()
save(fig, PRM + "conservative_throws_power_away.png")
