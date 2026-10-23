r"""같은 것을 둘로 갈라 견주면 — 씨앗만 바꾼 열 번의 실행.

이 쪽은 $\mu_1 - \mu_2$ 의 구간을 세운다. 두 무리가 **정말 다른 것**일 때의 이야기다.
핸더슨 등(2018)은 바꾼 것이 난수 씨앗뿐인 실행 열 개를 다섯씩 둘로 갈라 견주었고
$p = 0.0016$ 을 얻었다. 이 그림은 그 일이 어떻게 일어나는지를 재는 것이다.

① 열 개를 다섯씩 가르는 방법은 $126$ 가지다. 그 가운데 **가장 갈라 보이는 쪽**을
   고르면 차이가 얼마나 커 보이는가.
② 「좋은 것 셋만 골라 보고하기」가 평균을 얼마나 올리는가.
③ 참 차이를 씨앗 잡음에서 갈라내려면 실행이 몇 번 필요한가.

실행 사이의 흩어짐은 모의로 만든 것이며 핸더슨 등의 수치가 아니다.

만드는 파일: docs/ch08/two_sample_intervals/img/seed_split.png
"""

import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.05
NRUN = 10
CV = 0.40          # 실행 사이 변동계수 — 강화학습에서 드물지 않은 크기
REP = 20_000


def two_sample_p(a, b):
    """합동 t 의 양측 p 값."""
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    t = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / na + 1 / nb))
    return t, 2 * stats.t.sf(abs(t), na + nb - 2)


# === ① 126 가지 가름 =================================================
SPLITS = [s for s in itertools.combinations(range(NRUN), NRUN // 2)
          if 0 in s]          # 대칭이므로 절반만 — 126 가지
print(f"① 실행 {NRUN} 개를 {NRUN // 2} 씩 가르는 방법 = {len(SPLITS)} 가지")

best_p, any_sig, best_gap = [], [], []
for _ in range(REP):
    x = RNG.normal(1.0, CV, NRUN)
    ps, gaps = [], []
    for s in SPLITS:
        m = np.zeros(NRUN, bool)
        m[list(s)] = True
        t, p = two_sample_p(x[m], x[~m])
        ps.append(p)
        gaps.append(abs(x[m].mean() - x[~m].mean()) / x.mean())
    ps = np.array(ps)
    best_p.append(ps.min())
    any_sig.append(bool((ps < ALPHA).any()))
    best_gap.append(max(gaps))

best_p = np.array(best_p)
print(f"   한 번 가른 것의 1종오류는 이름값 {ALPHA} 다 (바꾼 것이 없으므로).")
print(f"   그런데 {len(SPLITS)} 가지를 모두 보고 **가장 작은 p 를 고르면**:")
print(f"     p < 0.05 인 가름이 하나라도 있을 확률 = {np.mean(any_sig):.3f}")
print(f"     가장 작은 p 의 중앙값 = {np.median(best_p):.5f}")
print(f"     가장 작은 p 가 0.01 보다 작을 확률 = {np.mean(best_p < 0.01):.3f}")
print(f"   가장 갈라 보이는 가름의 상대 차이 중앙값 = "
      f"{100 * np.median(best_gap):.1f}%")

# === ② 좋은 것 몇 개만 고르기 ========================================
TOPS = [1, 2, 3, 5, 10]
print(f"\n② 실행 {NRUN} 개 가운데 좋은 것 N 개만 골라 평균을 보고하면")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("      N    보고되는 평균   참 평균 대비")
TOPM = []
big = RNG.normal(1.0, CV, (REP, NRUN))
for k in TOPS:
    v = np.sort(big, axis=1)[:, -k:].mean(axis=1).mean()
    TOPM.append(v)
    print(f"   {k:>4}   {v:>12.4f}   {100 * (v - 1.0):>+10.1f}%")
print("   열 번 돌려 가장 좋은 하나를 적는 일은 평균을 "
      f"{100 * (TOPM[0] - 1):.0f}% 올려 적는 일이다.")

# === ③ 참 차이를 갈라내려면 ==========================================
NS = [3, 5, 10, 20, 50, 100, 200]
DELTAS = [0.05, 0.10, 0.20]
print(f"\n③ 참 차이 delta 를 잡아내는 검정력 (실행 사이 변동계수 {CV})")
print("       n" + "".join(f"   delta={d:<5}" for d in DELTAS))
POW = {d: [] for d in DELTAS}
for n in NS:
    row = []
    for d in DELTAS:
        a = RNG.normal(1.0 + d, CV, (REP, n))
        b = RNG.normal(1.0, CV, (REP, n))
        sp2 = ((n - 1) * a.var(1, ddof=1) + (n - 1) * b.var(1, ddof=1)) / (2 * n - 2)
        t = (a.mean(1) - b.mean(1)) / np.sqrt(sp2 * (2 / n))
        pw = float(np.mean(2 * stats.t.sf(np.abs(t), 2 * n - 2) < ALPHA))
        POW[d].append(pw)
        row.append(pw)
    print(f"   {n:>5}" + "".join(f"{v:>13.3f}" for v in row))
print("   5 번씩으로는 10% 차이도 거의 잡지 못한다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

ax1.hist(np.clip(best_p, 0, 0.1), bins=np.linspace(0, 0.1, 51), color=C_B,
         alpha=0.8, edgecolor="white", linewidth=0.4)
ax1.axvline(ALPHA, color=C_A, lw=2.4, ls="--")
ax1.set_xlim(0, 0.1)
ax1.text(ALPHA + 0.003, ax1.get_ylim()[1] * 0.92, f"${ALPHA}$",
         color=C_A, fontsize=11)
ax1.set_xlabel("$126$ 가지 가름 가운데 가장 작은 $p$ 값")
ax1.set_ylabel("빈도")
ax1.set_title(f"① 바꾼 것이 없는데도 — 가름을 고르면\n"
              f"$p < 0.05$ 가 나올 확률 ${np.mean(any_sig):.2f}$", fontsize=12)
ax1.grid(alpha=0.25)

xs = np.arange(len(TOPS))
bars = ax2.bar(xs, TOPM, color=[C_A, C_A, C_A, "#ff9896", C_C], alpha=0.85)
ax2.axhline(1.0, color="#333333", lw=2.0, ls="--")
ax2.text(0.42, 1.004, "참 평균", fontsize=10.5, color="#333333", ha="left")
for i, v in enumerate(TOPM):
    ax2.text(xs[i], v + 0.006, f"${v:.3f}$", ha="center", fontsize=10.5)
ax2.set_xticks(xs)
ax2.set_xticklabels([f"상위 ${k}$" if k < NRUN else f"전체 ${k}$" for k in TOPS],
                    fontsize=10.5)
ax2.set_ylabel("보고되는 평균")
ax2.set_ylim(0.97, max(TOPM) * 1.05)
ax2.set_title("② 좋은 것만 골라 적으면", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

for i, d in enumerate(DELTAS):
    ax3.semilogx(NS, POW[d], "o-", color=["#9ecae1", "#4292c6", "#08519c"][i],
                 lw=2.6, ms=9, label=f"참 차이 ${100 * d:.0f}\\%$")
ax3.axhline(0.8, color=C_A, lw=1.8, ls="--")
ax3.text(3.2, 0.82, "검정력 $0.8$", fontsize=10.5, color=C_A)
ax3.axvline(5, color="#333333", lw=1.6, ls=":")
ax3.text(3.15, 0.60, "흔히 쓰는\n실행 $5$ 번", fontsize=10, color="#333333")
ax3.set_xlabel("각 무리의 실행 횟수 $n$")
ax3.set_ylabel("검정력")
ax3.set_xticks([3, 10, 100])
ax3.set_xticklabels(["$3$", "$10$", "$10^{2}$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.03)
ax3.set_title("③ 참 차이를 잡아내려면", fontsize=12)
ax3.legend(loc="center right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/two_sample_intervals/img/seed_split.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
