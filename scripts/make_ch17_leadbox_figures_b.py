"""17장 들어가는 이야기 그림 — 둘째 그룹 일곱.

  8.  bca                      포함확률을 사는 값은 표본크기만큼의 계산
  9.  bootstrap_t              꼬리를 만드는 것은 작은 안쪽 표준오차다
  10. comparison               어려운 문제에서만 고를 일이 생긴다
  11. bootstrap_median         공식은 모르는 수를 묻고 재표집은 묻지 않는다
  12. visualization_confidence 95% 는 구간이 아니라 절차에 붙은 수다
  13. single_mean              한 줄을 빼먹으면 p 가 한가운데로 몰린다
  14. two_means                불균형과 이분산이 함께 와야 갈린다

만드는 파일:
  docs/ch17/bootstrap_ci/img/what_the_shift_costs.png
  docs/ch17/bootstrap_ci/img/tiny_se_makes_the_tail.png
  docs/ch17/bootstrap_ci/img/only_hard_problems_differ.png
  docs/ch17/bootstrap_ci/img/formula_needs_what_you_lack.png
  docs/ch17/bootstrap_ci/img/ninety_five_belongs_to_the_procedure.png
  docs/ch17/bootstrap_testing/img/p_piles_up_in_the_middle.png
  docs/ch17/bootstrap_testing/img/both_conditions_needed.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

CI = "docs/ch17/bootstrap_ci/img/"
TEST = "docs/ch17/bootstrap_testing/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def bca_interval(x, stat, B, rng, alpha=0.05):
    """편향과 편향을 자료에서 재어 읽는 자리를 옮긴다."""
    n = len(x)
    th = stat(x)
    bs = stat(x[rng.integers(0, n, (B, n))], axis=1)
    z0 = stats.norm.ppf(np.clip((bs < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([stat(np.delete(x, i)) for i in range(n)])
    d = jk.mean() - jk
    a = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-300)
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        adj = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(np.percentile(bs, 100 * np.clip(adj, 1e-4, 1 - 1e-4)))
    return out[0], out[1], np.percentile(bs, [2.5, 97.5])


# ════════════════════════════════════════════════════════════════════
# 8. bca — 포함확률을 사는 값
#
# 쪽의 bca_dial 이 「조정된 백분위수 수준과 그것이 만들어 내는 구간」을
# 그려 들머리의 둘째 문단을 담는다. **셋째 문단**은 그려져 있지 않다 —
# 「하한이 올라가면서 참값을 덮는 비율이 0.686 에서 0.756 으로 오른다.
# 그 대가로 관측값을 하나씩 빼 보는 셈을 표본크기만큼 더 돌려야 한다.」
# 사는 것과 치르는 것을 함께 그린다.
# ════════════════════════════════════════════════════════════════════
r8 = np.random.default_rng(17_108)
NS8 = [10, 20, 40, 80]
REPS8, B8 = 400, 1200
cov_pct, cov_bca = [], []
for n in NS8:
    hp = hb = 0
    for _ in range(REPS8):
        x = r8.exponential(1.0, n)
        lo_b, hi_b, (lo_p, hi_p) = bca_interval(
            x, lambda a, axis=None: np.var(a, ddof=1, axis=axis), B8, r8)
        hp += lo_p <= 1.0 <= hi_p
        hb += lo_b <= 1.0 <= hi_b
    cov_pct.append(hp / REPS8)
    cov_bca.append(hb / REPS8)
print("[bca] 지수 자료의 분산 — 명목 95%")
for n, a, b in zip(NS8, cov_pct, cov_bca):
    print(f"  n = {n:3d}:  분위수 그대로 {a:.3f} · 자리를 옮기면 {b:.3f}  "
          f"(+{b - a:.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS8, cov_pct, "s-", color="0.5", lw=3.0, ms=12,
         label="분위수를 그대로 읽기")
ax1.plot(NS8, cov_bca, "o-", color="tab:blue", lw=3.2, ms=13,
         label="읽는 자리를 옮기기")
ax1.axhline(0.95, color="0.2", lw=2.2, ls="--")
for i, n in enumerate(NS8):
    if n in (20, 40):
        ax1.annotate("", xy=(n, cov_bca[i]), xytext=(n, cov_pct[i]),
                     arrowprops=dict(arrowstyle="<->", color="0.3", lw=2.0))
        ax1.text(n * 1.08, (cov_bca[i] + cov_pct[i]) / 2,
                 f"+{cov_bca[i] - cov_pct[i]:.3f}", ha="left", va="center",
                 fontsize=11.5, color="0.2", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks(NS8, [str(n) for n in NS8])
ax1.set_xlabel("표본크기 n (로그 척도)", fontsize=11.5)
ax1.set_ylabel("참값을 담는 비율", fontsize=11.5)
ax1.set_ylim(min(cov_pct) - 0.04, 0.99)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25)
ax1.set_title("사는 것 — 포함확률", fontsize=12.5)

ns = np.array(NS8, float)
ax2.plot(ns, np.full_like(ns, 1.0), "s-", color="0.5", lw=3.0, ms=12,
         label="분위수를 그대로 읽기")
ax2.plot(ns, 1 + ns, "o-", color="tab:blue", lw=3.2,
         ms=13, label="읽는 자리를 옮기기 (잭나이프 n 번이 더 든다)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks(NS8, [str(n) for n in NS8])
ax2.set_yticks([1, 10, 100], ["1", "10", "100"])
ax2.set_xlabel("표본크기 n (로그 척도)", fontsize=11.5)
ax2.set_ylabel("통계량을 더 계산해야 하는 횟수 (상대)", fontsize=11.5)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(alpha=0.25, which="both")
ax2.set_title("치르는 것 — 관측값을 하나씩 빼 보는 셈", fontsize=12.5)
ax2.text(0.97, 0.06, "표본크기만큼 더 돌려야 한다 —\n"
         "n = 80 이면 여든 번이다", transform=ax2.transAxes, ha="right",
         va="bottom", fontsize=11.5, color="tab:blue", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="tab:blue"))

fig.suptitle("읽는 자리를 옮기면 구간이 따라간다 — 그 값은 계산이다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, CI + "what_the_shift_costs.png")


# ════════════════════════════════════════════════════════════════════
# 9. bootstrap_t — 꼬리를 만드는 것은 작은 안쪽 표준오차다
#
# 쪽의 studentized_pivot 이 세 분포를 겹쳐 그려 들머리의 첫·둘째 문단을
# 담는다. **넷째 문단**은 그려져 있지 않다 — 「재표본마다 표준오차를
# 다시 구해야 하고, 그 값이 우연히 0 에 가까워지면 비가 터져 분위수가
# 요동친다.」 터지는 자리를 눈으로 보이려면 비와 안쪽 표준오차를
# **함께** 찍어야 한다.
# ════════════════════════════════════════════════════════════════════
r9 = np.random.default_rng(17_109)
N9, B9 = 15, 20_000
x9 = r9.exponential(1.0, N9)
idx9 = r9.integers(0, N9, (B9, N9))
bx = x9[idx9]
mb = bx.mean(1)
sb = bx.std(1, ddof=1) / np.sqrt(N9)
tstar = (mb - x9.mean()) / sb
q025 = np.percentile(tstar, 2.5)
extreme = tstar <= q025
print(f"\n[bootstrap_t] n = {N9}, B = {B9:,}")
print(f"  t* 의 2.5 백분위수 {q025:.3f}")
print(f"  꼬리 쪽 재표본의 안쪽 표준오차 가운데값 "
      f"{np.median(sb[extreme]):.4f}")
print(f"  나머지의 가운데값 {np.median(sb[~extreme]):.4f}  "
      f"({np.median(sb[~extreme]) / np.median(sb[extreme]):.2f} 배)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(sb[~extreme][:6000], tstar[~extreme][:6000], "o", ms=2.6,
         color="0.6", alpha=0.35)
ax1.plot(sb[extreme], tstar[extreme], "o", ms=3.4, color="tab:red",
         alpha=0.6)
ax1.axhline(q025, color="tab:red", lw=2.2, ls="--")
ax1.text(sb.max() * 0.98, q025 + 0.25, f"아래 2.5% 자리 {q025:.2f}",
         ha="right", va="bottom", fontsize=11.5, color="tab:red",
         fontweight="bold")
ax1.set_xlabel("그 재표본 안에서 다시 구한 표준오차", fontsize=11.5)
ax1.set_ylabel("비 (재표본의 평균 빼기 원평균, 그 표준오차로 나눈 것)",
               fontsize=11)
ax1.set_ylim(np.percentile(tstar, 0.05), np.percentile(tstar, 99.5))
ax1.grid(alpha=0.25)
ax1.set_title("왼쪽 끝이 아래로 늘어져 있다", fontsize=12.5)
ax1.annotate("안쪽 표준오차가 작은 재표본이\n꼬리를 만든다",
             xy=(np.median(sb[extreme]), q025 - 1.0),
             xytext=(np.median(sb) * 1.15, np.percentile(tstar, 1.2)),
             fontsize=11.8, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

bins9 = np.linspace(0, sb.max(), 60)
ax2.hist(sb[~extreme], bins=bins9, density=True, alpha=0.6, color="0.55",
         label="나머지 재표본")
ax2.hist(sb[extreme], bins=bins9, density=True, alpha=0.65,
         color="tab:red", label="아래 2.5% 에 든 재표본")
ax2.axvline(np.median(sb[extreme]), color="tab:red", lw=2.6)
ax2.axvline(np.median(sb[~extreme]), color="0.3", lw=2.6, ls="--")
ax2.set_xlabel("그 재표본 안에서 다시 구한 표준오차", fontsize=11.5)
ax2.set_ylabel("비율", fontsize=11.5)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title(f"가운데값이 {np.median(sb[~extreme]) / np.median(sb[extreme]):.2f} 배 "
              f"다르다", fontsize=12.5)
ax2.text(0.97, 0.52, "그 값이 0 에 가까워지면\n비가 터지고 분위수가 요동친다",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11.3,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("비 자체를 재표집한다 — 그 분모가 약점이다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, CI + "tiny_se_makes_the_tail.png")


# ════════════════════════════════════════════════════════════════════
# 10. comparison — 어려운 문제에서만 고를 일이 생긴다
#
# 쪽의 coverage_vs_width 가 「네 검정의 크기와 폭」을 한 평면에 그려
# 들머리의 첫·둘째 문단을 담는다. **셋째 문단**은 그려져 있지 않다 —
# 「반대로 쉬운 문제에서는 고를 일이 아예 없다 — n = 100 의 평균에서는
# 다섯 방법의 포함확률이 0.934 와 0.947 사이에 다 모인다.」
# 같은 평면을 두 문제에서 그려야 그 대비가 보인다.
# ════════════════════════════════════════════════════════════════════
r10 = np.random.default_rng(17_110)
REPS10, B10 = 500, 1000


def three_intervals(x, stat, rng, B):
    n = len(x)
    th = stat(x)
    bs = stat(x[rng.integers(0, n, (B, n))], axis=1)
    pct = np.percentile(bs, [2.5, 97.5])
    basic = (2 * th - pct[1], 2 * th - pct[0])
    lo_b, hi_b, _ = bca_interval(x, stat, B, rng)
    return {"분위수": tuple(pct), "반사": basic, "자리 옮김": (lo_b, hi_b)}


PROBS = [("어려운 문제 (n = 15 · 분산)", 15,
          lambda a, axis=None: np.var(a, ddof=1, axis=axis), 1.0),
         ("쉬운 문제 (n = 100 · 평균)", 100,
          lambda a, axis=None: np.mean(a, axis=axis), 1.0)]
res10 = {}
for nm, n, stat, truth in PROBS:
    acc = {k: [0, []] for k in ("분위수", "반사", "자리 옮김")}
    for _ in range(REPS10):
        x = r10.exponential(1.0, n)
        for k, (lo, hi) in three_intervals(x, stat, r10, B10).items():
            acc[k][0] += lo <= truth <= hi
            acc[k][1].append(hi - lo)
    res10[nm] = {k: (v[0] / REPS10, float(np.mean(v[1])))
                 for k, v in acc.items()}
print("\n[comparison] 같은 평면을 두 문제에서")
for nm, d in res10.items():
    print(f"  {nm}")
    for k, (c, w_) in d.items():
        print(f"    {k:8s} 포함확률 {c:.3f} · 폭 {w_:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.0))
COLS10 = {"분위수": "0.45", "반사": "tab:orange", "자리 옮김": "tab:blue"}
for ax, (nm, d) in zip(axs, res10.items()):
    for k, (c, w_) in d.items():
        ax.plot([w_], [c], "o", ms=20, color=COLS10[k])
        ax.annotate(k, xy=(w_, c), xytext=(0, 19),
                    textcoords="offset points", ha="center", fontsize=11.5,
                    color=COLS10[k], fontweight="bold")
    ax.axhline(0.95, color="0.2", lw=2.2, ls="--")
    ws = [w_ for _, w_ in d.values()]
    cs = [c for c, _ in d.values()]
    pad = (max(ws) - min(ws)) * 0.55 + 1e-3
    ax.set_xlim(min(ws) - pad, max(ws) + pad)
    ax.set_ylim(min(min(cs) - 0.06, 0.90), 1.01)
    ax.set_xlabel("구간의 평균 폭", fontsize=11.5)
    ax.set_title(nm, fontsize=13)
    ax.grid(alpha=0.25)
axs[0].set_ylabel("참값을 담는 비율", fontsize=11.5)
axs[0].text(0.03, 0.06, "세 점이 흩어져 있다 —\n고를 일이 생긴다",
            transform=axs[0].transAxes, ha="left", va="bottom",
            fontsize=11.5, color="0.2", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))
axs[1].text(0.03, 0.06, "세 점이 한자리에 모인다 —\n고를 일이 없다",
            transform=axs[1].transAxes, ha="left", va="bottom",
            fontsize=11.5, color="0.2", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("방법을 고르는 일이 문제가 어려울 때만 생긴다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, CI + "only_hard_problems_differ.png")


# ════════════════════════════════════════════════════════════════════
# 11. bootstrap_median — 공식은 모르는 수를 묻는다
#
# 쪽의 median_bootstrap_atoms 가 계단을, median_welfare_line 이 선정선을
# 그린다. 들머리의 **둘째 문단**은 그려져 있지 않다 — 「점근 표준오차가
# 1/(2 f(m) 루트 n) 인데 거기 들어가는 f(m) 을 모르기 때문이다.
# 재표집은 그 값을 **묻지 않는다.**」
#
# 「모른다」를 말로만 두지 않고, f(m) 을 추정하는 손잡이(대역폭)를 돌려
# 공식의 답이 얼마나 움직이는지 재어 재표집과 맞댄다. 자료는 쪽의 것
# 그대로다 — LogNormal(10.5, 0.8^2) 에서 n = 200. 그 모집단의 점근
# 표준오차 2,574.7 은 쪽이 보기 5 에서 적어 둔 값이고, 표본중앙값
# 20,000 개의 실제 표준편차 2,570.9 와 맞는다.
# ════════════════════════════════════════════════════════════════════
r11 = np.random.default_rng(17_111)
MU11, SIG11, N11, REPS11, B11 = 10.5, 0.8, 200, 300, 1500
pop_med = np.exp(MU11)
f_pop = stats.lognorm.pdf(pop_med, SIG11, scale=np.exp(MU11))
true_se = 1 / (2 * f_pop * np.sqrt(N11))          # 모집단을 알 때의 참값
BWS = [0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0]
form_se = {bw: [] for bw in BWS}
boot_list = []
for _ in range(REPS11):
    x = r11.lognormal(MU11, SIG11, N11)
    m = np.median(x)
    sd = min(x.std(ddof=1), stats.iqr(x) / 1.349)   # 쓰이는 로버스트 척도
    for bw in BWS:
        h = bw * sd * N11 ** (-0.2)
        f = np.mean(stats.norm.pdf((x - m) / h)) / h
        form_se[bw].append(1 / (2 * f * np.sqrt(N11)))
    boot_list.append(np.median(x[r11.integers(0, N11, (B11, N11))],
                               axis=1).std(ddof=1))
print(f"\n[bootstrap_median] 모집단 중앙값 {pop_med:,.1f} 자리의 밀도 "
      f"{f_pop:.4e} · 점근 표준오차 {true_se:,.1f}")
for bw in BWS:
    v = np.array(form_se[bw])
    print(f"  대역폭 {bw:4.2f} 로 f(m) 을 재면 공식이 {v.mean():8.1f} "
          f"(표본마다의 흔들림 {v.std():6.1f})")
boot_arr = np.array(boot_list)
print(f"  재표집은 {boot_arr.mean():8.1f} (표본마다의 흔들림 "
      f"{boot_arr.std():6.1f}) — 대역폭을 고를 일이 없다")

fig, ax = plt.subplots(figsize=(11.6, 6.0))
means = [np.mean(form_se[bw]) for bw in BWS]
sds = [np.std(form_se[bw]) for bw in BWS]
ax.errorbar(range(len(BWS)), means, yerr=sds, fmt="s", ms=12,
            color="tab:red", ecolor="tab:red", elinewidth=2.4, capsize=8,
            capthick=2.2, label="공식 — f(m) 을 대역폭 하나로 추정해서")
ax.errorbar([len(BWS) + 0.7], [boot_arr.mean()], yerr=[boot_arr.std()],
            fmt="o", ms=16, color="tab:blue", ecolor="tab:blue",
            elinewidth=2.4, capsize=8, capthick=2.2,
            label="재표집 — 묻지 않는다")
ax.axhline(true_se, color="0.2", lw=2.4, ls="--")
ax.text(-0.45, true_se + 55, f"모집단을 알 때의 참값 {true_se:,.0f}",
        ha="left", va="bottom", fontsize=12, color="0.2",
        fontweight="bold")
ax.set_xticks(list(range(len(BWS))) + [len(BWS) + 0.7],
              [f"{bw:.2f}" for bw in BWS] + ["재표집"], fontsize=11.5)
ax.set_xlabel("f(m) 을 재는 대역폭 — 어느 값이 옳은지 자료는 말해 주지 않는다",
              fontsize=11.5)
ax.set_ylabel("중앙값의 표준오차 추정값", fontsize=12)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(alpha=0.25)
ax.annotate(f"좁게 잡으면 표본마다 {max(sds):.0f} 씩 흔들리고",
            xy=(0, means[0] + sds[0]), xytext=(0.75, max(means) * 1.16),
            fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.annotate(f"넓게 잡으면 참값보다 "
            f"{(max(means) / true_se - 1) * 100:.0f}% 높게 나온다",
            xy=(len(BWS) - 1, means[-1]), xytext=(2.6, max(means) * 1.06),
            fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.set_ylim(0, max(np.array(means) + np.array(sds)) * 1.3)
ax.set_title(f"소득꼴 자료 {N11} 개 · 표본을 바꿔 가며 {REPS11} 번",
             fontsize=13)
fig.suptitle("공식은 모르는 수를 하나 요구하고, 재표집은 요구하지 않는다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, CI + "formula_needs_what_you_lack.png")


# ════════════════════════════════════════════════════════════════════
# 12. visualization_confidence_levels — 절차에 붙은 수다
#
# 쪽에 그림이 하나뿐이고(신뢰수준에 따른 구간) 들머리가 **마지막 문단에서
# 대놓고 묻는다** — 「그 「되풀이했을 때」는 눈으로 어떻게 보이는가.」
# 그 물음에 그대로 답하는 그림이 없으므로 그것을 그린다.
# ════════════════════════════════════════════════════════════════════
r12 = np.random.default_rng(17_112)
N12, SHOW, REPS12, B12 = 20, 60, 2000, 800
SIG12 = 0.8
truth12 = np.exp(SIG12 ** 2 / 2)
los, his = [], []
for _ in range(REPS12):
    x = np.exp(r12.normal(0, SIG12, N12))
    bs = x[r12.integers(0, N12, (B12, N12))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    los.append(lo)
    his.append(hi)
los, his = np.array(los), np.array(his)
cover = (los <= truth12) & (truth12 <= his)
print(f"\n[visualization] 참 평균 {truth12:.4f} · n = {N12}")
print(f"  명목 95% 구간이 참값을 담은 비율 {cover.mean():.4f} "
      f"({cover.sum()}/{REPS12})")
print(f"  앞의 {SHOW} 개 가운데 놓친 것 {int((~cover[:SHOW]).sum())} 개")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4),
                               gridspec_kw={"width_ratios": [1.5, 1]})
for i in range(SHOW):
    c = "tab:blue" if cover[i] else "tab:red"
    ax1.plot([los[i], his[i]], [i, i], "-", color=c,
             lw=2.6 if cover[i] else 3.2, alpha=0.85)
ax1.axvline(truth12, color="0.15", lw=3.0)
ax1.text(truth12 + 0.02, SHOW * 0.985, "참 평균", ha="left", va="top",
         fontsize=12.5, color="0.15", fontweight="bold", rotation=90)
ax1.set_ylim(-1.5, SHOW + 0.5)
ax1.set_yticks([])
ax1.set_xlabel("평균", fontsize=11.5)
ax1.set_ylabel(f"표본 (앞의 {SHOW} 개만 그렸다)", fontsize=11.5)
ax1.set_title(f"구간 {SHOW} 개 가운데 빨간 것 "
              f"{int((~cover[:SHOW]).sum())} 개가 참값을 놓쳤다",
              fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)

ax2.bar([0], [cover.mean()], color="tab:blue", width=0.5)
ax2.bar([0], [1 - cover.mean()], bottom=[cover.mean()], color="tab:red",
        width=0.5)
ax2.axhline(0.95, color="0.15", lw=2.6, ls="--")
ax2.text(0.42, 0.955, "약속한 0.95", ha="right", va="bottom",
         fontsize=12, color="0.15", fontweight="bold")
ax2.text(0, cover.mean() / 2, f"담았다\n{cover.mean():.3f}", ha="center",
         va="center", fontsize=15, color="white", fontweight="bold")
ax2.text(0, cover.mean() + (1 - cover.mean()) / 2,
         f"놓쳤다 {1 - cover.mean():.3f}", ha="center", va="center",
         fontsize=12.5, color="white", fontweight="bold")
ax2.set_xticks([0], [f"표본 {REPS12:,} 개"], fontsize=12.5)
ax2.set_ylim(0, 1.05)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 0.95, 1.0],
               ["0", "0.25", "0.5", "0.75", "0.95", "1"])
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"그 비율은 세어 보아야 안다 — 여기서는 {cover.mean():.3f}", fontsize=12.5)

fig.suptitle("95% 는 구간에 붙은 수가 아니라 절차에 붙은 수다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, CI + "ninety_five_belongs_to_the_procedure.png")


# ════════════════════════════════════════════════════════════════════
# 13. single_mean — 한 줄을 빼먹으면 p 가 한가운데로 몰린다
#
# 쪽의 null_by_shifting 과 two_guidelines 가 귀무분포와 지침을 그린다.
# 들머리의 **셋째 문단**이 적는 「방향이 한쪽이라서 … 결국 아무것도
# 기각하지 못한다」는 **p 값의 분포**로 보아야 가장 또렷하다. 귀무가
# 참일 때 올바른 검정은 p 가 고르게 깔리는데, 옮기는 줄을 빼먹으면
# 한가운데로 몰려 0.05 아래가 아예 비어 버린다.
# ════════════════════════════════════════════════════════════════════
r13 = np.random.default_rng(17_113)
N13, REPS13, B13 = 12, 4000, 1200
p_right, p_wrong = [], []
for _ in range(REPS13):
    x = r13.exponential(1.0, N13)       # 참 평균이 1 — 귀무가 참이다
    obs = abs(x.mean() - 1.0) / (x.std(ddof=1) / np.sqrt(N13))
    shifted = x - x.mean() + 1.0        # 옮기는 한 줄
    for src_, box in ((shifted, p_right), (x, p_wrong)):
        s = src_[r13.integers(0, N13, (B13, N13))]
        t = np.abs(s.mean(1) - 1.0) / (s.std(ddof=1, axis=1)
                                       / np.sqrt(N13))
        box.append((t >= obs).mean())
p_right, p_wrong = np.array(p_right), np.array(p_wrong)
print(f"\n[single_mean] 귀무가 참인 자료 {REPS13:,} 벌")
print(f"  옮긴 쪽: p 의 가운데값 {np.median(p_right):.3f} · "
      f"0.05 아래 {np.mean(p_right < 0.05):.4f}")
print(f"  빼먹은 쪽: p 의 가운데값 {np.median(p_wrong):.3f} · "
      f"0.05 아래 {np.mean(p_wrong < 0.05):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
bins13 = np.linspace(0, 1, 26)
for ax, p_, nm, c in [(ax1, p_right, "옮기는 한 줄을 넣은 검정", "tab:blue"),
                      (ax2, p_wrong, "그 한 줄을 빼먹은 검정", "tab:red")]:
    ax.hist(p_, bins=bins13, density=True, color=c, alpha=0.75)
    ax.axhline(1.0, color="0.2", lw=2.4, ls="--")
    ax.axvline(0.05, color="0.3", lw=2.0, ls=":")
    ax.set_xlabel("p 값", fontsize=11.5)
    ax.set_title(nm, fontsize=13)
    ax.set_ylim(0, 4.2)
    ax.grid(alpha=0.25)
ax1.set_ylabel("비율", fontsize=11.5)
ax1.text(0.5, 1.12, "고른 분포 — n = 12 라 왼쪽이 조금 두텁다",
         ha="center",
         va="bottom", fontsize=11.5, color="0.2", fontweight="bold")
ax1.text(0.97, 0.93, f"0.05 아래 {np.mean(p_right < 0.05):.3f}",
         transform=ax1.transAxes, ha="right", va="top", fontsize=12,
         color="tab:blue", fontweight="bold")
ax2.text(0.97, 0.93, f"0.05 아래 {np.mean(p_wrong < 0.05):.4f}",
         transform=ax2.transAxes, ha="right", va="top", fontsize=12,
         color="tab:red", fontweight="bold")
ax2.annotate("한가운데로 몰려\n작은 p 가 아예 나오지 않는다",
             xy=(0.08, 0.6), xytext=(0.30, 3.3), fontsize=12,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))

fig.suptitle("한 줄을 빼먹으면 아무것도 기각하지 못한다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, TEST + "p_piles_up_in_the_middle.png")


# ════════════════════════════════════════════════════════════════════
# 14. two_means — 불균형과 이분산이 함께 와야 갈린다
#
# 쪽의 pool_vs_center 가 두 귀무분포와 네 방법의 크기를 그린다. 들머리의
# **셋째 문단**이 조건을 둘로 적는다 — 「둘이 같아 보이지만 분산이
# 다르고 사람 수가 불균형하면 갈린다.」 **조건이 둘**이라는 말은 두 축을
# 그려야 보인다. 한쪽만으로는 갈리지 않는 것이 요점이다.
# ════════════════════════════════════════════════════════════════════
r14 = np.random.default_rng(17_114)
RATIOS = [1.0, 1.5, 2.0, 3.0]        # 분산비
SPLITS = [(20, 20), (14, 26), (10, 30), (6, 34)]
REPS14, B14 = 1500, 600
grid_pool = np.zeros((len(SPLITS), len(RATIOS)))
grid_cent = np.zeros_like(grid_pool)
for i, (n1, n2) in enumerate(SPLITS):
    for j, ratio in enumerate(RATIOS):
        s1, s2 = np.sqrt(ratio), 1.0   # 작은 집단이 시끄러운 쪽
        hp = hc = 0
        for _ in range(REPS14):
            a = r14.standard_normal(n1) * s1
            b = r14.standard_normal(n2) * s2
            obs = abs(a.mean() - b.mean())
            pool = np.r_[a - a.mean(), b - b.mean()]
            pa = pool[r14.integers(0, len(pool), (B14, n1))].mean(1)
            pb = pool[r14.integers(0, len(pool), (B14, n2))].mean(1)
            hp += (np.abs(pa - pb) >= obs).mean() < 0.05
            ca = (a - a.mean())[r14.integers(0, n1, (B14, n1))].mean(1)
            cb = (b - b.mean())[r14.integers(0, n2, (B14, n2))].mean(1)
            hc += (np.abs(ca - cb) >= obs).mean() < 0.05
        grid_pool[i, j] = hp / REPS14
        grid_cent[i, j] = hc / REPS14
print("\n[two_means] 합쳐 놓고 뽑기 (명목 0.05)")
for i, (n1, n2) in enumerate(SPLITS):
    print(f"  {n1:2d} 대 {n2:2d}: " +
          " ".join(f"{v:.3f}" for v in grid_pool[i]))
print("  집단마다 중심 맞춰 따로 뽑기")
for i, (n1, n2) in enumerate(SPLITS):
    print(f"  {n1:2d} 대 {n2:2d}: " +
          " ".join(f"{v:.3f}" for v in grid_cent[i]))

fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.2))
vmax = max(grid_pool.max(), grid_cent.max())
for ax, g, nm in [(axs[0], grid_pool, "합쳐 놓고 두 집단을 다시 뽑기"),
                  (axs[1], grid_cent, "집단마다 중심 맞춰 따로 뽑기")]:
    im = ax.imshow(g, origin="lower", cmap="Reds", vmin=0.02, vmax=vmax,
                   aspect="auto")
    for i in range(len(SPLITS)):
        for j in range(len(RATIOS)):
            ax.text(j, i, f"{g[i, j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if g[i, j] > 0.6 * vmax else "0.15")
    ax.set_xticks(range(len(RATIOS)), [f"{r_:.1f}" for r_ in RATIOS],
                  fontsize=12)
    ax.set_yticks(range(len(SPLITS)), [f"{a} 대 {b}" for a, b in SPLITS],
                  fontsize=12)
    ax.set_xlabel("분산비 — 작은 집단 쪽이 이만큼 시끄럽다",
                  fontsize=11.5)
    ax.set_title(nm, fontsize=12.5)
axs[0].set_ylabel("사람 수의 나눔", fontsize=11.5)
fig.colorbar(im, ax=axs, label="아무 차이가 없는데 기각하는 비율",
             fraction=0.025)
axs[0].annotate("맨 아래 줄 — 사람 수가 같으면\n분산비를 키워도 "
                "움직이지 않는다", xy=(1.6, 0.06), xytext=(0.3, 1.0),
                fontsize=10.6,
                color="0.15", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.15", lw=1.8))

axs[1].annotate("사람 수가 불균형하고 분산비까지 크면\n합치는 쪽이 "
                f"{grid_pool[-1, -1]:.3f} 대 {grid_cent[-1, -1]:.3f} 로 "
                "더 부푼다",
                xy=(2.6, 3), xytext=(-0.4, 1.45), fontsize=10.6,
                color="0.15", fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.15", lw=1.8))
fig.suptitle("두 길은 사람 수가 같을 때만 같은 답을 낸다 — "
             "갈림의 방향은 분산비가 정한다", fontsize=15, y=0.99)
save(fig, TEST + "both_conditions_needed.png")
