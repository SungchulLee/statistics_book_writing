r"""같은 문항을 풀렸으면 짝을 지어야 한다.

두 모형을 **같은 문항 집합**으로 재면 문항별 정오가 강하게 상관한다 — 쉬운 문항은
둘 다 맞고 어려운 문항은 둘 다 틀린다. 그 상관을 버리고 전체 점수만 견주면 표준오차가
크게 부풀고, 구간이 겹쳐서 「차이 없음」으로 읽힌다.

짝지은 차이 $d_i = s_{A,i} - s_{B,i}$ 는 $\{-1, 0, +1\}$ 값을 갖고, 부호 뒤집기 순열은
**$0$ 이 아닌 문항** 곧 둘의 답이 갈린 문항만 쓴다.

① 문항별 정오의 $2 \times 2$ 표 — 정보는 비대각선에만 있다.
② 모형별 구간은 겹치지만 짝지은 차이의 구간은 $0$ 을 넘지 않는다.
③ 문항 수에 따른 검정력 — 짝지을 때와 짝짓지 않을 때.

만드는 파일: docs/ch17/permutation/img/paired_evals.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N_SHOW = 1000       # ①② 에 쓸 문항 수
B_SIGN = 20000      # 부호 뒤집기 횟수
B_POWER = 3000      # ③ 의 반복
TH_A, TH_B = 5.44, 4.82   # 두 모형의 능력 (정답률 87% 와 84% 가 되도록)
B_SD = 4.5          # 문항 난이도의 흩어짐 — 아주 쉬운 문항부터 아주 어려운 것까지


def logistic(z):
    return 1 / (1 + np.exp(-z))


def one_eval(n, rng):
    """난이도가 흩어진 문항 n 개를 두 모형에 풀린다 (문항반응 모형)."""
    b = B_SD * rng.standard_normal(n)
    sa = (rng.random(n) < logistic(TH_A - b)).astype(int)
    sb = (rng.random(n) < logistic(TH_B - b)).astype(int)
    return sa, sb


# ①② 에는 **전형적인 한 판**을 쓴다. 참 격차는 3.0 백분점인데 한 판의 관측 격차는
# 표준오차만큼 흔들리므로, 참값에서 0.4 백분점 안에 든 판이 나올 때까지 다시 뽑는다.
# (이것은 그림용 선택일 뿐이며, ③ 의 검정력은 선택 없이 전부를 센 값이다.)
TRUE_GAP = 0.030
for _tries in range(1, 200):
    sa, sb = one_eval(N_SHOW, RNG)
    if abs((sa.mean() - sb.mean()) - TRUE_GAP) < 0.004:
        break
print(f"  ①② 에 쓸 전형적인 판을 {_tries} 번째에 얻었다")
d = sa - sb
tab = np.array([[np.sum((sa == 1) & (sb == 1)), np.sum((sa == 1) & (sb == 0))],
                [np.sum((sa == 0) & (sb == 1)), np.sum((sa == 0) & (sb == 0))]])

# --- 두 가지 표준오차 -----------------------------------------------
se_a = sa.std(ddof=1) / np.sqrt(N_SHOW)
se_b = sb.std(ddof=1) / np.sqrt(N_SHOW)
se_naive = np.sqrt(se_a**2 + se_b**2)      # 상관을 버린 경우
se_paired = d.std(ddof=1) / np.sqrt(N_SHOW)  # 짝지은 경우
rho = np.corrcoef(sa, sb)[0, 1]

# --- 부호 뒤집기 순열 (이 쪽의 알고리즘 그대로) -----------------------
# d 의 부호를 독립적으로 뒤집는다. d_i = 0 인 문항은 뒤집어도 0 이므로
# 검정에 아무것도 보태지 못한다 — 갈린 문항만 일한다.
flips = RNG.choice([-1, 1], size=(B_SIGN, N_SHOW))
null_mean = (flips * d).mean(axis=1)
p_perm = (np.sum(np.abs(null_mean) >= abs(d.mean())) + 1) / (B_SIGN + 1)
p_paired_z = 2 * stats.norm.sf(abs(d.mean()) / se_paired)
p_naive_z = 2 * stats.norm.sf(abs(sa.mean() - sb.mean()) / se_naive)

# === ③ 검정력 =====================================================
NS = np.array([200, 400, 700, 1000, 1500, 2200, 3200])
pow_p, pow_n = [], []
for n in NS:
    b = B_SD * RNG.standard_normal((B_POWER, n))
    A = (RNG.random((B_POWER, n)) < logistic(TH_A - b)).astype(np.int8)
    B_ = (RNG.random((B_POWER, n)) < logistic(TH_B - b)).astype(np.int8)
    D = A - B_
    zp = D.mean(axis=1) / (D.std(axis=1, ddof=1) / np.sqrt(n))
    zn = (A.mean(axis=1) - B_.mean(axis=1)) / np.sqrt(
        (A.var(axis=1, ddof=1) + B_.var(axis=1, ddof=1)) / n)
    pow_p.append(np.mean(np.abs(zp) > 1.959964) * 100)
    pow_n.append(np.mean(np.abs(zn) > 1.959964) * 100)
    print(f"  n = {n:>4}  짝지음 {pow_p[-1]:5.1f}%   짝 안 지음 {pow_n[-1]:5.1f}%")
pow_p, pow_n = np.array(pow_p), np.array(pow_n)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

im = ax1.imshow(tab, cmap="Blues", vmin=0, vmax=tab.max())
for i in range(2):
    for j in range(2):
        off = i != j
        ax1.text(j, i, f"{tab[i, j]}", ha="center", va="center",
                 fontsize=20 if off else 16,
                 fontweight="bold" if off else "normal",
                 color="#d62728" if off else
                 ("white" if tab[i, j] > tab.max() * 0.6 else "#333333"))
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["맞음", "틀림"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["맞음", "틀림"])
ax1.set_xlabel("모형 B")
ax1.set_ylabel("모형 A")
ax1.set_title(f"① 문항 ${N_SHOW:,}$ 개의 정오 표".replace(",", "{,}"), fontsize=12,
              pad=26)
disc = tab[0, 1] + tab[1, 0]
ax1.text(0.5, 1.025, f"갈린 문항은 ${disc}$ 개 — 검정이 쓰는 것은 이것뿐이다",
         ha="center", fontsize=10.5, color="#d62728", transform=ax1.transAxes)

ax2.errorbar([0], [sa.mean() * 100], yerr=[1.96 * se_a * 100], fmt="o",
             color="#1f77b4", ms=9, capsize=7, lw=2.2, label="모형 A")
ax2.errorbar([1], [sb.mean() * 100], yerr=[1.96 * se_b * 100], fmt="o",
             color="#2ca02c", ms=9, capsize=7, lw=2.2, label="모형 B")
lo_a, hi_a = (sa.mean() - 1.96 * se_a) * 100, (sa.mean() + 1.96 * se_a) * 100
lo_b, hi_b = (sb.mean() - 1.96 * se_b) * 100, (sb.mean() + 1.96 * se_b) * 100
ax2.axhspan(max(lo_a, lo_b), min(hi_a, hi_b), color="#ffcccc", alpha=0.8, zorder=0)
ax2.text(0.5, (max(lo_a, lo_b) + min(hi_a, hi_b)) / 2, "구간이 겹친다",
         ha="center", fontsize=10.5, color="#a33")
ax2b = ax2.twinx()
ax2b.errorbar([2.05], [d.mean() * 100], yerr=[1.96 * se_paired * 100], fmt="s",
              color="#d62728", ms=9, capsize=7, lw=2.2)
ax2b.errorbar([2.60], [d.mean() * 100], yerr=[1.96 * se_naive * 100], fmt="s",
              color="#9e9e9e", ms=9, capsize=7, lw=2.2)
# 0 선은 오른쪽 눈금에만 걸리는 선이므로 왼쪽 영역까지 끌지 않는다
ax2b.hlines(0, 1.78, 2.88, color="black", lw=1.4, ls="--")
ax2b.text(1.82, 0.22, "차이 $0$", fontsize=10)
ax2b.set_ylabel("A $-$ B (백분점)", color="#d62728")
ax2b.tick_params(axis="y", labelcolor="#d62728")
ax2b.set_ylim(-2.2, 7.5)
ax2b.text(2.05, d.mean() * 100 + 1.96 * se_paired * 100 + 0.35, "짝지음",
          ha="center", fontsize=10, color="#d62728")
ax2b.text(2.60, d.mean() * 100 + 1.96 * se_naive * 100 + 0.35, "짝 안 지음",
          ha="center", fontsize=10, color="#777777")
ax2.axvline(1.65, color="#cccccc", lw=1.2)
ax2.set_xlim(-0.5, 3.0)
ax2.set_xticks([0, 1, 2.32])
ax2.set_xticklabels(["모형 A", "모형 B", "두 모형의 차이\n(오른쪽 눈금)"])
ax2.set_ylabel("정답률 ($\\%$)", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_title("② 겹치는 구간, 겹치지 않는 차이", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(NS, pow_p, "o-", color="#d62728", lw=2.4, ms=6, label="짝지은 검정")
ax3.plot(NS, pow_n, "s-", color="#9e9e9e", lw=2.4, ms=6, label="짝짓지 않은 검정")
ax3.axhline(80, color="black", lw=1.3, ls="--")
ax3.text(3000, 82, "검정력 $80\\%$", ha="right", fontsize=10)
ax3.set_xlabel("문항 수 $n$")
ax3.set_ylabel("검정력 ($\\%$)")
ax3.set_title("③ 같은 검정력에 드는 문항이 절반 이하다", fontsize=12)
ax3.set_xscale("log")
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax3.minorticks_off()
ax3.set_ylim(0, 105)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch17/permutation/img/paired_evals.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① 문항 {N_SHOW:,} 개 — A {sa.mean() * 100:.1f}%, B {sb.mean() * 100:.1f}% "
      f"(차이 {(sa.mean() - sb.mean()) * 100:.1f} 백분점)")
print(f"   둘 다 맞음 {tab[0, 0]}, A 만 맞음 {tab[0, 1]}, "
      f"B 만 맞음 {tab[1, 0]}, 둘 다 틀림 {tab[1, 1]}")
print(f"   갈린 문항 {disc} 개 = 전체의 {disc / N_SHOW * 100:.1f}%")
print(f"   문항별 정오의 상관 {rho:.3f}")

print(f"\n② 표준오차 — 짝 안 지음 {se_naive * 100:.2f} 백분점, "
      f"짝지음 {se_paired * 100:.2f} 백분점 ({se_naive / se_paired:.2f} 배)")
print(f"   모형별 95% 구간: A [{lo_a:.1f}, {hi_a:.1f}], B [{lo_b:.1f}, {hi_b:.1f}] "
      f"— 겹치는 폭 {min(hi_a, hi_b) - max(lo_a, lo_b):.2f} 백분점")
print(f"   짝지은 차이의 95% 구간: "
      f"[{(d.mean() - 1.96 * se_paired) * 100:.2f}, "
      f"{(d.mean() + 1.96 * se_paired) * 100:.2f}]")
print(f"   짝 안 지은 차이의 95% 구간: "
      f"[{(d.mean() - 1.96 * se_naive) * 100:.2f}, "
      f"{(d.mean() + 1.96 * se_naive) * 100:.2f}]")
print(f"\n   부호 뒤집기 순열 p = {p_perm:.4f}  (B = {B_SIGN:,})")
print(f"   짝지은 정규근사 p = {p_paired_z:.4f}")
print(f"   짝 안 지은 정규근사 p = {p_naive_z:.4f}")

print(f"\n③ 검정력 80% 를 넘는 문항 수")
for lab, arr in (("짝지음", pow_p), ("짝 안 지음", pow_n)):
    idx = np.where(arr >= 80)[0]
    print(f"   {lab:>8}: {NS[idx[0]] if len(idx) else '> ' + str(NS[-1])} 문항")
