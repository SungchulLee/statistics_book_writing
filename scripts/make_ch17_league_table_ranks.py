r"""순위는 닫힌 꼴 표준오차가 없는 전형적인 통계량이다 — 그래서 재표집이 맡는다.

기관 $52$ 곳이 저마다 다른 수의 시술을 하고 저마다 다른 성공률을 가진다. 성공률의
**차이는 실재**해도, 그것을 줄 세운 **순위**는 표본요동에 크게 흔들린다. 각 기관의
$0/1$ 결과를 재표집해 순위의 분포를 직접 만들어 본다.

① 순위표와 각 기관의 순위 $95\%$ 구간.
② 구간의 길이는 사례 수가 적을수록 길다.
③ 참 성공률을 그대로 둔 채 이듬해를 다시 뽑으면 순위가 얼마나 뛰는가.

만드는 파일: docs/ch17/bootstrap_ci/img/league_table_ranks.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)
K = 52            # 기관 수
B = 20000         # 재표집 횟수
P_MID, P_SD = 0.14, 0.28   # 성공률의 중심과 로짓 눈금의 흩어짐


def logistic(z):
    return 1 / (1 + np.exp(-z))


# --- 기관 만들기 ----------------------------------------------------
# 시술 건수는 기관마다 크게 다르다 (작은 곳과 큰 곳이 20 배 차이).
n = np.round(np.exp(RNG.uniform(np.log(40), np.log(850), K))).astype(int)
p_true = logistic(np.log(P_MID / (1 - P_MID)) + P_SD * RNG.standard_normal(K))
succ = RNG.binomial(n, p_true)
p_hat = succ / n


def ranks_of(rates, rng):
    """성공률이 높은 쪽이 1 등. 동점은 무작위로 가른다 (순위표가 늘 하는 일)."""
    jitter = rng.random(rates.shape) * 1e-9
    order = np.argsort(-(rates + jitter), axis=-1, kind="stable")
    out = np.empty_like(order)
    np.put_along_axis(out, order,
                      np.broadcast_to(np.arange(1, rates.shape[-1] + 1), order.shape),
                      axis=-1)
    return out


obs_rank = ranks_of(p_hat, RNG)

# --- 재표집 --------------------------------------------------------
# 기관 i 의 0/1 결과 n_i 개를 복원추출하면 성공 횟수는 정확히
# Binomial(n_i, p_hat_i) 를 따른다. 그래서 이항난수를 뽑는 것이 곧
# 비모수 붓스트랩이다.
boot_rates = RNG.binomial(n[None, :], p_hat[None, :], size=(B, K)) / n[None, :]
boot_ranks = ranks_of(boot_rates, RNG)
lo = np.percentile(boot_ranks, 2.5, axis=0)
hi = np.percentile(boot_ranks, 97.5, axis=0)
width = hi - lo

# --- 하위 1/4 에 확실히 드는 기관은 몇 곳인가 ------------------------
BOTTOM = int(np.ceil(0.75 * K)) + 1          # 하위 1/4 이 시작하는 순위
n_bottom = int(np.sum(lo >= BOTTOM))
n_top = int(np.sum(hi <= K - BOTTOM + 1))    # 상위 1/4 에 확실히 드는 기관

# --- 이듬해 ---------------------------------------------------------
# 참 성공률은 하나도 바꾸지 않고 한 해를 다시 뽑는다.
year2 = RNG.binomial(n[None, :], p_true[None, :], size=(2000, K)) / n[None, :]
r2 = ranks_of(year2, RNG)
shift = np.abs(r2 - obs_rank[None, :])

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 5.2))

o = np.argsort(obs_rank)
y = np.arange(K)
ax1.hlines(y, lo[o], hi[o], color="#9e9e9e", lw=1.6)
sc = ax1.scatter(obs_rank[o], y, s=16, c=n[o], cmap="viridis",
                 norm=plt.matplotlib.colors.LogNorm(), zorder=3)
cb = fig.colorbar(sc, ax=ax1, pad=0.02)
cb.set_label("그해 시술 건수", fontsize=9.5)
cb.set_ticks([50, 100, 200, 400, 800])
cb.set_ticklabels(["$50$", "$100$", "$200$", "$400$", "$800$"])
cb.ax.minorticks_off()
ax1.axvspan(BOTTOM - 0.5, K + 0.5, color="#f2dede", alpha=0.6, zorder=0)
ax1.text(K - 0.5, 1.5, "하위 $1/4$", ha="right", fontsize=9.5, color="#a33")
ax1.set_ylim(K - 0.5, -0.5)
ax1.set_xlim(0, K + 1)
ax1.set_xlabel("순위 ($1$ 등이 성공률 최고)")
ax1.set_yticks([])
ax1.set_ylabel("기관 — 발표된 순위 차례로 쌓았다")
ax1.set_title(f"① 기관 ${K}$ 곳, 순위의 $95\\%$ 구간", fontsize=12)
ax1.annotate(f"$1$ 등의 구간\n${lo[o][0]:.0f}$–${hi[o][0]:.0f}$ 등",
             xy=(hi[o][0], 0), xytext=(hi[o][0] + 3, 6.5), fontsize=9.5,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax1.grid(alpha=0.2, axis="x")

ax2.scatter(n, width, s=26, color="#1f77b4", alpha=0.8, edgecolors="none")
ax2.set_xscale("log")
ax2.set_xticks([50, 100, 200, 400, 800])
ax2.set_xticklabels(["$50$", "$100$", "$200$", "$400$", "$800$"])
ax2.minorticks_off()
ax2.set_xlabel("그해 시술 건수")
ax2.set_ylabel("순위 $95\\%$ 구간의 길이 (등)")
ax2.set_title("② 작은 기관일수록 순위를 모른다", fontsize=12)
ax2.axhline(K / 2, color="#d62728", lw=1.4, ls="--")
ax2.text(820, K / 2 + 1.2, f"전체의 절반(${K // 2}$ 등)", ha="right", fontsize=9.5,
         color="#d62728")
ax2.grid(alpha=0.25)

bins = np.arange(0, shift.max() + 2) - 0.5
ax3.hist(shift.ravel(), bins=bins, density=True, color="#9e9e9e",
         edgecolor="none")
m = shift.mean()
ax3.axvline(m, color="#d62728", lw=1.8, ls="--")
ax3.text(m + 1.0, ax3.get_ylim()[1] * 0.82,
         f"평균 ${m:.1f}$ 등", fontsize=10.5, color="#d62728")
f10 = np.mean(shift >= 10) * 100
ax3.text(0.97, 0.60, f"$10$ 등 이상 움직일 확률  ${f10:.0f}\\%$",
         transform=ax3.transAxes, ha="right", fontsize=10.5)
ax3.set_xlabel("이듬해 순위가 움직인 등수")
ax3.set_ylabel("밀도")
ax3.set_title("③ 실력이 하나도 안 변해도 이만큼 움직인다", fontsize=12)
ax3.set_xlim(-0.5, 32)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch17/bootstrap_ci/img/league_table_ranks.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n기관 {K} 곳 — 시술 건수 {n.min()}–{n.max()} 건 (중앙값 {int(np.median(n))})")
print(f"참 성공률 {p_true.min() * 100:.1f}–{p_true.max() * 100:.1f}% "
      f"(가장 좋은 곳이 가장 나쁜 곳의 {p_true.max() / p_true.min():.1f} 배)")
print(f"관측 성공률 {p_hat.min() * 100:.1f}–{p_hat.max() * 100:.1f}%")

print(f"\n① 순위 95% 구간 — 길이 중앙값 {np.median(width):.0f} 등, "
      f"가장 짧은 것 {width.min():.0f} 등, 가장 긴 것 {width.max():.0f} 등")
print(f"{'발표 순위':>8}{'시술 건수':>10}{'관측 성공률':>12}{'순위 95% 구간':>16}")
for j in list(o[:5]) + list(o[-3:]):
    print(f"{obs_rank[j]:>8}{n[j]:>10,}{p_hat[j] * 100:>11.1f}%"
          f"{f'{lo[j]:.0f}–{hi[j]:.0f} 등':>17}")

print(f"\n  순위 95% 구간이 하위 1/4(= {BOTTOM} 등 아래)에 온전히 드는 기관: {n_bottom} 곳")
print(f"  상위 1/4 에 온전히 드는 기관: {n_top} 곳")

print(f"\n② 구간 길이와 건수 — 건수 하위 1/3 의 평균 길이 "
      f"{width[n <= np.quantile(n, 1 / 3)].mean():.0f} 등, "
      f"상위 1/3 은 {width[n >= np.quantile(n, 2 / 3)].mean():.0f} 등")

print(f"\n③ 참 성공률을 그대로 두고 한 해를 다시 뽑으면")
print(f"  순위가 움직인 등수: 평균 {shift.mean():.1f}, 중앙값 {np.median(shift):.0f}, "
      f"최대 {shift.max()}")
print(f"  5 등 이상 움직일 확률 {np.mean(shift >= 5) * 100:.0f}%, "
      f"10 등 이상 {np.mean(shift >= 10) * 100:.0f}%, "
      f"20 등 이상 {np.mean(shift >= 20) * 100:.0f}%")
top_j = o[0]
print(f"  올해 1 등인 기관이 이듬해 받는 순위: 중앙값 {np.median(r2[:, top_j]):.0f} 등, "
      f"가운데 95% {np.percentile(r2[:, top_j], 2.5):.0f}–"
      f"{np.percentile(r2[:, top_j], 97.5):.0f} 등")
