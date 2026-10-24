r"""벤치마크 비교는 일원배치처럼 보이지만 실은 **난괴법**이다.

모형 $k$ 개를 **같은 문항 $n$ 개**로 잰다. 그러면 문항이 블록이다. 그런데 보고되는
수는 모형마다의 정확도 하나뿐이고, 그것들을 견주는 일은 블록을 버린 일원배치가 된다.

문항 $i$ 의 난이도를 $\theta_i \sim N(0, s^2)$, 모형 $m$ 의 능력을 $\beta_m$ 이라 두고
$P(\text{정답}) = \sigma(\beta_m - \theta_i)$ 로 채점표를 만든다.

① 참 차이가 **없을 때** 일원배치의 기각률이 $0.05$ 가 아니라 $0.001$ 로 내려간다.
   난괴법은 $0.05$ 를 지킨다.
② 까닭이 정확히 계산된다 — $E[F] \approx \sigma_E^2/\sigma_W^2$, 곧 분산비의 역수다.
③ 참 차이가 **있을 때**는 그 보수성이 그대로 검정력 손실이 된다.

만드는 파일: docs/ch13/anova_one_way/img/benchmark_blocks.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

K = 4
BASE = 1.1                               # 모형 능력의 기준 (정확도 약 0.65)
STEP = 0.09                              # 대립가설에서 모형 사이의 간격
S0 = 2.5                                 # 기준으로 삼는 난이도 흩어짐
SS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
NS = np.array([100, 200, 400, 800, 1600, 3200])
NFIX = 800
REPS = 4000
ALPHA = 0.05


def draw(n, s, beta, reps=REPS):
    th = RNG.normal(0.0, s, (reps, n))
    p = 1.0 / (1.0 + np.exp(-(beta[:, None, None] - th[None])))
    return (RNG.random((K, reps, n)) < p).astype(float)


def analyse(X):
    """같은 채점표에 두 분석을 돌린다."""
    _, _, n = X.shape
    one = stats.f_oneway(*[X[i] for i in range(K)], axis=1)
    gm = X.mean(axis=(0, 2), keepdims=True)
    mrow = X.mean(axis=2, keepdims=True)     # 모형 평균
    mcol = X.mean(axis=0, keepdims=True)     # 문항 평균
    sst = ((X - gm) ** 2).sum(axis=(0, 2))
    ssm = n * ((mrow - gm) ** 2).sum(axis=(0, 2))
    ssi = K * ((mcol - gm) ** 2).sum(axis=(0, 2))
    sse = sst - ssm - ssi
    msw = (sst - ssm) / (K * (n - 1))        # 일원배치의 오차평균제곱
    mse = sse / ((K - 1) * (n - 1))          # 난괴법의 오차평균제곱
    fb = (ssm / (K - 1)) / mse
    return dict(
        rej_one=float((one.pvalue < ALPHA).mean()),
        rej_blk=float((stats.f.sf(fb, K - 1, (K - 1) * (n - 1)) < ALPHA).mean()),
        f_one=float(one.statistic.mean()),
        ratio=float(msw.mean() / mse.mean()),
    )


H0 = np.full(K, BASE)
H1 = BASE + np.arange(K) * STEP
acc = draw(4000, S0, H1, 1).mean(axis=(1, 2))
print(f"대립가설에서 모형 {K} 개의 정확도 {np.round(acc, 3)} "
      f"(가장 먼 둘의 차이 {acc.max() - acc.min():.3f})")

# === ①② 참 차이가 없을 때 ===========================================
print(f"\n①② 네 모형이 정말 같을 때 (문항 {NFIX} 개, {REPS} 회)")
print(f"{'s':>5}{'일원배치':>11}{'난괴법':>10}{'평균 F':>9}{'1/분산비':>11}"
      f"{'모두 같은 답':>13}")
r_one, r_blk, f_one, inv, same = [], [], [], [], []
for s in SS:
    X = draw(NFIX, s, H0)
    d = analyse(X)
    tot = X.sum(axis=0)
    same.append(float(((tot == 0) | (tot == K)).mean()))
    r_one.append(d["rej_one"]); r_blk.append(d["rej_blk"])
    f_one.append(d["f_one"]); inv.append(1.0 / d["ratio"])
    print(f"{s:>5.1f}{r_one[-1]:>11.4f}{r_blk[-1]:>10.4f}{f_one[-1]:>9.3f}"
          f"{inv[-1]:>11.3f}{100 * same[-1]:>12.1f}%")
r_one, r_blk, f_one, inv, same = map(
    np.array, (r_one, r_blk, f_one, inv, same))
i0 = int(np.argmin(np.abs(SS - S0)))
print(f"\n  s = {S0} 에서 일원배치의 기각률 {r_one[i0]:.4f} "
      f"(명목의 {ALPHA / r_one[i0]:.0f} 분의 1)")
print(f"  평균 F {f_one[i0]:.3f} 가 분산비의 역수 {inv[i0]:.3f} 와 맞는다")

# === ③ 참 차이가 있을 때 ============================================
print(f"\n③ 모형 차이가 있을 때의 검정력 (s = {S0})")
print(f"{'문항 n':>8}{'일원배치':>11}{'난괴법':>10}")
p_one, p_blk = [], []
for n in NS:
    d = analyse(draw(int(n), S0, H1))
    p_one.append(d["rej_one"]); p_blk.append(d["rej_blk"])
    print(f"{n:>8}{p_one[-1]:>11.3f}{p_blk[-1]:>10.3f}")
p_one, p_blk = np.array(p_one), np.array(p_blk)

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

FLOOR = 1.0 / (2 * REPS)          # 0 번 기각한 자리를 그림에 올리기 위한 바닥
r_one_p = np.maximum(r_one, FLOOR)
ax1.semilogy(SS, r_one_p, "s--", color="#d62728", lw=2.5, ms=8,
             label="일원배치 (블록을 버린다)")
for s_, v in zip(SS, r_one):
    if v == 0:
        ax1.annotate("$0/4000$", xy=(s_, FLOOR), xytext=(s_ - 0.12, FLOOR * 1.9),
                     fontsize=9.5, color="#d62728", ha="right")
ax1.semilogy(SS, r_blk, "o-", color="#1f77b4", lw=2.8, ms=9,
             label="난괴법 (문항이 블록)")
ax1.axhline(ALPHA, color="black", lw=1.3, ls=":")
ax1.text(0.55, 0.062, "명목 $\\alpha = 0.05$", fontsize=10)
ax1.axvline(S0, color="#999999", lw=1.3, ls="-.")
ax1.annotate(f"${r_one[i0]:.4f}$", xy=(S0, r_one[i0]), xytext=(1.25, 0.0012),
             fontsize=11.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax1.set_ylim(FLOOR * 0.7, 0.2)
ax1.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
ax1.set_yticklabels(["$0.0001$", "$0.001$", "$0.01$", "$0.1$"])
ax1.minorticks_off()
ax1.set_xlabel("문항 난이도의 흩어짐 $s$")
ax1.set_ylabel("참 차이가 없을 때의 기각률")
ax1.set_title("① 블록을 버리면 검정이 멎는다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25, which="major")

ax2.plot(SS, inv, "-", color="#333333", lw=2.4,
         label="이론값 $\\sigma_E^2/\\sigma_W^2$")
ax2.plot(SS, f_one, "o", color="#d62728", ms=10,
         label="실제 $F$ 의 평균 (일원배치)")
ax2.axhline(1.0, color="black", lw=1.2, ls=":")
ax2.text(0.55, 1.03, "$H_0$ 가 참이면 $1$ 이어야 한다", fontsize=10)
ax2b = ax2.twinx()
ax2b.plot(SS, 100 * same, color="#2ca02c", lw=2.0, ls="--")
ax2b.set_ylabel(f"${K}$ 개 모형이 똑같이 답한 문항 ($\\%$)", color="#2ca02c")
ax2b.tick_params(axis="y", labelcolor="#2ca02c")
ax2b.set_ylim(0, 100)
ax2.set_xlabel("문항 난이도의 흩어짐 $s$")
ax2.set_ylabel("$F$ 의 크기")
ax2.set_ylim(0, 1.15)
ax2.set_title("② 분모가 재는 것이 틀렸다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.semilogx(NS, p_one, "s--", color="#d62728", lw=2.5, ms=8,
             label="일원배치")
ax3.semilogx(NS, p_blk, "o-", color="#1f77b4", lw=2.8, ms=9, label="난괴법")
ax3.axhline(ALPHA, color="black", lw=1.2, ls=":")
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("벤치마크 문항 수")
ax3.set_ylabel("모형 차이를 잡아낼 확률")
ax3.set_ylim(0, 1.0)
ax3.set_title(f"③ 그 값이 검정력으로 돌아온다 ($s = {S0}$)", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/anova_one_way/img/benchmark_blocks.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
