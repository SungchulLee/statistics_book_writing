r"""「붓스트랩 세계」를 한 번 더 들어가면 무슨 일이 일어나는가.

$F \to \hat F_n$ 은 한 번 하라고 있는 대입이다. 붓스트랩은 $\hat F_n$ 에서 **한 번** 뽑아
$\hat\theta$ 의 흔들림을 잴 뿐, 새 정보를 만들지 않는다. 뽑은 것을 다시 $\hat F$ 로 삼아
되풀이하면 분포가 말라 죽는다 — 꼬리가 먼저, 그다음 전부.

① 세대에 따른 자료 — 꼬리가 먼저 비고 끝내 한 점으로 모인다.
② 분산의 붕괴 — 비모수와 모수(가우스) 재귀가 같은 속도로 죽는다.
③ 무엇이 먼저 사라지는가 — 고유값 개수와 꼬리 분위수.

만드는 파일: docs/ch17/bootstrap/img/recursive_resampling.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)
N_SHOW = 100          # ① 의 표본 크기
GENS = 400            # 따라갈 세대 수
REPS = 400            # ②③ 의 반복


def nonparam_chain(x0, gens, rng):
    """비모수 재귀 — 세대마다 앞 세대에서 n 개를 복원추출한다.
    이것이 붓스트랩을 「붓스트랩 표본 위에서」 되풀이하는 것이다."""
    n = len(x0)
    x = x0.copy()
    out = [x.copy()]
    for _ in range(gens):
        x = x[rng.integers(0, n, n)]
        out.append(x.copy())
    return out


def param_chain(x0, gens, rng):
    """모수 재귀 — 세대마다 가우스를 최대가능도로 적합하고 거기서 n 개를 뽑는다.
    논문이 든 1 차원 가우스 보기가 이것이다."""
    n = len(x0)
    x = x0.copy()
    out = [x.copy()]
    for _ in range(gens):
        x = rng.normal(x.mean(), x.std(), n)
        out.append(x.copy())
    return out


# === ① 한 사슬을 따라가며 ===========================================
x0 = RNG.standard_normal(N_SHOW)
chain = nonparam_chain(x0, GENS, RNG)
SHOW = [0, 5, 20, 50, 120, 300, 400]

# === ②③ 여러 사슬의 평균 ============================================
NS = (25, 100, 400)
var_np = {n: np.zeros(GENS + 1) for n in NS}
var_p = {n: np.zeros(GENS + 1) for n in NS}
uniq = np.zeros(GENS + 1)
q99 = np.zeros(GENS + 1)
iqr = np.zeros(GENS + 1)

for _ in range(REPS):
    for n in NS:
        z = RNG.standard_normal(n)
        for k, xs in enumerate(nonparam_chain(z, GENS, RNG)):
            var_np[n][k] += xs.var()
        for k, xs in enumerate(param_chain(z, GENS, RNG)):
            var_p[n][k] += xs.var()
    z = RNG.standard_normal(N_SHOW)
    c = nonparam_chain(z, GENS, RNG)
    base99 = np.quantile(z, 0.99) - np.quantile(z, 0.01)
    baseiq = np.quantile(z, 0.75) - np.quantile(z, 0.25)
    for k, xs in enumerate(c):
        uniq[k] += len(np.unique(xs))
        q99[k] += (np.quantile(xs, 0.99) - np.quantile(xs, 0.01)) / base99
        iqr[k] += (np.quantile(xs, 0.75) - np.quantile(xs, 0.25)) / baseiq

for n in NS:
    var_np[n] /= REPS
    var_p[n] /= REPS
uniq /= REPS
q99 /= REPS
iqr /= REPS

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

for j, g in enumerate(SHOW):
    xs = chain[g]
    ax1.scatter(np.full(len(xs), j) + RNG.uniform(-0.17, 0.17, len(xs)), xs,
                s=14, color="#1f77b4", alpha=0.45, edgecolors="none")
    ax1.text(j, 3.3, f"${len(np.unique(xs))}$", ha="center", fontsize=10.5,
             color="#d62728")
ax1.axhline(0, color="black", lw=1.0, ls=":")
ax1.set_xticks(range(len(SHOW)))
ax1.set_xticklabels([f"${g}$" for g in SHOW])
ax1.set_xlabel("세대 (재표집을 되풀이한 횟수)")
ax1.set_ylabel("자료값")
ax1.set_title(f"① 한 사슬을 따라가면 ($n = {N_SHOW}$)\n"
              f"위의 붉은 수는 살아남은 고유값 개수", fontsize=12)
ax1.set_ylim(-3.4, 3.7)
ax1.grid(alpha=0.25, axis="y")

ks = np.arange(GENS + 1)
def mask(v):
    """사슬이 한 점에 흡수되면 모의 평균이 0 으로 떨어져 로그 눈금에서
    수직선을 긋는다. 식은 기댓값이라 살아남은 극소수 사슬이 떠받치는데,
    반복 횟수로는 그 꼬리를 잡을 수 없다. 바닥 아래는 그리지 않는다."""
    return np.where(v > 2e-6, v, np.nan)


for n, c in zip(NS, ("#d62728", "#1f77b4", "#2ca02c")):
    ax2.semilogy(ks, mask(var_np[n] / var_np[n][0]), color=c, lw=2.2,
                 label=f"비모수 재귀 $n = {n}$")
    ax2.semilogy(ks, mask(var_p[n] / var_p[n][0]), color=c, lw=1.4, ls=":")
    ax2.semilogy(ks, (1 - 1 / n) ** ks, color="black", lw=1.0, ls="--",
                 zorder=1)
ax2.semilogy([], [], color="black", lw=1.0, ls="--", label="$(1 - 1/n)^k$")
ax2.semilogy([], [], color="#555555", lw=1.4, ls=":", label="모수(가우스) 재귀")
ax2.set_xlabel("세대 $k$")
ax2.set_ylabel("남은 분산 (처음을 $1$ 로)")
ax2.set_title("② 분산은 세대마다 $1/n$ 씩 깎인다", fontsize=12)
ax2.set_ylim(1e-7, 2)
ax2.set_yticks([1e-6, 1e-4, 1e-2, 1])
ax2.set_yticklabels(["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"])
ax2.minorticks_off()
ax2.legend(loc="lower left", fontsize=9)
ax2.grid(alpha=0.25, which="major")

ax3.plot(ks, q99 / q99[0], color="#d62728", lw=2.4,
         label="꼬리 폭 ($1\\%$–$99\\%$)")
ax3.plot(ks, iqr / iqr[0], color="#1f77b4", lw=2.4, label="사분위수범위")
ax3.set_xlabel("세대 $k$")
ax3.set_ylabel("처음 대비 남은 비율")
ax3.set_xlim(0, 120)
ax3.set_ylim(0, 1.05)
ax3b = ax3.twinx()
ax3b.plot(ks, uniq, color="#9e9e9e", lw=2.0, ls="--")
ax3b.set_ylabel("살아남은 고유값 개수", color="#777777")
ax3b.tick_params(axis="y", labelcolor="#777777")
ax3b.set_ylim(0, N_SHOW * 1.05)
ax3.set_xlim(0, 120)   # twinx 가 가로 눈금을 되돌리므로 다시 건다
ax3.set_title("③ 꼬리가 먼저 간다", fontsize=12)
h1, l1 = ax3.get_legend_handles_labels()
ax3.legend(h1 + [plt.Line2D([], [], color="#9e9e9e", lw=2.0, ls="--")],
           l1 + ["고유값 개수 (오른쪽 눈금)"], loc="upper right", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch17/bootstrap/img/recursive_resampling.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① 한 사슬 ($n = {N_SHOW}$)")
print(f"{'세대':>6}{'고유값':>8}{'표준편차':>10}{'범위':>10}")
for g in SHOW:
    xs = chain[g]
    print(f"{g:>6}{len(np.unique(xs)):>8}{xs.std():>10.3f}"
          f"{xs.max() - xs.min():>10.3f}")

print(f"\n② 남은 분산 (반복 {REPS} 회 평균)")
print(f"{'세대':>6}" + "".join(f"{'n=' + str(n):>12}" for n in NS)
      + "".join(f"{'식 n=' + str(n):>12}" for n in NS))
for k in (0, 10, 25, 50, 100, 200, 400):
    print(f"{k:>6}" + "".join(f"{var_np[n][k] / var_np[n][0]:>12.2e}" for n in NS)
          + "".join(f"{(1 - 1 / n) ** k:>12.2e}" for n in NS))

print(f"\n   모수(가우스) 재귀도 같은가 — n = 100, 세대 100 에서")
print(f"   비모수 {var_np[100][100] / var_np[100][0]:.4f}, "
      f"모수 {var_p[100][100] / var_p[100][0]:.4f}, "
      f"식 {(1 - 1 / 100) ** 100:.4f}")

print(f"\n③ n = {N_SHOW} 에서 세대별")
print(f"{'세대':>6}{'고유값':>9}{'꼬리 폭':>10}{'사분위수범위':>14}")
for k in (0, 5, 10, 20, 50, 100, 200):
    print(f"{k:>6}{uniq[k]:>9.1f}{q99[k]:>10.3f}{iqr[k]:>14.3f}")
half_q = np.argmax(q99 < 0.5)
half_i = np.argmax(iqr < 0.5)
print(f"\n   절반으로 줄기까지 — 꼬리 폭 {half_q} 세대, 사분위수범위 {half_i} 세대")
