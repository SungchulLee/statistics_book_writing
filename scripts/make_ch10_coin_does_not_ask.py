"""10.5절 실험과 인과 — 동전은 그 변수가 무엇인지 묻지 않는다.

들머리의 물음을 그린다. 「동전을 한 번 던지는 일이 어떻게 **측정하지도
않은** 변수까지 균형 잡는가.」

왼쪽이 그 기제다. 자기선택에서는 어떤 변수가 치우치는 정도가 그 변수가
선택에 들어가는 세기에 그대로 비례한다. 무작위배정에서는 그 세기가
무엇이든 기대 치우침이 0 이다 --- 동전은 자료가 만들어지기 전에 던져져
어떤 변수와도 얽히지 않기 때문이다. 그래서 재지 않은 변수도 똑같이
균형이 잡힌다. 「무엇이 중요한지 모를 때에도 작동한다」가 이 그림이다.

오른쪽은 그 균형이 무엇인지 정확히 말한다. **매번 맞는다는 뜻이 아니라
치우치지 않는다는 뜻이다.** n = 50 에서는 표준화 평균차가 0.2 를 넘는
변수가 흔하고, n 이 커지면 1/sqrt(n) 으로 좁아진다. 자기선택의 중심은
n 이 아무리 커져도 그 자리에 남는다.

쪽의 randomization_ate_bias 는 변수 다섯의 한 번짜리 치우침과 추정값의
분포를 그린다. 여기서는 그 둘이 건드리지 않는 것 --- 치우침이 「그 변수가
선택에 들어가는 세기」의 함수라는 것과, 균형의 정확한 뜻 --- 만 본다.

자료생성과정은 쪽의 것과 같은 꼴이다. 공변량은 표준정규, 자기선택은
로짓이 공변량들의 가중합, 무작위배정은 공정한 동전이다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch10/causation/img/"

N_PAGE = 400          # 쪽의 모의실험과 같은 크기
REPS = 2000
rng = np.random.default_rng(777)


def smd(z, x):
    """표준화 평균차 — 처치군과 대조군의 평균차를 합동 표준편차로 나눈 것."""
    a, b = z[x == 1], z[x == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return (a.mean() - b.mean()) / s


def trial(n, weights, design, gen):
    """공변량마다 선택에 들어가는 세기를 달리 준 한 번의 시행."""
    K = len(weights)
    Z = gen.normal(0, 1, (n, K))
    if design == "self":
        logit = Z @ weights
        p = 1 / (1 + np.exp(-logit))
        x = (gen.random(n) < p).astype(int)
    else:
        x = (gen.random(n) < 0.5).astype(int)
    return np.array([smd(Z[:, k], x) for k in range(K)])


# ── (좌) 치우침은 그 변수가 선택에 들어가는 세기를 따라간다 ─────────
W = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2])
self_rows = np.empty((REPS, len(W)))
rand_rows = np.empty((REPS, len(W)))
for i in range(REPS):
    self_rows[i] = trial(N_PAGE, W, "self", rng)
    rand_rows[i] = trial(N_PAGE, W, "rand", rng)
self_m, self_s = self_rows.mean(0), self_rows.std(0, ddof=1)
rand_m, rand_s = rand_rows.mean(0), rand_rows.std(0, ddof=1)

# ── (우) 균형은 매번 맞는 것이 아니다 ──────────────────────────────
NS = [50, 200, 1000]
K_WATCH = 4                     # 왼쪽 그림의 세기 0.60 짜리 변수를 따라간다
spread = {n: np.array([trial(n, W, "rand", rng)[K_WATCH]
                       for _ in range(REPS)]) for n in NS}
self_big = {n: np.array([trial(n, W, "self", rng)[K_WATCH]
                         for _ in range(REPS)]) for n in NS}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))

ax1.errorbar(W, self_m, yerr=self_s, fmt="o-", ms=8, lw=2.4, capsize=5,
             color="tab:red", label="참가자가 스스로 골랐다")
ax1.errorbar(W, rand_m, yerr=rand_s, fmt="D-", ms=7, lw=2.4, capsize=5,
             color="tab:blue", label="동전으로 배정했다")
ax1.axhline(0, color="0.15", lw=1.6)
ax1.axhspan(-0.1, 0.1, color="tab:green", alpha=0.12)
ax1.text(1.19, 0.105, "흔히 쓰는 균형 기준 0.1", ha="right", va="bottom",
         fontsize=10.5, color="tab:green", fontweight="bold")
ax1.annotate("재지 않은 변수도 여기에 있다\n— 동전은 묻지 않는다",
             xy=(1.05, rand_m[-2]), xytext=(0.52, -0.30), fontsize=11,
             color="tab:blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
ax1.set_xlabel("그 변수가 선택에 들어가는 세기", fontsize=11)
ax1.set_ylabel("표준화 평균차 (처치군 빼기 대조군)", fontsize=11)
ax1.set_title(f"$n = {N_PAGE}$ 에서 {REPS:,} 번 되풀이한 평균 (수염은 표준편차)",
              fontsize=12.5)
ax1.set_ylim(-0.45, 0.95)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)

COLS = ["tab:blue", "tab:purple", "tab:cyan"]
bins = np.linspace(-0.75, 1.2, 70)
for n, col in zip(NS, COLS):
    over = (np.abs(spread[n]) > 0.1).mean()
    ax2.hist(spread[n], bins=bins, density=True, histtype="step", lw=2.4,
             color=col,
             label=f"동전 · $n = {n}$  (0.1 을 넘는 일 {100*over:.0f}%)")
ax2.hist(self_big[200], bins=bins, density=True, color="tab:red", alpha=0.25,
         label=f"스스로 고름 · $n = 200$  (평균 "
               f"{self_big[200].mean():+.3f})")
ax2.axvline(0, color="0.15", lw=1.8)
ax2.axvline(self_big[200].mean(), color="tab:red", lw=2.0, ls="--")
ax2.set_xlim(-0.6, 1.0)
ax2.set_xlabel("한 변수의 표준화 평균차", fontsize=11)
ax2.set_ylabel("밀도", fontsize=11)
ax2.set_title("균형은 매번 맞는 것이 아니라 치우치지 않는 것이다",
              fontsize=12.5)
ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax2.grid(alpha=0.25)

fig.suptitle("동전은 그 변수가 무엇인지 묻지 않는다 — 그래서 재지 않은 "
             "변수까지 맞춘다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "coin_does_not_ask.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N_PAGE} · {REPS:,} 번 되풀이 · 표준화 평균차의 평균")
print("   선택에 들어가는 세기   스스로 고름      동전")
for w, sm, rm in zip(W, self_m, rand_m):
    print(f"        {w:.2f}            {sm:+.4f}        {rm:+.4f}")
print(f"\n세기 {W[K_WATCH]:.2f} 짜리 변수 하나의 표준화 평균차가 흩어지는 폭")
for n in NS:
    v = spread[n]
    print(f"  동전 · n = {n:>4d}  평균 {v.mean():+.4f}  표준편차 "
          f"{v.std(ddof=1):.4f}  (2/sqrt(n) = {2/np.sqrt(n):.4f})  "
          f"· 0.1 을 넘는 일 {100*(np.abs(v) > 0.1).mean():.1f}%")
for n in NS:
    v = self_big[n]
    print(f"  스스로 고름 · n = {n:>4d}  평균 {v.mean():+.4f}  표준편차 "
          f"{v.std(ddof=1):.4f}   <- 중심이 옮겨 가지 않는다")
