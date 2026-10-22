r"""최댓값을 추정하면 참 최댓값보다 커진다 — 이름이 붙은 MLE 편향.

이 쪽의 $\hat\sigma^2_{\mathrm{MLE}}$ 처럼, 잘 추정한 값들에 **비선형 함수**를
씌우면 불편성이 깨진다. 함수가 $\max$ 일 때 그 편향에는 이름이 있다 —
강화학습의 **최대화 편향**이다. 행동 $m$ 개의 가치를 각각 맞게 추정해도
그 가운데 가장 큰 것은 참 최댓값보다 **위로** 치우친다.

① 반 하셀트·게즈·실버(2016) 그림 1 을 다시 그린다 — 단일 추정과 이중 추정.
② 오차가 균등일 때 편향은 $(m-1)/(m+1)$ 과 맞는다 (같은 논문의 결과).
③ 표본을 늘려도 편향/표준오차 비는 그대로다 — 편향이 잡음에 묻히지 않는다.

만드는 파일: docs/ch06/mle/img/maximization_bias.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
REP = 200_000
MS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def one_round(m, rep, draw):
    """참 가치는 모두 0. Q 와 Q' 를 똑같이 따로 뽑는다."""
    q = draw((rep, m))
    qp = draw((rep, m))
    j = q.argmax(axis=1)
    single = q.max(axis=1)                      # max_a Q(s,a)
    double = qp[np.arange(rep), j]              # Q'(s, argmax_a Q(s,a))
    return single.mean(), double.mean()


# === ① 표준정규 오차 ==================================================
print(f"① 오차가 표준정규일 때 ({REP:,} 번 평균)")
print(f"{'행동 수 m':>9}{'단일 추정':>12}{'이중 추정':>12}{'sqrt(2 ln m) 위 어림':>22}")
S1, D1 = [], []
for m in MS:
    s, d = one_round(m, REP, lambda sh: RNG.standard_normal(sh))
    S1.append(s)
    D1.append(d)
    print(f"{m:>9}{s:>12.4f}{d:>12.4f}{np.sqrt(2 * np.log(m)):>14.4f}")
print("   단일 추정은 m 이 커질수록 위로 벌어지고 이중 추정은 0 에 붙어 있다.")

# === ② 균등 오차와 (m-1)/(m+1) ========================================
print(f"\n② 오차가 균등 [-1, 1] 일 때 — 논문의 (m-1)/(m+1)")
print(f"{'행동 수 m':>9}{'모의한 편향':>14}{'(m-1)/(m+1)':>14}{'차':>10}")
S2, F2 = [], []
for m in MS:
    s, _ = one_round(m, REP, lambda sh: RNG.uniform(-1, 1, sh))
    f = (m - 1) / (m + 1)
    S2.append(s)
    F2.append(f)
    print(f"{m:>9}{s:>14.5f}{f:>14.5f}{s - f:>10.5f}")
print(f"   가장 큰 차가 {max(abs(a - b) for a, b in zip(S2, F2)):.5f} 다 — "
      f"식과 모의가 맞는다.")

# === ③ 표본을 늘리면 ==================================================
NS = np.array([10, 30, 100, 300, 1000, 3000, 10000])
M3 = [2, 10, 100]
print(f"\n③ 행동마다 n 번 재어 평균한다 (오차의 표준편차 = 1/sqrt(n))")
print(f"{'n':>7}" + "".join(f"{f'm={m} 편향':>12}" for m in M3)
      + "".join(f"{f'm={m} 편향/SE':>13}" for m in M3))
B3 = {m: [] for m in M3}
R3 = {m: [] for m in M3}
for n in NS:
    row = []
    for m in M3:
        s, _ = one_round(m, 60_000,
                         lambda sh: RNG.standard_normal(sh) / np.sqrt(n))
        B3[m].append(s)
        R3[m].append(s * np.sqrt(n))        # 표준오차 1/sqrt(n) 로 나눈 값
        row.append(s)
    print(f"{n:>7}" + "".join(f"{v:>12.5f}" for v in row)
          + "".join(f"{R3[m][-1]:>13.3f}" for m in M3))
print("   편향은 1/sqrt(n) 로 줄지만 표준오차도 같은 속도로 준다 — "
      "비는 n 이 백 배가 되어도 그대로다.")
for m in M3:
    print(f"   m = {m:>3} : 비의 평균 {np.mean(R3[m]):.3f} "
          f"(위 어림 sqrt(2 ln m) = {np.sqrt(2 * np.log(m)):.3f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_D, C_T = "#ff7f0e", "#1f77b4", "#2ca02c"

xs = np.arange(len(MS))
w = 0.38
ax1.bar(xs, S1, 0.62, color=C_S, label="단일 추정 — 가장 큰 것을 쓴다")
ax1.plot(xs, [np.sqrt(2 * np.log(m)) for m in MS], "--", color=C_T, lw=2.0,
         label="$\\sqrt{2\\ln m}$ — 위 어림")
ax1.axhline(0, color="#666666", lw=1.2)
ax1.plot(xs, D1, "o-", color=C_D, lw=2.2, ms=8, zorder=5,
         label="이중 추정 — 고르는 자료와 재는 자료를 나눈다")
ax1.annotate(f"이중 추정은 모두 $|\\cdot| < "
             f"{max(abs(v) for v in D1):.3f}$", xy=(0.5, 0.22),
             xytext=(1.0, -0.62), textcoords="data", fontsize=10.5,
             color=C_D,
             arrowprops=dict(arrowstyle="->", color=C_D, lw=1.1))
ax1.set_ylim(-0.95, 4.0)
ax1.set_xticks(xs)
ax1.set_xticklabels([f"${m}$" for m in MS], fontsize=9)
ax1.set_xlabel("행동의 수 $m$")
ax1.set_ylabel("참 최댓값과의 차 (참값은 $0$)")
ax1.set_title("① 오차가 표준정규일 때", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(xs, S2, "o", color=C_S, ms=9, label="모의한 편향")
ax2.plot(xs, F2, "-", color=C_T, lw=2.4, label="$(m-1)/(m+1)$")
ax2.set_xticks(xs)
ax2.set_xticklabels([f"${m}$" for m in MS], fontsize=9)
ax2.set_xlabel("행동의 수 $m$")
ax2.set_ylabel("참 최댓값과의 차")
ax2.set_ylim(0, 1.08)
ax2.annotate(f"가장 큰 차 "
             f"${max(abs(a - b) for a, b in zip(S2, F2)):.5f}$",
             xy=(0.05, 0.9), xycoords="axes fraction", fontsize=11,
             color="#555555")
ax2.set_title("② 오차가 균등 $[-1,\\,1]$ 일 때", fontsize=12)
ax2.legend(loc="lower right", fontsize=10.5)
ax2.grid(alpha=0.25)

COL3 = {2: "#9ecae1", 10: "#4292c6", 100: "#08519c"}
for m in M3:
    ax3.loglog(NS, B3[m], "o-", color=COL3[m], lw=2.4, ms=8,
               label=f"$m = {m}$")

ax3.set_xlabel("행동마다 재어 본 횟수 $n$")
ax3.set_ylabel("최댓값의 편향")
ax3.set_xticks([10, 100, 1000, 10000])
ax3.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$", "$10^{4}$"])
ax3.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax3.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
ax3.minorticks_off()
ax3.set_ylim(2e-3, 3)
ax3.annotate("편향 / 표준오차\n"
             + "\n".join(f"$m = {m}$ :  ${np.mean(R3[m]):.2f}$" for m in M3),
             xy=(0.97, 0.97), xycoords="axes fraction", fontsize=10.5,
             ha="right", va="top", color="#333333")
ax3.set_title("③ 더 재어도 비는 그대로다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/mle/img/maximization_bias.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
