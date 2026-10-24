r"""$\tau$ 는 **모든 쌍을 똑같이 세는데**, 쌍의 대부분은 애초에 쉽다.

모형 $20$ 개의 참 점수를 리더보드꼴로 둔다 — 위는 촘촘하고 아래로 갈수록 벌어진다.
자동 지표는 참 점수에 잡음 $N(0, \sigma^2)$ 를 더해 잰다고 하자.

① 차이가 큰 쌍은 거의 언제나 맞고, 차이가 작은 쌍은 동전 던지기에 가깝다. 그런데
   쌍의 수는 큰 차이 쪽에 몰려 있다.
② 그래서 상위 몇 개만 떼어 보면 $\tau$ 가 급격히 떨어진다.
③ 인접한 쌍만 보면, **우리가 알고 싶은 자리가 정확히 가장 어려운 자리**다.

만드는 파일: docs/ch10/correlation_test/img/tau_easy_pairs.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

M = 20                                  # 리더보드에 오른 모형 수
TOP, A, P = 89.0, 0.40, 1.30            # 참 점수 = TOP - A*rank^P
SIGMA = 1.5                             # 자동 지표의 잡음 (점수 단위)
REPS = 20000
KS = np.arange(2, M + 1)
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20)]

idx = np.arange(M)
q = TOP - A * (idx ** P)
print(f"참 점수 ({M} 개): {np.round(q, 2)}")
print(f"상위 다섯의 간격: {np.round(-np.diff(q[:6]), 2)}")

pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
gaps = np.array([q[a] - q[b] for a, b in pairs])
corr = np.zeros(len(pairs))
adj = np.zeros(M - 1)
tau_all, tau_k = [], {k: [] for k in KS}
for _ in range(REPS):
    s = q + RNG.normal(0.0, SIGMA, M)
    tau_all.append(stats.kendalltau(s, q).statistic)
    for k in KS:
        tau_k[k].append(stats.kendalltau(s[:k], q[:k]).statistic)
    corr += np.array([(s[a] - s[b]) > 0 for a, b in pairs])
    adj += np.array([(s[i] - s[i + 1]) > 0 for i in range(M - 1)])
corr /= REPS
adj /= REPS
tau_mean = float(np.mean(tau_all))
tau_k = {k: float(np.mean(v)) for k, v in tau_k.items()}

print(f"\n전체 켄달 타우의 평균 {tau_mean:.3f}")
print(f"\n① 참 차이 크기별")
print(f"{'차이':>10}{'쌍 수':>7}{'비율':>8}{'맞힐 확률':>11}")
cnt, acc = [], []
for lo, hi in BINS:
    m = (gaps >= lo) & (gaps < hi)
    cnt.append(int(m.sum()))
    acc.append(float(corr[m].mean()) if m.sum() else np.nan)
    print(f"{f'{lo}-{hi}':>10}{cnt[-1]:>7}{100 * cnt[-1] / len(pairs):>7.1f}%"
          f"{acc[-1]:>11.3f}")
big = gaps >= 5
print(f"   차이가 {5} 이상인 쌍이 전체의 {100 * big.mean():.0f}% 이고 "
      f"그 쌍들의 정확도가 {corr[big].mean():.3f} 다")

print(f"\n② 상위 k 개만 떼어 재면")
for k in (20, 15, 10, 8, 5, 3, 2):
    print(f"   k = {k:>2} : tau {tau_k[k]:.3f}")

print(f"\n③ 인접한 쌍 (i, i+1) 의 순서를 맞힐 확률")
for i in (0, 1, 2, 4, 9, 14, 18):
    print(f"   {i + 1:>2} 위 대 {i + 2:>2} 위 : {adj[i]:.3f}  "
          f"(참 차이 {q[i] - q[i + 1]:.2f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
labels = [f"${lo}$–${hi}$" for lo, hi in BINS]
xs = np.arange(len(BINS))

ax1b = ax1.twinx()
ax1b.bar(xs, cnt, 0.62, color="#cccccc")
ax1b.set_ylabel("그 구간의 쌍 수", color="#888888")
ax1b.tick_params(axis="y", labelcolor="#888888")
ax1b.set_ylim(0, max(cnt) * 1.55)
ax1.plot(xs, acc, "o-", color="#d62728", lw=2.8, ms=9, zorder=5)
for x, a in zip(xs, acc):
    ax1.text(x, a + 0.028, f"${a:.2f}$", ha="center", fontsize=10.5,
             color="#d62728", zorder=6)
ax1.axhline(0.5, color="black", lw=1.3, ls=":")
ax1.text(0.05, 0.525, "동전 던지기", fontsize=10)
ax1.set_zorder(ax1b.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.set_xticks(xs)
ax1.set_xticklabels(labels, fontsize=9.5)
ax1.set_xlabel("두 모형의 참 점수 차이")
ax1.set_ylabel("그 쌍의 순서를 맞힐 확률", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.set_ylim(0.42, 1.08)
ax1.set_title("① 쉬운 쌍이 대부분이다", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(KS, [tau_k[k] for k in KS], "o-", color="#1f77b4", lw=2.8, ms=8)
ax2.axhline(tau_mean, color="#2ca02c", lw=2.0, ls="--",
            label=f"전체 $\\tau = {tau_mean:.3f}$")
ax2.axhline(0.0, color="black", lw=1.3, ls=":")
for k in (5, 3):
    ax2.annotate(f"$k = {k}$ 이면 ${tau_k[k]:.2f}$", xy=(k, tau_k[k]),
                 xytext=(k + 2.4, tau_k[k] - 0.17), fontsize=10.5,
                 color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xticks([2, 5, 10, 15, 20])
ax2.set_xticklabels(["$2$", "$5$", "$10$", "$15$", "$20$"])
ax2.set_xlabel("상위 $k$ 개만 떼어 쟀을 때")
ax2.set_ylabel("켄달 $\\tau$")
ax2.set_ylim(-0.08, 1.02)
ax2.set_title("② 위로 갈수록 무너진다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(np.arange(1, M), adj, "o-", color="#1f77b4", lw=2.8, ms=8)
ax3.axhline(0.5, color="black", lw=1.3, ls=":")
ax3.text(13.5, 0.525, "동전 던지기", fontsize=10)
ax3.annotate(f"$1$ 위 대 $2$ 위\n${adj[0]:.3f}$", xy=(1, adj[0]),
             xytext=(4.2, 0.90), fontsize=11.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.set_xticks([1, 5, 10, 15, 19])
ax3.set_xticklabels(["$1$–$2$", "$5$–$6$", "$10$–$11$", "$15$–$16$",
                     "$19$–$20$"], fontsize=9.5)
ax3.set_xlabel("견주는 두 자리")
ax3.set_ylabel("그 순서를 맞힐 확률")
ax3.set_ylim(0.45, 1.03)
ax3.set_title("③ 궁금한 자리가 가장 어렵다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch10/correlation_test/img/tau_easy_pairs.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
