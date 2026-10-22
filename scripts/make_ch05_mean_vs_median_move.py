r"""평균을 올리는 일과 중앙값을 올리는 일은 다르다 — 치우친 지표에서.

이 쪽 $\S 6$ 은 「평균과 중앙값 중 무엇을 쓸 것인가」를 묻는다. 치우친 지표에서
그 물음은 보고 양식의 문제가 아니라 **무엇을 목표로 삼을 것인가**의 문제다.
둘은 같은 방향으로 움직이지 않고, 때로는 반대로 움직인다.

① 같은 양을 누구에게 주느냐에 따라 평균과 중앙값이 다르게 움직인다.
② 평균은 오르는데 중앙값은 내려가게 만들 수 있다.
③ 치우침이 심할수록 둘이 더 크게 갈린다.

만드는 파일: docs/ch05/non_normal/img/mean_vs_median_move.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
N = 200_000
SIG = 1.2                                   # 로그정규의 로그 표준편차
X = RNG.lognormal(0, SIG, N)
X = X / X.mean()                            # 평균을 1 로 맞춘다
M0, Q0 = float(X.mean()), float(np.median(X))
print(f"바탕 지표 — 로그정규(sigma = {SIG}), n = {N:,}")
print(f"   평균 {M0:.4f},  중앙값 {Q0:.4f},  "
      f"상위 1% 가 합계의 {100 * np.sort(X)[-N // 100:].sum() / X.sum():.1f}%")

# === ① 같은 양을 누구에게 ============================================
BUDGET = 0.05                               # 전체 합계의 5% 를 더 쓰게 한다
total = X.sum()
order = np.argsort(X)
print(f"\n① 합계의 {100 * BUDGET:.0f}% 를 더 쓰게 만든다 — 누구에게 주느냐")
print(f"{'받는 사람':>18}{'평균 변화':>12}{'중앙값 변화':>14}")
CASES = []
for nm, lo, hi in (("상위 $1\\%$ 에게", 99, 100), ("상위 $10\\%$ 에게", 90, 100),
                   ("가운데 $20\\%$ 에게", 40, 60),
                   ("하위 $50\\%$ 에게", 0, 50)):
    y = X.copy()
    idx = order[int(N * lo / 100):int(N * hi / 100)]
    y[idx] += BUDGET * total / len(idx)
    dm = float(y.mean()) / M0 - 1
    dq = float(np.median(y)) / Q0 - 1
    CASES.append((nm, dm, dq))
    print(f"{nm.replace(chr(92)+'%','%').replace('$',''):>18}"
          f"{100 * dm:>11.2f}%{100 * dq:>13.2f}%")
print("   평균은 누구에게 주든 똑같이 5% 오른다. 중앙값은 받는 사람이 "
      "가운데를 지날 때만 움직인다.")

# === ② 반대로 움직이게 ===============================================
print(f"\n② 평균은 올리고 중앙값은 내리기")
y = X.copy()
top = order[int(N * 0.99):]
bot = order[:int(N * 0.60)]
y[top] += 0.10 * total / len(top)           # 상위 1% 에게 합계의 10%
y[bot] *= 0.97                              # 하위 60% 는 3% 줄인다
dm2 = float(y.mean()) / M0 - 1
dq2 = float(np.median(y)) / Q0 - 1
print(f"   상위 1% 에게 합계의 10% 를 더 주고 하위 60% 를 3% 깎으면")
print(f"   평균 {100 * dm2:+.2f}%,  중앙값 {100 * dq2:+.2f}%")
print(f"   합계도 {100 * (y.sum() / total - 1):+.2f}% 로 늘었다. "
      f"**지표 하나만 보면 좋아진 것으로 읽힌다.**")

# === ③ 치우침이 심할수록 =============================================
SIGS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8]
print(f"\n③ 치우침을 바꾸며 — '평균을 1% 올리는 데 드는 사람'")
print(f"{'sigma':>7}{'왜도':>9}{'상위 1% 가 가진 몫':>18}"
      f"{'평균/중앙값':>13}{'상위 1% 로 평균 1% 올리기':>26}")
SKEW, SHARE, RATIO, NEED = [], [], [], []
for sg in SIGS:
    z = RNG.lognormal(0, sg, N)
    z = z / z.mean()
    sk = float((np.exp(sg ** 2) + 2) * np.sqrt(np.exp(sg ** 2) - 1))
    sh = float(np.sort(z)[-N // 100:].sum() / z.sum())
    rt = float(z.mean() / np.median(z))
    # 상위 1% 의 값을 몇 배로 키워야 평균이 1% 오르는가
    need = 1 + 0.01 / sh
    SKEW.append(sk)
    SHARE.append(sh)
    RATIO.append(rt)
    NEED.append(need)
    print(f"{sg:>7.1f}{sk:>9.2f}{100 * sh:>17.1f}%{rt:>13.3f}"
          f"{need:>25.3f}배")
print("   치우칠수록 상위 1% 의 몫이 커져 '평균을 움직이는 일'이 그 1% 의 "
      "손에 들어간다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_M, C_Q = "#d62728", "#1f77b4"

xs = np.arange(len(CASES))
w = 0.38
ax1.bar(xs - w / 2, [100 * c[1] for c in CASES], w, color=C_M, label="평균")
ax1.bar(xs + w / 2, [100 * c[2] for c in CASES], w, color=C_Q, label="중앙값")
for i, c in enumerate(CASES):
    ax1.text(xs[i] - w / 2, 100 * c[1] + 1.0, f"${100 * c[1]:.1f}$",
             ha="center", fontsize=10, color=C_M)
    ax1.text(xs[i] + w / 2, 100 * c[2] + 1.0, f"${100 * c[2]:.1f}$",
             ha="center", fontsize=10, color=C_Q)
ax1.set_xticks(xs)
ax1.set_xticklabels([c[0] for c in CASES], fontsize=9)
ax1.set_ylabel("변화 (%)")
ax1.set_ylim(0, 52)
ax1.set_title(f"① 합계의 ${100 * BUDGET:.0f}\\%$ 를 누구에게", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25, axis="y")

ax2.bar([0, 1], [100 * dm2, 100 * dq2], 0.5, color=[C_M, C_Q])
for i, v in enumerate([100 * dm2, 100 * dq2]):
    ax2.text(i, v + (0.4 if v > 0 else -1.0), f"${v:+.2f}\\%$", ha="center",
             fontsize=12)
ax2.axhline(0, color="#333333", lw=1.4)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["평균", "중앙값"], fontsize=11.5)
ax2.set_ylabel("변화 (%)")
ax2.set_ylim(-5.0, 12.0)
ax2.set_title("② 상위 $1\\%$ 에 더 주고 하위 $60\\%$ 를 깎으면", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(SKEW, [100 * s for s in SHARE], "o-", color=C_M, lw=2.6, ms=9,
         label="상위 $1\\%$ 가 가진 몫")
ax3.set_xscale("log")
ax3.set_xlabel("왜도")
ax3.set_ylabel("상위 $1\\%$ 가 합계에서 차지하는 몫 (%)", color=C_M)
ax3.tick_params(axis="y", colors=C_M)
ax3b = ax3.twinx()
ax3b.plot(SKEW, RATIO, "s--", color=C_Q, lw=2.2, ms=8, label="평균 / 중앙값")
ax3b.set_ylabel("평균 / 중앙값", color=C_Q)
ax3b.tick_params(axis="y", colors=C_Q)
ax3.set_xticks([0.6, 2, 6, 20, 60])
ax3.set_xticklabels(["$0.6$", "$2$", "$6$", "$20$", "$60$"])
ax3.minorticks_off()
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
ax3.set_title("③ 치우칠수록 갈린다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/non_normal/img/mean_vs_median_move.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
