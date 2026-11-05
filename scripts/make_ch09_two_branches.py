"""9.3 이표본 t — 자료를 보고 갈라서면 **두 갈래 다** 틀린다.

왜 이 그림인가. 「먼저 등분산을 검정하고 고른다」를 재는 글은 거의 언제나
**전체 평균**만 적는다. 그 수는 꽤 멀쩡해 보인다 — $\\sigma$ 비 $3$ 에서
$0.054$ 다. 그런데 그 평균은 **서로 반대로 틀린 두 갈래**를 섞은 것이다.

통과해서 합동 $t$ 로 간 표본에서는 $0.353$ 이고, 기각해서 웰치로 간
표본에서는 $0.046$ 이다. 평균만 보면 둘 다 보이지 않는다.

더 고약한 것이 왼쪽 칸의 두 곡선이다. **거르고 나면 합동 $t$ 가 더
나빠진다.** 사전검정을 통과했다는 것은 분산이 같다는 뜻이 아니라 **큰 분산
쪽이 마침 작게 나왔다**는 뜻이고, 그러면 합동분산이 참값을 더 심하게
과소평가한다. 거름망이 하필 합동 $t$ 가 가장 틀리는 표본을 골라 준다.

그리고 $\\sigma$ 비가 **정확히 1** 인 왼쪽 끝을 보라. 분산이 같은데도
기각 갈래의 오류율이 $0.088$ 이다. 잘못 기각된 $5\\%$ 는 분산비가 극단으로
나온 표본들이고, 그 표본에서는 **웰치마저** 명목값을 지키지 못한다.
망가뜨리는 것은 「틀린 검정을 골랐다」가 아니라 **「보았다」**는 것이다.

실행:  python3 scripts/make_ch09_two_branches.py   (저장소 최상위에서)
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch09/two_sample_tests/img/"
N1, N2 = 10, 30
REP = 400_000
ALPHA = 0.05

ratios = np.arange(1.0, 3.01, 0.125)
lo = stats.f.ppf(ALPHA / 2, N1 - 1, N2 - 1)
hi = stats.f.ppf(1 - ALPHA / 2, N1 - 1, N2 - 1)

rng = np.random.default_rng(20261105)
pass_pool, rej_welch, whole, pooled_all, pooled_pass, rate = [], [], [], [], [], []
for r in ratios:
    a = rng.normal(0, r, (REP, N1))
    b = rng.normal(0, 1, (REP, N2))
    pp = stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue < ALPHA
    pw = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < ALPHA
    F = a.var(1, ddof=1) / b.var(1, ddof=1)
    passed = (F >= lo) & (F <= hi)
    pass_pool.append(pp[passed].mean())
    rej_welch.append(pw[~passed].mean())
    whole.append(np.where(passed, pp, pw).mean())
    pooled_all.append(pp.mean())
    pooled_pass.append(pp[passed].mean())
    rate.append(passed.mean())

A = [np.array(v) for v in (pass_pool, rej_welch, whole, pooled_all, pooled_pass, rate)]
pass_pool, rej_welch, whole, pooled_all, pooled_pass, rate = A

print(f"n = ({N1}, {N2}) · 반복 {REP:,} · 명목 α = {ALPHA}")
# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print(" σ비  통과→합동  기각→웰치  두 단계 전체   통과 비율")
for i, r in enumerate(ratios):
    if r in (1.0, 1.25, 1.5, 2.0, 3.0):
        print(f"{r:5.2f}{pass_pool[i]:10.4f}{rej_welch[i]:10.4f}"
              f"{whole[i]:13.4f}{rate[i]:12.4f}")

# ════════════════════════════════════════════════════════════════════
RED, BLUE, ORANGE, GREY = "#C62828", "#1565C0", "#EF6C00", "#607D8B"
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)

ax = axes[0]
ax.axhline(ALPHA, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(2.72, ALPHA + 0.008, "명목 0.05", color="0.35", fontsize=10.5, ha="center")
ax.plot(ratios, pass_pool, color=RED, lw=2.6, label="통과했다 → 합동 $t$")
ax.plot(ratios, rej_welch, color=BLUE, lw=2.6, label="기각했다 → 웰치")
ax.plot(ratios, whole, color=ORANGE, lw=2.6, ls=(0, (6, 3)),
        label="두 갈래를 섞은 전체")
ax.annotate(f"{pass_pool[-1]:.3f}", xy=(ratios[-1], pass_pool[-1]),
            xytext=(ratios[-1] - 0.45, pass_pool[-1] + 0.02), color=RED, fontsize=11)
ax.annotate(f"{rej_welch[0]:.3f}", xy=(ratios[0], rej_welch[0]),
            xytext=(ratios[0] + 0.1, rej_welch[0] + 0.03), color=BLUE, fontsize=11,
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
ax.set_xlabel("작은 집단의 표준편차 배수  $\\sigma_1/\\sigma_2$", fontsize=11.5)
ax.set_ylabel("실제 제1종 오류율", fontsize=11.5)
ax.set_title("(가) 두 갈래를 따로 보면 어느 쪽도 0.05 가 아니다", fontsize=12.6)
ax.set_ylim(0, 0.40)
ax.legend(fontsize=10.4, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[1]
ax.axhline(ALPHA, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.plot(ratios, pooled_all, color=GREY, lw=2.6, label="합동 $t$ — 모든 표본")
ax.plot(ratios, pooled_pass, color=RED, lw=2.6, label="합동 $t$ — 통과한 표본만")
ax.fill_between(ratios, pooled_all, pooled_pass, color=RED, alpha=0.14)
# matplotlib 는 마크다운을 모른다 — **굵게** 는 별표가 글자로 찍힌다.
# 띠 안에 직접 적고 화살표를 쓰지 않는다.
ax.text(2.58, 0.268, "거를수록 벌어진다", color=RED, fontsize=11.8,
        ha="center", va="center", rotation=27)
ax.set_xlabel("작은 집단의 표준편차 배수  $\\sigma_1/\\sigma_2$", fontsize=11.5)
ax.set_ylabel("실제 제1종 오류율", fontsize=11.5)
ax.set_title("(나) 거름망은 합동 $t$ 가 가장 틀리는 표본을 고른다", fontsize=12.6)
ax.set_ylim(0, 0.40)
ax.legend(fontsize=10.4, loc="upper left")
ax.grid(alpha=0.25)

fig.suptitle("자료를 보고 갈라서면 두 갈래 다 틀린다 — 전체 평균이 그것을 가린다",
             fontsize=14.4, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.95))
path = OUT + "two_branches.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
