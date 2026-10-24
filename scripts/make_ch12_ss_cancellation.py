r"""계산용 공식이 음수인 제곱합을 내놓을 때 — 손으로 세는 자리의 함정.

이 쪽은 제곱합을 직접 구해 분산분석표를 채운다. 교과서가 흔히 주는 「계산용 공식」

    SS = sum(x^2) - (sum x)^2 / n

은 자료를 **한 번만** 훑으면 되어 손계산기 시절에 값졌다. 그런데 평균이 크면 두 큰
수를 빼는 자리에서 유효숫자가 통째로 사라진다. 모든 값에 상수를 더해도 제곱합은
수학적으로 **변하지 않는데**, 이 공식으로 계산하면 변한다.

① 상수를 키우며 SSW 를 두 방식으로 — float64.
② 그래서 F 통계량이 얼마나 틀리는가.
③ float32 에서는 **음수인 제곱합**이 나온다.

만드는 파일: docs/ch12/anova_one_way/img/ss_cancellation.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
K, N = 3, 50
MU = np.array([0.0, 0.4, 0.9])          # 참 평균이 조금씩 다르다
BASE = np.array([RNG.normal(MU[g], 1.0, N) for g in range(K)])
OFFSETS = np.array([1e0, 1e2, 1e4, 1e6, 1e7, 1e8, 1e9])


def ss_shortcut(x):
    """계산용 공식 — 자료를 한 번만 훑는다."""
    return (x ** 2).sum() - x.sum() ** 2 / x.size


def ss_twopass(x):
    """두 번 훑기 — 평균을 먼저 구하고 편차를 제곱한다."""
    return ((x - x.mean()) ** 2).sum()


def anova(groups, ss):
    """제곱합 계산을 ss 로 바꿔 끼워 F 를 구한다."""
    allx = np.concatenate(groups)
    sst = ss(allx)
    ssw = sum(ss(g) for g in groups)
    ssb = sst - ssw
    dfb, dfw = len(groups) - 1, allx.size - len(groups)
    return ssb / dfb / (ssw / dfw), ssw, ssb


true_F, true_ssw, _ = anova([g for g in BASE], ss_twopass)
print(f"집단 {K} 개 × {N} 개, 참 평균 {list(MU)}")
print(f"상수를 더하기 전: SSW = {true_ssw:.4f}, F = {true_F:.6f}")
print("상수를 더해도 제곱합과 F 는 수학적으로 **그대로여야 한다.**\n")

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("        더한 상수   계산용 SSW   두번훑기 SSW      계산용 F   두번훑기 F")
SW_S, SW_T, F_S, F_T = [], [], [], []
for c in OFFSETS:
    gs = [g + c for g in BASE]
    f_s, w_s, _ = anova(gs, ss_shortcut)
    f_t, w_t, _ = anova(gs, ss_twopass)
    SW_S.append(w_s); SW_T.append(w_t); F_S.append(f_s); F_T.append(f_t)
    print(f"   {c:>12.0e}   {w_s:>10.2f}   {w_t:>13.4f}   {f_s:>11.4f}   {f_t:>10.6f}")
print(f"\n   두번훑기는 상수가 10^9 이어도 F = {F_T[-1]:.6f} 로 꿈쩍하지 않는다.")
print(f"   계산용 공식은 10^8 에서 F = {F_S[-2]:.4f}, 10^9 에서 {F_S[-1]:.4f} 다.")

# === ③ float32 ======================================================
print("\n③ float32 로 같은 일을 하면")
print("        더한 상수   계산용 SSW (float32)")
SW32 = []
for c in OFFSETS:
    x32 = np.concatenate([g + c for g in BASE]).astype(np.float32)
    v = float((x32 ** 2).sum() - x32.sum() ** 2 / x32.size)
    SW32.append(v)
    flag = "   <- 음수다" if v < 0 else ""
    print(f"   {c:>12.0e}   {v:>20.1f}{flag}")
neg = [c for c, v in zip(OFFSETS, SW32) if v < 0]
if neg:
    print(f"   상수 {neg[0]:.0e} 에서는 **제곱합이 음수로 나온다** "
          f"({min(SW32):.1f}). 제곱을 더한 값인데 음수다.")
    print("   더 키우면 부호는 돌아오지만 값이 자릿수로 틀린다 — "
          "어느 쪽도 쓸 수 없다.")

print("\n   웰퍼드의 온라인 갱신은 한 번만 훑으면서도 이 문제가 없다.")
m = 0.0
s = 0.0
xx = np.concatenate([g + OFFSETS[-1] for g in BASE])
for i, v in enumerate(xx, 1):
    d = v - m
    m += d / i
    s += d * (v - m)
print(f"   상수 10^9 에서 웰퍼드가 낸 전체 제곱합 = {s:.4f} "
      f"(두번훑기 {ss_twopass(xx):.4f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_T, C_R = "#d62728", "#2ca02c", "#1f77b4"

ax1.loglog(OFFSETS, np.abs(SW_S), "o-", color=C_S, lw=2.8, ms=10,
           label="계산용 공식")
ax1.loglog(OFFSETS, SW_T, "s-", color=C_T, lw=2.8, ms=10, label="두 번 훑기")
ax1.axhline(true_ssw, color="#333333", lw=1.8, ls="--")
ax1.text(1.5, true_ssw * 1.25, f"참값 ${true_ssw:.0f}$", fontsize=10.5,
         color="#333333")
ax1.set_xlabel("모든 값에 더한 상수")
ax1.set_ylabel("집단 안 제곱합 $SS_W$")
ax1.set_title("① 상수를 더하면 달라진다 (float64)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.semilogx(OFFSETS, F_S, "o-", color=C_S, lw=2.8, ms=10, label="계산용 공식")
ax2.semilogx(OFFSETS, F_T, "s-", color=C_T, lw=2.8, ms=10, label="두 번 훑기")
ax2.axhline(true_F, color="#333333", lw=1.8, ls="--")
ax2.text(2e6, true_F * 1.6, f"참 $F = {true_F:.3f}$", fontsize=10.5,
         color="#333333")
ax2.set_xlabel("모든 값에 더한 상수")
ax2.set_ylabel("$F$ 통계량")
ax2.set_title("② 분산분석표가 통째로 틀어진다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10.5)
ax2.grid(alpha=0.25)

pos = [v if v > 0 else np.nan for v in SW32]
negv = [-v if v < 0 else np.nan for v in SW32]
ax3.loglog(OFFSETS, pos, "o-", color=C_S, lw=2.8, ms=10, label="양수로 나온 값")
ax3.loglog(OFFSETS, negv, "v", color="#333333", ms=15,
           label="음수로 나온 값 (부호를 뒤집어 그림)")
ax3.axhline(true_ssw, color="#333333", lw=1.8, ls="--")
ax3.set_xlabel("모든 값에 더한 상수")
ax3.set_ylabel("계산용 공식의 $SS$ (float32)")
ax3.set_title("③ float32 에서는 음수가 나온다", fontsize=12)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/anova_one_way/img/ss_cancellation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
