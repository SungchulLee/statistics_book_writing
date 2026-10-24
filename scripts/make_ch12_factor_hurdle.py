r"""전략 여럿을 늘어놓고 가장 좋은 것을 고르면 — 금융이 올린 문턱.

이 쪽은 분산분석으로 포트폴리오·섹터·팩터를 견준다. 그 자리의 다중비교 문제를
금융학계가 수로 겪은 적이 있고, 결론이 「$t > 2$ 로는 모자라다」였다.

모두 **참 초과수익이 0 인** 전략 k 개를 만들어 놓고 재어 본다.

① 가장 좋아 보이는 하나를 고르면 t 가 얼마나 커지는가.
② 집단 전체의 1종오류를 5% 로 묶으려면 문턱이 얼마여야 하는가.
③ 분산분석의 F 는 그 문턱을 지키는데, 「제일 좋은 것」을 고르면 지키지 못한다.

만드는 파일: docs/ch12/practical_applications/img/factor_hurdle.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 한글 글꼴에 U+2212 가 없다

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
N_MONTH = 240          # 20 년치 월간 수익률
REP = 20_000
KS = [1, 2, 5, 10, 20, 50, 100, 316]


def max_t(k, rep=REP, n=N_MONTH):
    """참 평균이 모두 0 인 전략 k 개에서 t 의 최댓값."""
    out = np.empty(rep)
    step = max(1, 2_000_000 // (k * n))
    done = 0
    while done < rep:
        m = min(step, rep - done)
        x = RNG.standard_normal((m, k, n))
        t = x.mean(2) / (x.std(2, ddof=1) / np.sqrt(n))
        out[done:done + m] = t.max(1)
        done += m
    return out


# === ① 가장 좋아 보이는 하나 ========================================
print(f"① 참 초과수익이 모두 0 인 전략 k 개, 월간 {N_MONTH} 개 (반복 {REP:,})")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("        k   최대 t 중앙값   t > 2.0 일 확률")
MED, P2 = [], []
for k in KS:
    t = max_t(k)
    MED.append(float(np.median(t)))
    P2.append(float(np.mean(t > 2.0)))
    print(f"   {k:>6}   {MED[-1]:>13.3f}   {P2[-1]:>15.4f}")
print(f"   전략이 {KS[-1]} 개면 **아무 실력이 없어도** 가장 좋은 것의 t 가 "
      f"중앙값 {MED[-1]:.2f} 이고,")
print(f"   t > 2.0 이 나올 확률이 {P2[-1]:.3f} 다.")

# === ② 문턱을 얼마로 올려야 하는가 ==================================
print("\n② 집단 전체의 1종오류를 5% 로 묶는 문턱")
print("        k   모의로 찾은 문턱   시닥 근사   본페로니 근사")
THR, SID, BON = [], [], []
for k in KS:
    t = max_t(k)
    THR.append(float(np.quantile(t, 0.95)))
    SID.append(float(stats.t.ppf((0.95) ** (1 / k), N_MONTH - 1)))
    BON.append(float(stats.t.ppf(1 - 0.05 / k, N_MONTH - 1)))
    print(f"   {k:>6}   {THR[-1]:>16.3f}   {SID[-1]:>10.3f}   {BON[-1]:>13.3f}")
i316 = KS.index(316)
print(f"   팩터 {KS[i316]} 개를 모두 시험했다고 보면 문턱이 "
      f"{THR[i316]:.2f} 로 올라간다.")
print("   하비·류·주가 권한 3.0 은 이 단순한 보정이 아니라 거짓발견율과 "
      "검정 사이 상관까지 넣어 얻은 값이다.")

# === ③ 분산분석의 F 는 지킨다 =======================================
KS3 = [2, 3, 5, 10, 20, 50]
print(f"\n③ 참 평균이 모두 같을 때의 1종오류 (반복 {REP:,})")
print("        k   일원배치 F   「제일 좋은 것」을 골라 t 검정")
F_RATE, PICK = [], []
for k in KS3:
    x = RNG.standard_normal((REP, k, N_MONTH))
    g = x.mean(2)
    sw = x.var(2, ddof=1).mean(1)                    # 집단 안 분산
    sb = N_MONTH * g.var(1, ddof=1)                  # 집단 간 분산
    f = sb / sw
    F_RATE.append(float(np.mean(f > stats.f.ppf(0.95, k - 1, k * (N_MONTH - 1)))))
    t = g / (x.std(2, ddof=1) / np.sqrt(N_MONTH))
    PICK.append(float(np.mean(t.max(1) > stats.t.ppf(0.975, N_MONTH - 1))))
    print(f"   {k:>6}   {F_RATE[-1]:>10.4f}   {PICK[-1]:>26.4f}")
print("   F 는 k 가 커져도 0.05 에 붙어 있고, 고르는 쪽은 k 와 함께 올라간다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

ax1.semilogx(KS, MED, "o-", color=C_A, lw=2.8, ms=10, label="가장 좋은 전략의 $t$ (중앙값)")
ax1.axhline(2.0, color="#333333", lw=2.0, ls="--")
ax1.text(1.15, 2.08, "흔히 쓰는 문턱 $t = 2$", fontsize=10.5, color="#333333")
ax1.plot([316], [MED[i316]], "o", color="#333333", ms=14, zorder=5)
ax1.annotate(f"팩터 $316$ 개\n$t = {MED[i316]:.2f}$", (316, MED[i316]),
             xytext=(-0.6, -2.4), textcoords="offset fontsize",
             fontsize=10.5, color="#333333", ha="right")
ax1.set_xlabel("늘어놓고 본 전략 수 $k$")
ax1.set_ylabel("가장 좋은 것의 $t$")
ax1.set_xticks([1, 10, 100])
ax1.set_xticklabels(["$1$", "$10$", "$10^{2}$"])
ax1.minorticks_off()
ax1.set_title("① 실력이 없어도 최고는 좋아 보인다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25)

ax2.semilogx(KS, THR, "o-", color=C_A, lw=2.8, ms=10, label="모의로 찾은 문턱")
ax2.semilogx(KS, BON, "s--", color=C_B, lw=2.2, ms=8, label="본페로니 근사")
ax2.axhline(2.0, color="#888888", lw=1.8, ls=":")
ax2.axhline(3.0, color=C_C, lw=2.4, ls="--")
ax2.text(1.15, 3.07, "하비·류·주가 권한 $3.0$", fontsize=10.5, color=C_C)
ax2.text(1.15, 1.82, "흔히 쓰는 $2.0$", fontsize=10.5, color="#888888")
ax2.set_xlabel("시험한 팩터 수 $k$")
ax2.set_ylabel("$5\\%$ 를 지키는 $t$ 문턱")
ax2.set_xticks([1, 10, 100])
ax2.set_xticklabels(["$1$", "$10$", "$10^{2}$"])
ax2.minorticks_off()
ax2.set_title("② 문턱은 시험한 수와 함께 올라간다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(KS3, F_RATE, "o-", color=C_C, lw=2.8, ms=10, label="일원배치 $F$")
ax3.plot(KS3, PICK, "s-", color=C_A, lw=2.8, ms=10,
         label="제일 좋은 것을 골라 $t$ 검정")
ax3.axhline(0.05, color="#333333", lw=1.8, ls="--")
ax3.text(2.4, 0.075, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax3.set_xlabel("견준 전략 수 $k$")
ax3.set_ylabel("$1$ 종오류")
ax3.set_ylim(0, 1.02)
ax3.set_title("③ 묶어서 묻는 것과 골라서 묻는 것", fontsize=12)
ax3.legend(loc="center right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/practical_applications/img/factor_hurdle.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
