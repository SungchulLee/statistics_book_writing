"""10장 장 개요 들머리 — 한 번에 묻고, 그 다음에 이름을 부른다.

이 장이 있는 까닭을 그린다. 처음에 래너크셔 우유 실험(어긋난 출발선과
큰 표본)을 그렸는데, 그것은 12.1 F-검정 절차의 상자가 통째로 — 같은
인용·같은 수·같은 그림으로 — 다루는 이야기였다. 장 개요가 뒤쪽 상자의
축약본이 되어 있었고, 정작 **이 장의 물음**은 마지막 한 문장이었다.

여기서는 그 물음을 그린다. 장의 뼈대가 10.1(옴니버스 F 로 묻는다) →
10.3(그 다음 어느 쌍인가) → 10.5(두 절차가 기대는 가정)이므로 앞의 둘을
한 그림에 세운다. ch10 어디에도 다중비교의 동기를 그린 그림이 없다 —
10.1 의 정리하며와 연습문제가 수로만 적어 두었다.

왼쪽: 집단이 늘 때 쌍별 t-검정을 반복하면 가족단위 오류율(FWER)이 어디까지
  가는가. 세 곡선을 함께 둔다 — 모의실험 추정값, 흔히 인용되는 1 - 0.95^m
  상한, 그리고 옴니버스 F. 상한을 함께 두는 까닭은 **실제가 그보다 낮기**
  때문이다. 쌍별 비교들이 같은 집단과 합동분산 추정값을 공유해 양의 상관을
  가지며, 10.1 이 그 사실을 적어 두었다. 상한만 적으면 부풀린 수가 된다.

오른쪽: F 가 기각한 뒤가 남는다. 다섯 집단 가운데 **하나만** 모평균이 다른
  자료에서 F-검정의 검정력은 거의 1 인데, 보정하지 않은 쌍별 t-검정으로
  사후비교를 하면 **귀무가설이 참인 쌍**을 다섯 번에 한 번꼴로 유의하다고
  판정한다. 효과크기를 키워도 그 비율이 줄지 않는 것이 요점이다 —
  10.3 이 있는 까닭이다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import itertools

import numpy as np
from scipy import stats

OUT = "docs/ch10/img/"
M = 40_000
N = 10                      # 집단마다의 관측 수
rng = np.random.default_rng(1201)


def anova(x):
    """F 와 평균, 그리고 오차평균제곱을 한꺼번에."""
    m_, k_, n_ = x.shape[0], x.shape[1], x.shape[2]
    mu = x.mean(2)
    gm = x.mean((1, 2))
    ssb = n_ * ((mu - gm[:, None]) ** 2).sum(1)
    ssw = ((x - mu[:, :, None]) ** 2).sum((1, 2))
    dfw = k_ * (n_ - 1)
    mse = ssw / dfw
    return (ssb / (k_ - 1)) / mse, mu, mse, dfw


# ════════════════════════════════════════════════════════════════════
# 왼쪽 — 집단이 늘면 쌍별 t-검정의 반복이 FWER 을 놓친다
# ════════════════════════════════════════════════════════════════════
KS = list(range(2, 9))
rep_t, f_one, bound = [], [], []
print("귀무가설이 참 (모든 모평균이 같다) · 집단마다 "
      f"{N} 명 · 반복 {M:,} · 명목수준 0.05")
print("   k   m   쌍별 t 반복   옴니버스 F   독립일 때 상한")
for k in KS:
    x = rng.standard_normal((M, k, N))
    F, mu, mse, dfw = anova(x)
    se = np.sqrt(mse * 2 / N)
    tc = stats.t.ppf(0.975, dfw)
    hit = np.zeros(M, bool)
    for a, b in itertools.combinations(range(k), 2):
        hit |= np.abs(mu[:, a] - mu[:, b]) / se > tc
    npair = k * (k - 1) // 2
    rep_t.append(float(hit.mean()))
    f_one.append(float((F > stats.f.ppf(0.95, k - 1, dfw)).mean()))
    bound.append(1 - 0.95 ** npair)
    print(f"   {k}  {npair:3d}      {rep_t[-1]:.4f}     {f_one[-1]:.4f}"
          f"    {bound[-1]:.4f}")

# ════════════════════════════════════════════════════════════════════
# 오른쪽 — F 가 기각한 다음이 남는다
# ════════════════════════════════════════════════════════════════════
K2 = 5
DELTAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
NULLP = [(a, b) for a, b in itertools.combinations(range(K2), 2) if a and b]
f_rate, bad_t, bad_q = [], [], []
print(f"\n다섯 집단 가운데 하나만 모평균이 다른 자료 · 집단마다 {N} 명")
print("   효과크기   F 가 기각   그 가운데 귀무가설이 참인 쌍을 기각한 비율")
print("                         쌍별 t     투키 HSD")
for d in DELTAS:
    x = rng.standard_normal((M, K2, N))
    x[:, 0, :] += d
    F, mu, mse, dfw = anova(x)
    rej = F > stats.f.ppf(0.95, K2 - 1, dfw)
    se = np.sqrt(mse * 2 / N)
    tc = stats.t.ppf(0.975, dfw)
    qc = stats.studentized_range.ppf(0.95, K2, dfw) / np.sqrt(2)
    bt = np.zeros(M, bool)
    bq = np.zeros(M, bool)
    for a, b in NULLP:                      # 귀무가설이 참인 쌍만 본다
        z = np.abs(mu[:, a] - mu[:, b]) / se
        bt |= z > tc
        bq |= z > qc
    f_rate.append(float(rej.mean()))
    bad_t.append(float(bt[rej].mean()))
    bad_q.append(float(bq[rej].mean()))
    print(f"    {d:.1f}      {f_rate[-1]:.4f}      {bad_t[-1]:.4f}"
          f"     {bad_q[-1]:.4f}")

# ════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

ax1.plot(KS, bound, "s--", color="0.55", lw=2.0, ms=9,
         label="상한 1 - 0.95^m (쌍별 검정이 서로 독립일 때)")
ax1.plot(KS, rep_t, "o-", color="tab:red", lw=3.2, ms=12,
         label="쌍별 t-검정 반복 — 모의실험 추정값")
ax1.plot(KS, f_one, "^-", color="tab:blue", lw=3.2, ms=11,
         label="옴니버스 F-검정 (일원배치 분산분석)")
ax1.axhline(0.05, color="0.2", lw=2.0, ls=":")
ax1.text(KS[-1], 0.062, "명목수준 0.05", ha="right", va="bottom", fontsize=11,
         color="0.2", fontweight="bold")
ax1.annotate(f"k = 5 이면 비교가 m = 10 —\n"
             f"FWER 이 {rep_t[3]:.4f} 로 명목수준의 "
             f"{rep_t[3] / 0.05:.1f} 배다",
             xy=(5, rep_t[3]), xytext=(2.15, 0.52), fontsize=11.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.set_xticks(KS, [str(k) for k in KS], fontsize=12)
ax1.set_xlabel("집단의 수 k", fontsize=11.5)
ax1.set_ylabel("가족단위 오류율 FWER (모든 귀무가설이 참)", fontsize=11.5)
ax1.set_ylim(0, 0.82)
ax1.legend(fontsize=10.6, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_title("쌍별 t-검정의 반복은 FWER 을 통제하지 못한다", fontsize=12.5)

ax2.plot(DELTAS, f_rate, "-", color="0.6", lw=2.4,
         label="옴니버스 F-검정의 검정력")
ax2.plot(DELTAS, bad_t, "o-", color="tab:red", lw=3.2, ms=12,
         label="사후비교: 보정하지 않은 쌍별 t-검정")
ax2.plot(DELTAS, bad_q, "^-", color="tab:blue", lw=3.2, ms=11,
         label="사후비교: 투키 HSD")
ax2.axhline(0.05, color="0.2", lw=2.0, ls=":")
ax2.text(DELTAS[-1], 0.062, "명목수준 0.05", ha="right", va="bottom",
         fontsize=11, color="0.2", fontweight="bold")
i2 = DELTAS.index(2.0)
ax2.annotate(f"효과크기를 키워도 줄지 않는다 —\n"
             f"기각 사례의 {bad_t[i2]:.0%} 가 귀무가설이 참인 쌍이다",
             xy=(2.0, bad_t[i2]), xytext=(0.55, 0.90), fontsize=11.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax2.set_xlabel("한 집단의 모평균 이동 (표준편차 단위 효과크기)",
               fontsize=11.5)
ax2.set_ylabel("확률 (아래 두 곡선은 기각을 조건으로 한 FWER)",
               fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10.6, loc="center right")
ax2.grid(alpha=0.25)
ax2.set_title("옴니버스 F-검정은 어느 쌍이 다른지 식별하지 못한다",
              fontsize=12.5)

fig.suptitle("옴니버스 검정과 사후비교는 서로 다른 문제다",
             fontsize=14.5, y=0.99)
plt.tight_layout()
path = OUT + "one_question_then_which_pair.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("\nsaved", path)
