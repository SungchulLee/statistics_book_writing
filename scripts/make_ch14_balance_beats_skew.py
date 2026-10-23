r"""치우침보다 균형이다 — 두 표본 t 와 F 의 실제 수준.

이 쪽 절 6 은 이렇게 적는다. 「치우친 모집단에서는 작은 표본에서 t 검정의 제1종
오류율이 부풀려질 수 있다. 왜곡은 대체로 크지 않아, 명목 0.05 일 때 실제로
0.06-0.08 정도이다.」 그리고 로버스트한 조건으로 대칭 · n >= 20 · 꼬리를 든다.

재어 보면 두 가지가 다르다.

  - 균형 설계에서는 치우침이 아무리 세도 수준이 **올라가지 않고 내려간다**.
    왜도 31 에서 0.028 이다. 보수적이지 부풀려지지 않는다.
  - 목록에 없는 조건이 하나 있는데 그것이 제일 크다 — **균형**이다.

까닭은 정확하다. n1 = n2 이고 두 표본이 같은 분포에서 나오면, 둘을 맞바꾸는
것만으로 (Xbar1 - Xbar2) 가 부호를 바꾸므로 t 통계량의 귀무분포가 **F 가
무엇이든 정확히 0 에 대칭**이다. 두 꼬리가 같을 수밖에 없다.

어긋나면 그 보호가 사라진다. 그런데 양측 합은 멀쩡해 보인다 — 한쪽 꼬리가
사라진 만큼 다른 쪽이 커져서다.

① 왜도에 따른 양측 수준 — 균형과 불균형.
② 두 꼬리를 따로 — 합이 가리는 것.
③ 일원분산분석의 F 는 단측이라 가릴 데가 없다.

만드는 파일: docs/ch14/applications/img/balance_beats_skew.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats                                   # noqa: E402

RNG = np.random.default_rng(0)
REP = 40_000
SKP = [0.1, 0.3, 1.0, 2.5, 5.0, 8.0]          # 로그정규의 손잡이


def sigma(s):
    return np.sqrt(np.log(1 + s))


def pop_skew(s):
    """로그정규의 모집단 왜도."""
    e = np.exp(sigma(s) ** 2)
    return (e + 2) * np.sqrt(e - 1)


def two_sample(n1, n2, s, rng, rep=REP):
    """H0 가 참인 두 표본 (같은 로그정규). 합동분산 t."""
    sg = sigma(s)
    a = rng.lognormal(0, sg, (rep, n1))
    b = rng.lognormal(0, sg, (rep, n2))
    t, _ = stats.ttest_ind(a, b, axis=1)
    c = stats.t.ppf(0.975, n1 + n2 - 2)
    return float((t < -c).mean()), float((t > c).mean())


def anova(ns, s, rng, rep=REP):
    sg = sigma(s)
    g = [rng.lognormal(0, sg, (rep, n)) for n in ns]
    N, k = sum(ns), len(ns)
    allm = np.concatenate(g, axis=1).mean(1, keepdims=True)
    ssb = sum(n * (x.mean(1, keepdims=True) - allm) ** 2
              for n, x in zip(ns, g)).ravel()
    ssw = sum(((x - x.mean(1, keepdims=True)) ** 2).sum(1) for x in g)
    F = (ssb / (k - 1)) / (ssw / (N - k))
    return float((F > stats.f.ppf(0.95, k - 1, N - k)).mean())


DES = [((20, 20), "균형 $(20, 20)$"),
       ((10, 30), "$(10, 30)$"),
       ((5, 35), "$(5, 35)$")]
SK = np.array([pop_skew(s) for s in SKP])
LO = {d: [] for d, _ in DES}
HI = {d: [] for d, _ in DES}
for s in SKP:
    for d, _ in DES:
        lo, hi = two_sample(*d, s, RNG)
        LO[d].append(lo)
        HI[d].append(hi)
LO = {d: np.array(v) for d, v in LO.items()}
HI = {d: np.array(v) for d, v in HI.items()}

ADES = [((20, 20, 20), "균형 $(20,20,20)$"),
        ((10, 20, 30), "$(10,20,30)$"),
        ((5, 20, 35), "$(5,20,35)$")]
AL = {d: np.array([anova(d, s, RNG) for s in SKP]) for d, _ in ADES}

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"로그정규 모집단, H0 참, 되풀이 {REP:,}, 명목 수준 0.05")
print("\n① 두 표본 합동 t 의 양측 수준")
print("    모집단 왜도   (20,20)   (10,30)    (5,35)")
for i, s in enumerate(SKP):
    print(f"   {SK[i]:>11.2f}   {LO[(20,20)][i]+HI[(20,20)][i]:>7.4f}   "
          f"{LO[(10,30)][i]+HI[(10,30)][i]:>7.4f}   "
          f"{LO[(5,35)][i]+HI[(5,35)][i]:>7.4f}")
b0, b1 = LO[(20, 20)][0] + HI[(20, 20)][0], LO[(20, 20)][-1] + HI[(20, 20)][-1]
print(f"   균형 설계는 왜도가 세질수록 {b0:.4f} -> {b1:.4f} 로 **내려간다.**")
print(f"   불균형 (5,35) 는 {LO[(5,35)][0]+HI[(5,35)][0]:.4f} -> "
      f"{LO[(5,35)][-1]+HI[(5,35)][-1]:.4f} 로 오른다.")

print("\n② 두 꼬리를 따로 (명목은 각각 0.025)")
print("    설계        왜도    아래꼬리   위꼬리      비")
for d, _ in DES:
    for i in (2, len(SKP) - 1):
        r = HI[d][i] / max(LO[d][i], 1e-9)
        print(f"   {str(d):<10} {SK[i]:>6.2f}   {LO[d][i]:>7.4f}  "
              f"{HI[d][i]:>7.4f}   {r:>7.1f}")
print("   **균형이면 두 꼬리가 같다 — 운이 아니라 정리다.** 두 표본을 맞바꾸면")
print("   t 의 부호만 바뀌므로 귀무분포가 모집단이 무엇이든 0 에 대칭이다.")
print(f"   (5,35) 왜도 {SK[-1]:.1f} 에서는 아래꼬리가 {LO[(5,35)][-1]:.4f} 로 "
      "사실상 사라진다.")
print("   그런데 **양측 합은 멀쩡해 보인다** — 사라진 만큼 반대쪽이 커져서다.")

print("\n③ 일원분산분석 F (단측이라 가릴 데가 없다)")
print("    모집단 왜도   (20,20,20)   (10,20,30)   (5,20,35)")
for i, s in enumerate(SKP):
    print(f"   {SK[i]:>11.2f}   {AL[(20,20,20)][i]:>10.4f}   "
          f"{AL[(10,20,30)][i]:>10.4f}   {AL[(5,20,35)][i]:>9.4f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
CS = ["#2ca02c", "#ff7f0e", "#d62728"]

ax1.axhspan(0.06, 0.08, color="#9467bd", alpha=0.13)
ax1.text(6.0, 0.0845, "이 쪽이 적은 $0.06$–$0.08$", fontsize=10,
         color="#9467bd")
for (d, lab), c in zip(DES, CS):
    ax1.semilogx(SK, LO[d] + HI[d], "o-", color=c, lw=2.8, ms=8, label=lab)
ax1.axhline(0.05, color="#333333", lw=2.2, ls="--")
ax1.text(1.4, 0.0515, "명목 $0.05$", fontsize=10.5, color="#333333")
ax1.set_xlabel("모집단 왜도 (로그정규)")
ax1.set_ylabel("양측검정의 실제 수준")
ax1.set_ylim(0, 0.095)
ax1.set_xticks([1, 3, 10, 30])
ax1.set_xticklabels(["$1$", "$3$", "$10$", "$30$"])
ax1.minorticks_off()
ax1.set_title("① 균형이면 내려가고 아니면 올라간다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

i_hi = len(SKP) - 1
xs = np.arange(len(DES))
w = 0.36
ax2.bar(xs - w / 2, [LO[d][i_hi] for d, _ in DES], width=w, color="#1f77b4",
        label="아래꼬리 $P(T < -t_{0.975})$")
ax2.bar(xs + w / 2, [HI[d][i_hi] for d, _ in DES], width=w, color="#d62728",
        label="위꼬리 $P(T > +t_{0.975})$")
ax2.axhline(0.025, color="#333333", lw=2.2, ls="--")
ax2.text(-0.42, 0.0265, "각 꼬리의 명목 $0.025$", fontsize=10,
         color="#333333")
for i, (d, _) in enumerate(DES):
    ax2.text(i, max(LO[d][i_hi], HI[d][i_hi]) + 0.003,
             f"합 ${LO[d][i_hi]+HI[d][i_hi]:.4f}$", ha="center", fontsize=10)
ax2.set_xticks(xs)
ax2.set_xticklabels([lab for _, lab in DES], fontsize=10)
ax2.set_ylabel("꼬리별 기각률")
ax2.set_ylim(0, 0.082)
ax2.set_title(f"② 왜도 ${SK[i_hi]:.0f}$ 에서 두 꼬리를 따로", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

for (d, lab), c in zip(ADES, CS):
    ax3.semilogx(SK, AL[d], "s-", color=c, lw=2.8, ms=8, label=lab)
ax3.axhline(0.05, color="#333333", lw=2.2, ls="--")
ax3.text(1.4, 0.0515, "명목 $0.05$", fontsize=10.5, color="#333333")
ax3.set_xlabel("모집단 왜도 (로그정규)")
ax3.set_ylabel("$F$ 검정의 실제 수준")
ax3.set_ylim(0.015, 0.075)
ax3.set_xticks([1, 3, 10, 30])
ax3.set_xticklabels(["$1$", "$3$", "$10$", "$30$"])
ax3.minorticks_off()
ax3.set_title("③ 분산분석에는 가릴 꼬리가 없다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch14/applications/img/balance_beats_skew.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
