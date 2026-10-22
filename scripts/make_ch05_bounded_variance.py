r"""유계이면 분산에 천장이 있다 — 분포를 모르고도 구간이 나오는 까닭.

이 쪽은 균등모집단에서 $S^2$ 의 흔들림이 카이제곱 예측보다 **작다**는 것을 본다.
좁은 것은 우연이 아니다. 자료가 $[a, b]$ 안에 갇혀 있으면 분산에 **천장**이 있고
($\operatorname{Var} \le (b-a)^2/4$), 그 천장이 분포를 가정하지 않고도 쓸 수 있는
구간을 만들어 준다.

① $[0,1]$ 위 분포들의 분산 — 균등은 $1/12$, 천장은 $1/4$.
② 그래서 분포를 몰라도 구간이 나온다 — 호에프딩 대 정규 근사.
③ 분산을 알면 좁아진다 — 경험적 번스타인.

만드는 파일: docs/ch05/applications/img/bounded_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
DELTA = 0.05

# === ① 천장 ==========================================================
print("① [0,1] 위 분포의 분산 — 천장은 1/4 = 0.25")
FAM = [("베타$(5,5)$", 5, 5), ("균등 = 베타$(1,1)$", 1, 1),
       ("베타$(0.5,0.5)$", 0.5, 0.5), ("베타$(0.1,0.1)$", 0.1, 0.1),
       ("베타$(0.02,0.02)$", 0.02, 0.02)]
VARS, LBL = [], []
for nm, a, b in FAM:
    v = a * b / ((a + b) ** 2 * (a + b + 1))
    VARS.append(v)
    LBL.append(nm)
    print(f"   {nm.replace('$',''):>22} : Var = {v:.5f}  "
          f"(천장의 {100 * v / 0.25:5.1f}%)")
print("   두 점(0 과 1)에 반씩 몰린 분포에서만 0.25 에 닿는다 — 포포비추의 부등식이다.")
print(f"   균등은 1/12 = {1/12:.5f} 로 천장의 {100 / 3:.1f}% 밖에 안 된다. "
      f"이 쪽의 S^2 이 좁은 까닭이 그것이다.")

# === ② 분포를 몰라도 =================================================
NS = [10, 20, 50, 100, 200, 500, 1000]
REP = 40_000
print(f"\n② [0,1] 자료의 95% 구간 — 포함률 (반복 {REP:,})")
print(f"{'n':>6}" + "".join(f"{k:>26}" for k in
                            ("정규근사(베타 0.05,3)", "호에프딩(같은 자료)")))
COV_N, COV_H, W_N, W_H = [], [], [], []
A3, B3 = 0.05, 3.0        # 아주 치우친 유계 분포
mu3 = A3 / (A3 + B3)
for n in NS:
    x = RNG.beta(A3, B3, (REP, n))
    xb = x.mean(1)
    sd = x.std(1, ddof=1)
    hw_n = stats.t.ppf(1 - DELTA / 2, n - 1) * sd / np.sqrt(n)
    hw_h = np.sqrt(np.log(2 / DELTA) / (2 * n))
    COV_N.append(float(np.mean(np.abs(xb - mu3) <= hw_n)))
    COV_H.append(float(np.mean(np.abs(xb - mu3) <= hw_h)))
    W_N.append(float(2 * hw_n.mean()))
    W_H.append(float(2 * hw_h))
    print(f"{n:>6}{COV_N[-1]:>26.4f}{COV_H[-1]:>26.4f}")
print("   정규근사는 작은 n 에서 95% 에 못 미치고, 호에프딩은 넘치게 지킨다. "
      "호에프딩은 분포를 전혀 쓰지 않았다.")

# === ③ 분산을 쓰면 좁아진다 ==========================================
print(f"\n③ 구간의 폭 (n = 200)")
N3 = 200
print(f"{'분포':>22}{'참 분산':>10}{'정규근사':>11}{'호에프딩':>11}"
      f"{'경험 번스타인':>14}")
WID = []
for nm, a, b in FAM:
    x = RNG.beta(a, b, (20_000, N3))
    sd = x.std(1, ddof=1)
    v = x.var(1, ddof=1)
    L = np.log(2 / DELTA)
    w_n = float(2 * (stats.t.ppf(1 - DELTA / 2, N3 - 1) * sd
                     / np.sqrt(N3)).mean())
    w_h = float(2 * np.sqrt(L / (2 * N3)))
    w_b = float(2 * (np.sqrt(2 * v * L / N3) + 7 * L / (3 * (N3 - 1))).mean())
    WID.append((w_n, w_h, w_b))
    print(f"{nm.replace('$',''):>22}{v.mean():>10.4f}{w_n:>11.4f}"
          f"{w_h:>11.4f}{w_b:>14.4f}")
print("   분산이 작은 분포에서는 경험 번스타인이 호에프딩보다 좁다. "
      "분산이 천장에 가까우면 둘이 비슷해진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_N, C_H, C_B = "#1f77b4", "#d62728", "#2ca02c"

xs = np.arange(len(FAM))
ax1.bar(xs, VARS, 0.58, color=["#9ecae1", "#4292c6", "#2171b5", "#08519c",
                               "#08306b"])
ax1.axhline(0.25, color=C_H, lw=2.0, ls="--")
ax1.text(0.0, 0.258, "천장 $(b-a)^2/4 = 0.25$ — 두 점에 반씩", color=C_H,
         fontsize=10.5)
ax1.axhline(1 / 12, color="#333333", lw=1.4, ls=":")
ax1.text(4.4, 1 / 12 + 0.008, "균등 $1/12$", color="#333333", fontsize=10,
         ha="right")
for i, v in enumerate(VARS):
    ax1.text(i, v + 0.007, f"${v:.3f}$", ha="center", fontsize=10)
ax1.set_xticks(xs)
ax1.set_xticklabels(LBL, fontsize=9, rotation=12)
ax1.set_ylabel("분산")
ax1.set_ylim(0, 0.30)
ax1.set_title("① 유계이면 천장이 있다", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

ax2.semilogx(NS, COV_N, "o-", color=C_N, lw=2.4, ms=8, label="정규근사 ($t$ 구간)")
ax2.semilogx(NS, COV_H, "s-", color=C_H, lw=2.4, ms=8, label="호에프딩")
ax2.axhline(0.95, color="#333333", lw=1.8, ls="--")
ax2.text(11, 0.955, "말한 $95\\%$", fontsize=10.5, color="#333333")
ax2.set_xlabel("표본크기 $n$")
ax2.set_ylabel("실제 포함률")
ax2.set_xticks([10, 100, 1000])
ax2.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$"])
ax2.minorticks_off()
ax2.set_ylim(0.60, 1.02)
ax2.set_title("② 분포를 몰라도 지켜진다 (베타 $0.05,\\,3$)", fontsize=12)
ax2.legend(loc="lower right", fontsize=10.5)
ax2.grid(alpha=0.25)

w = 0.27
wn = [t[0] for t in WID]
wh = [t[1] for t in WID]
wb = [t[2] for t in WID]
ax3.bar(xs - w, wn, w, color=C_N, label="정규근사")
ax3.bar(xs, wb, w, color=C_B, label="경험 번스타인")
ax3.bar(xs + w, wh, w, color=C_H, label="호에프딩")
ax3.set_xticks(xs)
ax3.set_xticklabels(LBL, fontsize=9, rotation=12)
ax3.set_ylabel(f"$95\\%$ 구간의 폭 ($n = {N3}$)")
ax3.set_title("③ 분산을 쓰면 좁아진다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch05/applications/img/bounded_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
