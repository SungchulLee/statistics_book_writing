r"""분산분석이 태어난 자리 — 밭이 아니라 유전학 논쟁이었다.

$1918$ 년 피셔가 푼 문제는 이것이다. 멘델의 유전은 **알갱이**인데 키나 몸무게는
**이어져** 있고, 친척끼리 상관이 있다. 둘이 어떻게 같이 성립하는가.

답은 작은 유전자를 많이 더하는 것이었고, 그것을 보이려면 **총변동을 원인별 몫으로
쪼개야** 했다. 그 몫을 재려고 그가 표준편차의 제곱에 붙인 이름이 **분산**이다.

① 작은 멘델 유전자를 더하면 이어진 형질이 된다.
② 친척 사이의 상관이 혈연의 몫 그대로 나온다.
③ 총분산이 원인별로 정확히 쪼개진다 — 이것이 분산분석이다.

만드는 파일: docs/ch12/anova_one_way/img/fisher1918.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)
K = 50            # 유전자 자리 수
P = 0.5           # 대립유전자 빈도
N = 200_000       # 가계 수
H2 = 0.5          # 유전율 — 환경 분산을 여기에 맞춰 정한다


def haplo(n, k=K, rng=RNG):
    """집단에서 뽑은 반수체 하나 (자리마다 0/1)."""
    return (rng.random((n, k)) < P).astype(np.int8)


def child(h_from_dad, h_from_mom):
    return h_from_dad, h_from_mom


def transmit(h1, h2, rng=RNG):
    """감수분열 — 자리마다 두 반수체 가운데 하나를 독립적으로 물려준다."""
    pick = rng.random(h1.shape) < 0.5
    return np.where(pick, h1, h2).astype(np.int8)


def gvalue(h1, h2):
    """가산 유전값 = 대립유전자 수의 합."""
    return (h1 + h2).sum(axis=1).astype(np.float64)


# --- 유전 분산에 맞추어 환경 분산을 정한다 ---------------------------
VAR_G = K * 2 * P * (1 - P)          # = 25
VAR_E = VAR_G * (1 - H2) / H2


def pheno(g, rng=RNG):
    return g + rng.normal(0, np.sqrt(VAR_E), g.shape)


# === 가계를 만든다 ==================================================
# 조부모 한 쌍 -> 자식 둘(A, B) -> 각자 바깥에서 배우자를 얻어 손자 C, D
gp1 = (haplo(N), haplo(N))
gp2 = (haplo(N), haplo(N))
A = (transmit(*gp1), transmit(*gp2))        # 자식 A
B = (transmit(*gp1), transmit(*gp2))        # 자식 B — A 와 친형제
spA = (haplo(N), haplo(N))
spB = (haplo(N), haplo(N))
C = (transmit(*A), transmit(*spA))          # A 의 자식
D = (transmit(*B), transmit(*spB))          # B 의 자식 — C 와 사촌
# 반형제: A 가 다른 배우자와 낳은 자식
spA2 = (haplo(N), haplo(N))
C2 = (transmit(*A), transmit(*spA2))        # C 와 아버지만 같다

G = {k: gvalue(*v) for k, v in
     dict(gp1=gp1, A=A, B=B, C=C, D=D, C2=C2).items()}
Ph = {k: pheno(v) for k, v in G.items()}
Ph["MZ"] = pheno(G["C"])                     # C 의 일란성 쌍둥이

PAIRS = [
    ("일란성 쌍둥이", "C", "MZ", 1.0),
    ("부모–자식", "A", "C", 0.5),
    ("친형제", "A", "B", 0.5),
    ("조부모–손자", "gp1", "C", 0.25),
    ("반형제", "C", "C2", 0.25),
    ("사촌", "C", "D", 0.125),
]

obs = [np.corrcoef(Ph[a], Ph[b])[0, 1] for _, a, b, _ in PAIRS]
exp = [f * H2 for *_, f in PAIRS]
for (nm, _, _, f), o, e in zip(PAIRS, obs, exp):
    print(f"  {nm:>12}  혈연의 몫 {f:5.3f}   이론 {e:.4f}   실측 {o:.4f}")

# === ① 유전자 수에 따른 분포 =========================================
# p = 1/2 이면 유전값은 정확히 Binomial(2k, 1/2) 이다. 모의로 그리면 격자와
# 구간이 어긋나 톱니가 생기므로 확률을 그대로 쓴다. 눈금은 표준화하고,
# 높이는 격자 간격으로 나누어 밀도로 맞춘다 (그래야 k 끼리 견줄 수 있다).
from scipy.stats import binom, norm  # noqa: E402

DISTS = {}
for k in (1, 2, 5, 50):
    v = np.arange(2 * k + 1)
    pm = binom.pmf(v, 2 * k, 0.5)
    sd = np.sqrt(2 * k * 0.25)
    DISTS[k] = ((v - k) / sd, pm * sd)

# === ③ 분산의 쪼개짐 =================================================
g_c = G["C"]
e_c = Ph["C"] - g_c
rows = []
for h2 in (0.2, 0.5, 0.8):
    ve = VAR_G * (1 - h2) / h2
    p_ = g_c + RNG.normal(0, np.sqrt(ve), N)
    rows.append((h2, np.var(g_c), ve, np.var(p_),
                 np.var(g_c) + ve, np.var(g_c) / np.var(p_)))

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

zz = np.linspace(-3.6, 3.6, 300)
ax1.plot(zz, norm.pdf(zz), color="black", lw=1.5, ls="--", label="정규분포")
for k, c in zip((1, 2, 5, 50), ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4")):
    xv, yv = DISTS[k]
    if k <= 5:
        ax1.vlines(xv, 0, yv, color=c, lw=3.0, label=f"유전자 ${k}$ 개")
        ax1.plot(xv, yv, "o", color=c, ms=5)
    else:
        ax1.plot(xv, yv, color=c, lw=2.8, label=f"유전자 ${k}$ 개")
ax1.set_xlim(-3.6, 3.6)
ax1.set_ylim(0, 0.46)
ax1.set_xlabel("표준화한 유전값")
ax1.set_ylabel("밀도 (확률 $\\div$ 격자 간격)")
ax1.set_title("① 알갱이를 더하면 이어진다", fontsize=12)
ax1.legend(loc="upper right", fontsize=9.5)
ax1.grid(alpha=0.25)

y = np.arange(len(PAIRS))
ax2.barh(y, obs, 0.55, color="#1f77b4", label="모의로 잰 상관")
ax2.plot(exp, y, "D", color="#d62728", ms=9, zorder=5,
         label=f"이론: 혈연의 몫 $\\times\\ h^2$ (${H2}$)")
for i, (o, e) in enumerate(zip(obs, exp)):
    ax2.text(o + 0.012, i, f"${o:.3f}$", va="center", fontsize=10)
ax2.set_yticks(y)
ax2.set_yticklabels([nm for nm, *_ in PAIRS], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlim(0, 0.62)
ax2.set_xlabel("표현형 상관")
ax2.set_title(f"② 친척 사이의 상관 ($h^2 = {H2}$)", fontsize=12)
ax2.legend(loc="lower right", fontsize=9.5)
ax2.grid(alpha=0.25, axis="x")

x = np.arange(len(rows))
w = 0.55
vg = np.array([r[1] for r in rows])
ve = np.array([r[2] for r in rows])
ax3.bar(x, vg, w, color="#1f77b4", label="유전이 만든 몫 $V_G$")
ax3.bar(x, ve, w, bottom=vg, color="#ff7f0e", label="환경이 만든 몫 $V_E$")
ax3.plot(x, [r[3] for r in rows], "_", color="black", ms=36, mew=2.5,
         label="실제 총분산 $V_P$")
for i, r in enumerate(rows):
    ax3.text(i, r[3] + 3.5, f"$V_G + V_E = {r[4]:.1f}$\n실측 $V_P = {r[3]:.1f}$",
             ha="center", fontsize=9.5)
ax3.set_xticks(x)
ax3.set_xticklabels([f"$h^2 = {r[0]}$" for r in rows], fontsize=11)
ax3.set_ylabel("분산")
ax3.set_title("③ 총변동을 원인별로 쪼갠다", fontsize=12)
ax3.set_ylim(0, max(r[3] for r in rows) * 1.55)
ax3.legend(loc="upper right", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch12/anova_one_way/img/fisher1918.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① 유전자 수에 따른 서로 다른 유전값의 가짓수")
for k in (1, 2, 5, 50):
    print(f"   {k:>3} 개 -> {2 * k + 1} 가지")

print(f"\n② 친척 상관 (가계 {N:,} 벌, 유전자 {K} 자리, h^2 = {H2})")
print(f"{'관계':>14}{'혈연의 몫':>11}{'이론':>9}{'실측':>9}")
for (nm, _, _, f), o, e in zip(PAIRS, obs, exp):
    print(f"{nm:>14}{f:>11.3f}{e:>9.4f}{o:>9.4f}")

print(f"\n③ 분산의 쪼개짐")
print(f"{'h^2':>6}{'V_G':>9}{'V_E':>9}{'V_G+V_E':>10}{'실측 V_P':>11}{'V_G/V_P':>10}")
for r in rows:
    print(f"{r[0]:>6.1f}{r[1]:>9.2f}{r[2]:>9.2f}{r[4]:>10.2f}{r[3]:>11.2f}"
          f"{r[5]:>10.4f}")
