r"""수능 $9$ 등급제 — 순위를 아홉 칸으로 접으면 무엇이 남고 무엇이 사라지는가.

① 아홉 등급의 폭. 가운데는 $0.5\sigma$ 안에 $20\%$ 가 들어가고 양끝은 열려 있다.
② 시험에 잡음이 있으면 등급이 바뀐다 — 신뢰도별 오분류율.
③ 원점수와 등급이 각각 참 실력과 얼마나 맞는가.

만드는 파일: docs/ch16/foundations/img/suneung_grade.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N = 400_000

PCT = np.array([4, 7, 12, 17, 20, 17, 12, 7, 4])  # 등급별 비율 (%)
CUM = np.cumsum(PCT) / 100.0  # 누적 (위에서부터)
CUT = stats.norm.isf(CUM[:-1])  # 등급 경계의 z (높은 쪽부터)


def grade(z):
    """z 가 클수록 1 등급. np.digitize 는 오름차순을 원하므로 뒤집어 쓴다."""
    return 9 - np.digitize(z, CUT[::-1])


# === ② 잡음이 있으면 등급이 바뀐다 ==================================
# 관측점수 = 참 실력 + 오차. 신뢰도 rel = Var(참)/Var(관측).
def misgrade(rel, n=N, rng=RNG):
    t = rng.standard_normal(n)
    obs = np.sqrt(rel) * t + np.sqrt(1 - rel) * rng.standard_normal(n)
    obs = (obs - obs.mean()) / obs.std()
    t = (t - t.mean()) / t.std()
    gt, go = grade(t), grade(obs)
    return gt, go, obs, t


RELS = np.array([0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99])
miss, g1_true_in_g2 = [], []
for r in RELS:
    gt, go, _, _ = misgrade(r, n=120_000)
    miss.append(np.mean(gt != go))
    # 관측 2 등급인 학생 가운데 참 실력은 1 등급인 비율
    g1_true_in_g2.append(np.mean(gt[go == 2] == 1))
miss = np.array(miss)
g1_true_in_g2 = np.array(g1_true_in_g2)

# === ③ 참 실력과의 상관 =============================================
cor_raw, cor_grade = [], []
for r in RELS:
    gt, go, obs, t = misgrade(r, n=120_000)
    cor_raw.append(np.corrcoef(t, obs)[0, 1])
    cor_grade.append(np.corrcoef(t, -go)[0, 1])
cor_raw = np.array(cor_raw)
cor_grade = np.array(cor_grade)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

# ① 아홉 칸
edges = np.concatenate([[3.6], CUT, [-3.6]])
cmap = plt.get_cmap("RdYlBu_r")
xs = np.linspace(-3.6, 3.6, 900)
ax1.plot(xs, stats.norm.pdf(xs), color="black", lw=1.4)
for g in range(9):
    hi, lo = edges[g], edges[g + 1]
    seg = xs[(xs >= lo) & (xs <= hi)]
    ax1.fill_between(seg, 0, stats.norm.pdf(seg), color=cmap(g / 8), alpha=0.75)
    mid = np.clip((hi + lo) / 2, -3.0, 3.0)
    ax1.text(mid, stats.norm.pdf(mid) + 0.012, f"{g + 1}", ha="center", fontsize=11)
    ax1.text(mid, -0.023, f"{PCT[g]}$\\%$", ha="center", fontsize=8.5, color="#555555")
for c in CUT:
    ax1.axvline(c, color="white", lw=1.0)
ax1.set_xlim(-3.6, 3.6)
ax1.set_ylim(-0.04, 0.46)
ax1.set_xlabel("표준화 점수 ($z$)")
ax1.set_ylabel("밀도")
ax1.set_title("① 아홉 칸으로 접는다", fontsize=12)
ax1.annotate(
    f"$5$ 등급 — 폭 ${CUT[3] - CUT[4]:.2f}\\sigma$ 안에 $20\\%$",
    xy=(0.15, 0.26),
    xytext=(1.45, 0.405),
    ha="left",
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)
ax1.annotate(
    f"$1$ 등급 경계\n$z = {CUT[0]:.2f}$",
    xy=(CUT[0], 0.055),
    xytext=(2.55, 0.175),
    ha="center",
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

# ② 오분류
ax2.plot(RELS, miss * 100, "o-", color="#d62728", lw=2.3, ms=6, label="등급이 바뀌는 학생")
ax2.plot(
    RELS,
    g1_true_in_g2 * 100,
    "s-",
    color="#1f77b4",
    lw=2.3,
    ms=6,
    label="$2$ 등급인데 참 실력은 $1$ 등급",
)
ax2.set_xlabel("시험의 신뢰도 — 관측 분산 가운데 참 실력의 몫")
ax2.set_ylabel("비율 ($\\%$)")
ax2.set_title("② 한 번의 시험이 등급을 바꾼다", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)
ax2.set_xlim(0.68, 1.0)
ax2.set_ylim(0, 72)
for r0 in (0.90,):
    i = int(np.where(RELS == r0)[0][0])
    ax2.annotate(
        f"신뢰도 ${r0}$ 면\n{miss[i] * 100:.0f}$\\%$ 가 바뀐다",
        xy=(r0, miss[i] * 100),
        xytext=(0.785, 18),
        ha="center",
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

# ③ 참 실력과의 상관
ax3.plot(RELS, cor_raw, "o-", color="#2ca02c", lw=2.3, ms=6, label="원점수")
ax3.plot(RELS, cor_grade, "s-", color="#ff7f0e", lw=2.3, ms=6, label="등급 ($1$–$9$)")
ax3.fill_between(RELS, cor_grade, cor_raw, color="#ff7f0e", alpha=0.18)
ax3.set_xlabel("시험의 신뢰도")
ax3.set_ylabel("참 실력과의 상관")
ax3.set_title("③ 접는 데서 더 잃는 몫", fontsize=12)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)
ax3.set_xlim(0.68, 1.0)
ax3.set_ylim(0.78, 1.01)
i = int(np.where(RELS == 0.90)[0][0])
ax3.annotate(
    f"신뢰도 ${RELS[i]}$ 에서\n${cor_raw[i]:.3f} \\to {cor_grade[i]:.3f}$",
    xy=(RELS[i], (cor_raw[i] + cor_grade[i]) / 2),
    xytext=(0.80, 0.83),
    ha="center",
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

fig.tight_layout()
out = "docs/ch16/foundations/img/suneung_grade.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① 아홉 등급의 경계와 폭")
print(f"{'등급':>4}{'비율':>7}{'누적':>8}{'z 구간':>22}{'폭(σ)':>9}")
for g in range(9):
    hi, lo = edges[g], edges[g + 1]
    hs = "∞" if g == 0 else f"{hi:+.3f}"
    ls = "-∞" if g == 8 else f"{lo:+.3f}"
    w = "—" if g in (0, 8) else f"{hi - lo:.3f}"
    print(f"{g + 1:>4}{PCT[g]:>6}%{CUM[g] * 100:>7.0f}%{f'[{ls}, {hs}]':>22}{w:>9}")

print("\n② 신뢰도별 — 등급이 참 실력의 등급과 다른 학생의 비율")
for r, m, q in zip(RELS, miss, g1_true_in_g2):
    print(f"  신뢰도 {r:.2f}:  {m * 100:>5.1f}% 가 바뀜,  2 등급 중 참 1 등급 {q * 100:>5.1f}%")

print("\n③ 참 실력과의 상관")
for r, a, b in zip(RELS, cor_raw, cor_grade):
    print(f"  신뢰도 {r:.2f}:  원점수 {a:.4f}   등급 {b:.4f}   잃는 몫 {a - b:.4f}")

# 잡음이 전혀 없을 때 등급만으로 잃는 몫
t = RNG.standard_normal(N)
print(f"\n  잡음이 아예 없을 때(신뢰도 1) 등급의 상관: {np.corrcoef(t, -grade(t))[0, 1]:.4f}")
