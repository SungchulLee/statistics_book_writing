r"""교호작용이 그 실험의 물음이었다 — ToothGrowth 의 OJ 대 VC.

이 쪽은 이원배치의 사후검정을 다루고, 교호작용이 유의하면 주효과의 쌍별 비교가
오해를 부른다고 적는다. 쪽이 쓰는 ToothGrowth 자료가 바로 그 보기다.

자료: 기니피그 60 마리, 비타민 C 를 세 용량(0.5, 1, 2 mg/day)으로 두 가지 방법
(오렌지주스 OJ, 아스코르브산 VC)으로 준 뒤 치아모세포 길이를 잰 것. 칸마다 10 마리.
R `datasets::ToothGrowth` 이고 출처는 Bliss (1952), 참고문헌은 Crampton (1947).

① 칸평균 — 두 선이 2 mg 에서 만난다.
② 용량 안에서 OJ-VC 를 따로 보면(단순효과) 0.5 와 1 에서만 갈린다.
③ 주효과만 보면 「OJ 가 3.70 앞선다」 하나로 뭉개진다.

만드는 파일: docs/ch11/post_hoc/img/toothgrowth_interaction.png
"""

import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats   # noqa: E402

# === 자료 (R datasets::ToothGrowth, 60 마리) =========================
DATA = {
    ("OJ", 0.5): [15.2, 21.5, 17.6, 9.7, 14.5, 10.0, 8.2, 9.4, 16.5, 9.7],
    ("OJ", 1.0): [19.7, 23.3, 23.6, 26.4, 20.0, 25.2, 25.8, 21.2, 14.5, 27.3],
    ("OJ", 2.0): [25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23.0],
    ("VC", 0.5): [4.2, 11.5, 7.3, 5.8, 6.4, 10.0, 11.2, 11.2, 5.2, 7.0],
    ("VC", 1.0): [16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5],
    ("VC", 2.0): [23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5],
}
DOSES = [0.5, 1.0, 2.0]
SUPPS = ["OJ", "VC"]
n = 10
N = 60
cells = {k: np.array(v, float) for k, v in DATA.items()}

# === 포화 이원배치의 MSE 와 자유도 ===================================
sse = sum(((v - v.mean()) ** 2).sum() for v in cells.values())
dfe = N - len(cells)                     # 60 - 6 = 54
mse = sse / dfe
print(f"자료 {N} 마리, 칸 {len(cells)} 개 × {n} 마리")
print(f"MSE = {mse:.4f}, 오차 자유도 = {dfe}")

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n① 칸평균")
print("   용량     OJ      VC    차(OJ-VC)")
M = {k: v.mean() for k, v in cells.items()}
DIFF = []
for d in DOSES:
    diff = M[("OJ", d)] - M[("VC", d)]
    DIFF.append(diff)
    print(f"   {d:>4}  {M[('OJ', d)]:>6.2f}  {M[('VC', d)]:>6.2f}   {diff:>+8.2f}")

# === ② 단순효과: 용량 안에서 OJ 대 VC ================================
# 비교 셋에 본페로니를 걸고 합동 MSE 를 쓴다
se_pair = np.sqrt(2 * mse / n)
t3 = stats.t.ppf(1 - 0.05 / (2 * 3), dfe)
print(f"\n② 단순효과 — 용량 안에서 OJ 대 VC (비교 3 개, 본페로니)")
print(f"   합동 SE = {se_pair:.4f}, 임계 t = {t3:.4f}, 반폭 = {t3 * se_pair:.4f}")
print("   용량      차      보정 p    유의")
HW3 = t3 * se_pair
PS = []
for d, diff in zip(DOSES, DIFF):
    p_raw = 2 * stats.t.sf(abs(diff) / se_pair, dfe)
    p_adj = min(1.0, p_raw * 3)
    PS.append(p_adj)
    print(f"   {d:>4}  {diff:>+7.2f}  {p_adj:>9.5f}    "
          f"{'예' if p_adj < 0.05 else '아니다'}")
print("   **0.5 와 1 에서는 갈리고 2 에서는 갈리지 않는다. 이것이 교호작용이다.**")

# === ③ 주효과만 보면 ================================================
oj = np.concatenate([cells[("OJ", d)] for d in DOSES])
vc = np.concatenate([cells[("VC", d)] for d in DOSES])
main = oj.mean() - vc.mean()
se_main = np.sqrt(2 * mse / 30)
t_main = stats.t.ppf(0.975, dfe)
print(f"\n③ 용량을 평균해 넘기고 주효과만 보면")
print(f"   OJ {oj.mean():.3f} 대 VC {vc.mean():.3f},  차 {main:+.3f}")
print(f"   95% 구간 [{main - t_main * se_main:+.3f}, {main + t_main * se_main:+.3f}]"
      f",  p = {2 * stats.t.sf(main / se_main, dfe):.5f}")
print("   「OJ 가 앞선다」 한 줄로 뭉개진다 — 2 mg 에서 차이가 없다는 사실이 사라진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_OJ, C_VC, C_R = "#ff7f0e", "#1f77b4", "#d62728"

for s, c, mk in (("OJ", C_OJ, "o"), ("VC", C_VC, "s")):
    m = [M[(s, d)] for d in DOSES]
    e = [cells[(s, d)].std(ddof=1) / np.sqrt(n) for d in DOSES]
    ax1.errorbar(DOSES, m, yerr=e, fmt=mk + "-", color=c, lw=2.8, ms=11,
                 capsize=5, label=f"{s} ({'오렌지주스' if s == 'OJ' else '아스코르브산'})")
ax1.annotate("", xy=(2.0, M[("VC", 2.0)]), xytext=(2.0, M[("OJ", 2.0)]),
             arrowprops=dict(arrowstyle="<->", color=C_R, lw=2.0))
ax1.text(1.72, 28.6, f"$2$ mg 에서 차 ${DIFF[2]:+.2f}$", fontsize=10.5, color=C_R)
ax1.set_xlabel("용량 (mg/day)")
ax1.set_ylabel("치아모세포 길이")
ax1.set_xticks(DOSES)
ax1.set_ylim(4, 31)
ax1.set_title("① 두 선이 $2$ mg 에서 만난다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10.5)
ax1.grid(alpha=0.25)

xs = np.arange(3)
cols = [C_R if p < 0.05 else "#999999" for p in PS]
ax2.bar(xs, DIFF, color=cols, alpha=0.85, width=0.55)
ax2.errorbar(xs, DIFF, yerr=HW3, fmt="none", ecolor="#333333", capsize=6, lw=2)
ax2.axhline(0, color="#333333", lw=2.0)
# 막대 높이에 맞춰 띄우면 제목이나 축 밖으로 나간다 — 바닥에 가지런히 둔다
for i, (d, p) in enumerate(zip(DIFF, PS)):
    ax2.text(xs[i], -3.4, f"$p = {p:.3f}$", ha="center", fontsize=10.5,
             color=C_R if p < 0.05 else "#666666")
ax2.set_xticks(xs)
ax2.set_xticklabels([f"${d}$ mg" for d in DOSES], fontsize=11)
ax2.set_ylabel("OJ $-$ VC (치아모세포 길이)")
ax2.set_ylim(-4, 10.5)
ax2.set_title("② 용량 안에서 따로 보면 (본페로니 $3$ 비교)", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.bar([0], [main], color="#9467bd", alpha=0.85, width=0.4)
ax3.errorbar([0], [main], yerr=t_main * se_main, fmt="none", ecolor="#333333",
             capsize=7, lw=2)
ax3.axhline(0, color="#333333", lw=2.0)
for i, (d, diff) in enumerate(zip(DOSES, DIFF)):
    ax3.plot([1], [diff], "o", color=C_R if PS[i] < 0.05 else "#999999", ms=13)
    ax3.annotate(f"${d}$ mg", (1, diff), xytext=(0.8, -0.25),
                 textcoords="offset fontsize", fontsize=10, color="#555555")
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["주효과 하나로", "용량마다 따로"], fontsize=11)
ax3.set_ylabel("OJ $-$ VC")
ax3.set_xlim(-0.5, 1.8)
ax3.set_title("③ 뭉치면 사라지는 것", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch11/post_hoc/img/toothgrowth_interaction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
