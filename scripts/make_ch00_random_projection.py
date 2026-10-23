r"""사영이 버리는 것이 거의 없을 때 — 무작위 부분공간과 거리 보존.

이 쪽은 사영이 공간을 둘로 쪼개고 한쪽을 버린다고 적는다. 버리는 쪽이 아까워
보이는데, 차원이 높으면 **거의 아무것도 버리지 않는** 사영이 있다.

무작위로 고른 k 차원 부분공간 위로의 직교사영 P = QQ^T 를 쓴다. 점들 사이의 거리가
얼마나 흐트러지는지를 재고, 필요한 k 가 무엇에 달려 있는지 본다.

① k 를 키우며 거리의 뒤틀림.
② 뒤틀림이 원래 차원 d 에 달려 있는가 — 달려 있지 않다.
③ 존슨–린덴슈트라우스 한계가 말하는 k (점 수에만 로그로 달려 있다).

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/random_projection.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

RNG = np.random.default_rng(0)


def distortion(n, d, k, rep=12, rng=RNG):
    """점 n 개를 d 차원에 두고 무작위 k 차원 부분공간으로 직교사영.

    거리비 ||P(x-y)|| * sqrt(d/k) / ||x-y|| 의 분포를 돌려준다.
    """
    worst, mid = [], []
    for _ in range(rep):
        X = rng.normal(0, 1, (n, d))
        # 무작위 정규직교 기저 Q (d x k) — P = QQ^T 가 직교사영이다
        Q, _ = np.linalg.qr(rng.normal(0, 1, (d, k)))
        Z = X @ Q                      # 부분공간 좌표. ||QQ^T v|| = ||Q^T v||
        i, j = np.triu_indices(n, 1)
        d0 = np.linalg.norm(X[i] - X[j], axis=1)
        d1 = np.linalg.norm(Z[i] - Z[j], axis=1) * np.sqrt(d / k)
        r = d1 / d0
        worst.append(float(np.max(np.abs(r - 1))))
        mid.append(float(np.median(np.abs(r - 1))))
    return float(np.mean(worst)), float(np.mean(mid))


# === ① k 를 키우며 ==================================================
N, D = 200, 2000
KS = [2, 5, 10, 25, 50, 100, 200, 400, 800]
print(f"점 {N} 개를 {D} 차원에 두고 무작위 k 차원 부분공간으로 사영")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("      k   가장 심한 뒤틀림   중앙값")
W1, M1 = [], []
for k in KS:
    w, m = distortion(N, D, k)
    W1.append(w); M1.append(m)
    print(f"   {k:>4}   {w:>15.4f}   {m:>7.4f}")
print(f"   {D} 차원을 {KS[4]} 차원으로 줄이면 **가장 심한 쌍**이 {W1[4]:.1%} 뒤틀리는데")
print(f"   **중앙값은 {M1[4]:.1%}** 다. 둘이 여섯 배 넘게 벌어진다 —")
print(f"   JL 한계가 보수적인 까닭이 이것이다. 한계는 가장 심한 쌍을 묶는다.")

# === ② 원래 차원 d 를 바꿔 가며 =====================================
DS = [100, 300, 1000, 3000, 10000]
K2 = 50
print(f"\n② k = {K2} 로 고정하고 원래 차원 d 만 바꾸면 (점 {N} 개)")
print("        d   가장 심한 뒤틀림   중앙값")
W2, M2 = [], []
for d in DS:
    w, m = distortion(N, d, K2)
    W2.append(w); M2.append(m)
    print(f"   {d:>6}   {w:>15.4f}   {m:>7.4f}")
print(f"   d 가 {DS[0]} 에서 {DS[-1]} 로 백 배가 되어도 뒤틀림이 "
      f"{W2[0]:.3f} 에서 {W2[-1]:.3f} 로 거의 그대로다.")
print("   **필요한 차원은 원래 차원에 달려 있지 않다.**")

# === ③ 존슨–린덴슈트라우스 한계 =====================================
def jl_min_dim(n, eps):
    """sklearn 의 johnson_lindenstrauss_min_dim 과 같은 식.

    sklearn 은 올림이 아니라 버림(astype)을 쓴다. 값을 맞추려면 그대로 따라야 한다.
    """
    return int(4 * np.log(n) / (eps ** 2 / 2 - eps ** 3 / 3))


NS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
EPS = [0.5, 0.2, 0.1]
print("\n③ 거리 뒤틀림을 eps 안으로 묶는 데 필요한 차원 (JL 한계)")
print("       점 수      eps=0.5   eps=0.2   eps=0.1")
JL = {e: [] for e in EPS}
for n in NS:
    row = []
    for e in EPS:
        v = jl_min_dim(n, e)
        JL[e].append(v)
        row.append(v)
    print(f"   {int(n):>9,}   {row[0]:>8,}  {row[1]:>8,}  {row[2]:>8,}")
print(f"   점이 백만 개여도 eps = 0.5 면 {jl_min_dim(1e6, 0.5):,} 차원이면 된다 "
      f"— sklearn 이 내놓는 값과 같다.")
print(f"   점을 백 배로 늘려도 ({int(NS[2]):,} → {int(NS[4]):,}) 필요한 차원은 "
      f"{JL[0.1][2]:,} 에서 {JL[0.1][4]:,} 로 {JL[0.1][4]/JL[0.1][2]:.1f} 배만 는다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

ax1.loglog(KS, W1, "o-", color=C_A, lw=2.8, ms=10, label="가장 심한 쌍")
ax1.loglog(KS, M1, "s-", color=C_B, lw=2.6, ms=9, label="중앙값")
ax1.axhline(0.1, color="#333333", lw=1.8, ls="--")
ax1.text(2.3, 0.115, "뒤틀림 $10\\%$", fontsize=10.5, color="#333333")
ax1.set_xlabel("사영한 차원 $k$")
ax1.set_ylabel("거리의 상대 뒤틀림")
ax1.set_xticks([2, 10, 100, 800])
ax1.set_xticklabels(["$2$", "$10$", "$100$", "$800$"])
ax1.set_yticks([0.01, 0.1, 1])
ax1.set_yticklabels(["$0.01$", "$0.1$", "$1$"])
ax1.minorticks_off()
ax1.set_title(f"① ${D}$ 차원을 $k$ 차원으로 (점 ${N}$ 개)", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.semilogx(DS, W2, "o-", color=C_A, lw=2.8, ms=11, label="가장 심한 쌍")
ax2.semilogx(DS, M2, "s-", color=C_B, lw=2.6, ms=9, label="중앙값")
ax2.set_xlabel("원래 차원 $d$")
ax2.set_ylabel("거리의 상대 뒤틀림")
ax2.set_ylim(0, max(W2) * 1.3)
ax2.set_xticks([100, 1000, 10000])
ax2.set_xticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$"])
ax2.minorticks_off()
ax2.set_title(f"② $k = {K2}$ 로 고정하고 $d$ 만 키우면", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

for e, c in zip(EPS, ["#9ecae1", "#4292c6", "#08519c"]):
    ax3.loglog(NS, JL[e], "o-", color=c, lw=2.6, ms=9,
               label=f"$\\varepsilon = {e}$")
ax3.plot([1e6], [jl_min_dim(1e6, 0.5)], "o", color=C_A, ms=14, zorder=5)
ax3.annotate(f"점 백만 개 → ${jl_min_dim(1e6, 0.5)}$ 차원",
             (1e6, jl_min_dim(1e6, 0.5)), xytext=(-0.5, -2.2),
             textcoords="offset fontsize", fontsize=10.5, color=C_A, ha="right")
ax3.set_xlabel("점의 수 $n$")
ax3.set_ylabel("필요한 차원 $k$")
ax3.set_xticks([100, 10000, 1000000])
ax3.set_xticklabels(["$10^{2}$", "$10^{4}$", "$10^{6}$"])
ax3.set_yticks([100, 1000, 10000, 100000, 1000000])
ax3.set_yticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$", "$10^{6}$"])
ax3.minorticks_off()
ax3.set_title("③ 존슨–린덴슈트라우스 한계", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/square_matrices/img/random_projection.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
