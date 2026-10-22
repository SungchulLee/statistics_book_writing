r"""희소성이 실제로 무엇을 아껴 주는가 — 메모리와 시간을 잰다.

$0$ 이 많다고 저절로 작아지거나 빨라지지 않는다. $0$ 을 **빼고 저장**하려면
자리를 적어 두어야 하고(CSR 의 열 인덱스), $0$ 을 **건너뛰고 곱**하려면
메모리 접근이 흩어진다. 반면 $0$ 이 **열 단위로 모여** 있으면 행렬이 통째로
작아져 밀집 연산이 그대로 빨라진다 — 라쏘의 희소성이 바로 그 꼴이다.

① 메모리 — CSR 은 밀도 $0.5$ 에서야 밀집과 같아진다.
② 시간 — 밀집 행렬곱은 밀도와 무관하게 일정하고, CSR 이 이기려면 밀도가
   $0.2$ 아래여야 한다. 열 단위로 모인 $0$ 은 밀도에 정확히 비례해 빨라진다.
③ 밀도 $0.1$ 에서 셋을 나란히.

시간은 이 기계에서 BLAS 스레드를 하나로 묶고 잰 값이다. 절대값은 기계마다
다르지만 **세 곡선의 모양**은 바뀌지 않는다.

만드는 파일: docs/ch18/lasso/img/sparsity_cost.png
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import scipy.sparse as sp   # noqa: E402

RNG = np.random.default_rng(0)
N, P, K, REP = 4000, 4000, 64, 5
DENS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
SHOW = 0.1


def best_of(fn, rep=REP):
    fn()
    return min(_timed(fn) for _ in range(rep))


def _timed(fn):
    a = time.perf_counter()
    fn()
    return time.perf_counter() - a


B = RNG.standard_normal((P, K)).astype(np.float32)
print(f"설계행렬 {N}x{P} float32, 곱할 행렬 {P}x{K}, "
      f"되풀이 {REP} 회의 최솟값")
print(f"{'밀도':>7}{'밀집 MB':>10}{'CSR MB':>9}"
      f"{'밀집 ms':>10}{'CSR ms':>9}{'열압축 ms':>11}")
res = {k: [] for k in ["dmb", "smb", "dt", "st", "ct"]}
for dens in DENS:
    M = ((RNG.random((N, P)) < dens).astype(np.float32)
         * RNG.standard_normal((N, P)).astype(np.float32))
    S = sp.csr_matrix(M)
    res["dmb"].append(M.nbytes / 1e6)
    res["smb"].append((S.data.nbytes + S.indices.nbytes
                       + S.indptr.nbytes) / 1e6)
    res["dt"].append(best_of(lambda: M @ B) * 1e3)
    res["st"].append(best_of(lambda: S @ B) * 1e3)
    nc = max(1, int(round(P * dens)))          # 0 이 열 단위로 모인 경우
    Mc = np.ascontiguousarray(M[:, :nc])
    Bc = np.ascontiguousarray(B[:nc])
    res["ct"].append(best_of(lambda: Mc @ Bc) * 1e3)
    print(f"{dens:>7.3f}{res['dmb'][-1]:>10.1f}{res['smb'][-1]:>9.1f}"
          f"{res['dt'][-1]:>10.2f}{res['st'][-1]:>9.2f}{res['ct'][-1]:>11.2f}")

i = DENS.index(SHOW)
print(f"\n밀도 {SHOW} (= {100 * (1 - SHOW):.0f}% 희소) 에서")
print(f"   밀집   {res['dt'][i]:6.2f} ms   (기준)")
print(f"   CSR    {res['st'][i]:6.2f} ms   {res['dt'][i] / res['st'][i]:.1f} 배")
print(f"   열압축 {res['ct'][i]:6.2f} ms   "
      f"{res['dt'][i] / res['ct'][i]:.1f} 배")
print(f"\n메모리 — float32 자료에 int32 인덱스를 붙이므로 CSR 은 "
      f"0 이 아닌 원소마다 8 바이트,")
print(f"   밀집은 모든 칸에 4 바이트다. 그래서 밀도 0.5 에서 정확히 같아진다 "
      f"({res['smb'][1]:.1f} 대 {res['dmb'][1]:.1f} MB).")
j = next(k for k in range(len(DENS)) if res["st"][k] < res["dt"][k])
print(f"시간 — CSR 이 밀집을 처음 이기는 밀도는 {DENS[j]} 이다 "
      f"({res['st'][j]:.2f} 대 {res['dt'][j]:.2f} ms). "
      f"{100 * (1 - DENS[j]):.0f}% 를 지워야 본전이다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_D, C_S, C_C = "#1f77b4", "#d62728", "#2ca02c"
d = np.array(DENS)

ax1.loglog(d, res["dmb"], "o-", color=C_D, lw=2.6, ms=7, label="밀집 배열")
ax1.loglog(d, res["smb"], "s-", color=C_S, lw=2.6, ms=7, label="CSR 희소 형식")
ax1.axvline(0.5, color="#555555", lw=1.3, ls=":")
ax1.annotate("밀도 $0.5$ —\n여기서 둘이 같다", xy=(0.5, res["smb"][1]),
             xytext=(0.022, 170), textcoords="data", fontsize=10.5,
             color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax1.set_xticks([0.005, 0.02, 0.1, 0.5, 1.0])
ax1.set_xticklabels(["$0.005$", "$0.02$", "$0.1$", "$0.5$", "$1$"],
                    fontsize=9.5)
ax1.set_yticks([1, 3, 10, 30, 100, 300])
ax1.set_yticklabels(["$1$", "$3$", "$10$", "$30$", "$100$", "$300$"])
ax1.minorticks_off()
ax1.set_xlabel("밀도 ($0$ 이 아닌 원소의 비율)")
ax1.set_ylabel("메모리 (MB)")
ax1.set_ylim(0.5, 400)
ax1.set_title("① 반을 지워야 본전이다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25)

ax2.loglog(d, res["dt"], "o-", color=C_D, lw=2.6, ms=7, label="밀집 행렬곱")
ax2.loglog(d, res["st"], "s-", color=C_S, lw=2.6, ms=7, label="CSR (흩어진 $0$)")
ax2.loglog(d, res["ct"], "^-", color=C_C, lw=2.6, ms=7,
           label="열을 통째로 버림 (라쏘의 꼴)")
ax2.axvline(DENS[j], color="#555555", lw=1.3, ls=":")
ax2.annotate(f"CSR 이 밀집을 이기기\n시작하는 밀도 ${DENS[j]}$",
             xy=(DENS[j], res["st"][j]), xytext=(0.012, 42),
             textcoords="data", fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax2.set_xticks([0.005, 0.02, 0.1, 0.5, 1.0])
ax2.set_xticklabels(["$0.005$", "$0.02$", "$0.1$", "$0.5$", "$1$"],
                    fontsize=9.5)
ax2.set_yticks([0.1, 1, 10, 100])
ax2.set_yticklabels(["$0.1$", "$1$", "$10$", "$100$"])
ax2.minorticks_off()
ax2.set_xlabel("밀도")
ax2.set_ylabel("행렬곱 시간 (ms)")
ax2.set_ylim(0.07, 200)
ax2.set_title("② 밀집은 밀도를 쳐다보지도 않는다", fontsize=12)
ax2.legend(loc="lower right", fontsize=9.5)
ax2.grid(alpha=0.25)

vals = [res["dt"][i], res["st"][i], res["ct"][i]]
names = ["밀집\n(그냥 둔다)", "CSR\n(흩어진 $0$)", "열 버림\n(라쏘의 꼴)"]
ax3.bar(np.arange(3), vals, 0.55, color=[C_D, C_S, C_C])
for x, v in enumerate(vals):
    ax3.text(x, v * 1.14, f"${v:.2f}$ ms", ha="center", fontsize=11)
    if x:
        ax3.text(x, v * 0.55, f"${vals[0] / v:.0f}$ 배", ha="center",
                 fontsize=12, color="white", fontweight="bold")
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(names, fontsize=10.5)
ax3.set_yscale("log")
ax3.set_yticks([1, 3, 10, 30])
ax3.set_yticklabels(["$1$", "$3$", "$10$", "$30$"])
ax3.minorticks_off()
ax3.set_ylabel("행렬곱 시간 (ms)")
ax3.set_ylim(0.7, 70)
ax3.set_title(f"③ 같은 ${100 * (1 - SHOW):.0f}\\%$ 희소, 다른 결과", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch18/lasso/img/sparsity_cost.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
