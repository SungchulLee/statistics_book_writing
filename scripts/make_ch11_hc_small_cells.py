r"""HC3 는 어디서부터 믿을 만한가 — 칸이 작을 때의 로버스트 분산분석.

이 쪽은 포화 요인 OLS 에 HC3 공분산과 월드 F 를 걸어 주효과·교호작용을 검정하고,
HC3 가 「HC0 나 HC1 보다 소표본 성능이 좋다」고 적어 둔다. 그 말을 수로 확인하고,
**어디서부터** 믿을 만한지까지 잰다.

3x3 포화 요인설계, 참 효과는 모두 0, 칸마다 분산이 다르다(이분산).
칸 반복수를 키우며 네 가지 HC 의 1종오류를 잰다.

① 교호작용 검정의 1종오류 — HC0 는 칸이 작을 때 크게 부푼다.
② 주효과에서도 같은 모양.
③ 이름값에 닿는 자리 — 전체 표본 수로 보면 어디쯤인가.

만드는 파일: docs/ch11/anova_welch/img/hc_small_cells.png
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
A, B = 3, 3                       # 요인 수준
ALPHA = 0.05
REP = 4000
# 칸마다 다른 표준편차 — 가장 큰 칸이 가장 작은 칸의 네 배
SD = np.array([[1.0, 1.0, 2.0],
               [1.0, 2.0, 3.0],
               [2.0, 3.0, 4.0]])


def design(nrep):
    """포화 요인모형의 설계행렬과 칸 표시."""
    cells = [(i, j) for i in range(A) for j in range(B)]
    rows, cid = [], []
    for ci, (i, j) in enumerate(cells):
        for _ in range(nrep):
            r = [1.0]
            r += [1.0 if i == a else 0.0 for a in range(1, A)]
            r += [1.0 if j == b else 0.0 for b in range(1, B)]
            r += [1.0 if (i == a and j == b) else 0.0
                  for a in range(1, A) for b in range(1, B)]
            rows.append(r)
            cid.append(ci)
    return np.array(rows), np.array(cid), cells


def hc_cov(X, e, h, kind):
    """HC0~HC3 공분산."""
    n, p = X.shape
    if kind == 0:
        w = e ** 2
    elif kind == 1:
        w = e ** 2 * n / (n - p)
    elif kind == 2:
        w = e ** 2 / (1 - h)
    else:
        w = e ** 2 / (1 - h) ** 2
    XtXi = np.linalg.pinv(X.T @ X)
    return XtXi @ (X.T * w) @ X @ XtXi


def run(nrep, rep=REP):
    X, cid, cells = design(nrep)
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    Hat = X @ XtXi @ X.T
    h = np.clip(np.diag(Hat), 0, 1 - 1e-10)
    sd_vec = np.array([SD[cells[c]] for c in cid])

    # 교호작용 R: 마지막 (A-1)(B-1) 개 계수가 0
    q_int = (A - 1) * (B - 1)
    R_int = np.zeros((q_int, p)); R_int[:, p - q_int:] = np.eye(q_int)
    # 요인 A 주효과: 2~A 번째 계수
    R_a = np.zeros((A - 1, p)); R_a[:, 1:A] = np.eye(A - 1)

    out = {k: {"int": 0, "a": 0} for k in range(4)}
    for _ in range(rep):
        y = RNG.normal(0.0, sd_vec)          # 참 효과는 모두 0
        beta = XtXi @ (X.T @ y)
        e = y - X @ beta
        for k in range(4):
            V = hc_cov(X, e, h, k)
            for nm, R in (("int", R_int), ("a", R_a)):
                q = R.shape[0]
                M = R @ V @ R.T
                stat = float((R @ beta) @ np.linalg.solve(M, R @ beta)) / q
                if stat > stats.f.ppf(1 - ALPHA, q, n - p):
                    out[k][nm] += 1
    return {k: {nm: out[k][nm] / rep for nm in ("int", "a")} for k in range(4)}


NREPS = [2, 3, 4, 6, 8, 12, 20, 30]
print(f"{A}x{B} 포화 요인설계, 참 효과는 모두 0, 칸 표준편차 1~4 (이분산)")
print(f"이름값 {ALPHA}, 반복 {REP:,}\n")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   칸 반복  전체 n     HC0 교호   HC1 교호   HC2 교호   HC3 교호")
RES = {k: {"int": [], "a": []} for k in range(4)}
TOT = []
for nr in NREPS:
    r = run(nr)
    TOT.append(A * B * nr)
    for k in range(4):
        RES[k]["int"].append(r[k]["int"])
        RES[k]["a"].append(r[k]["a"])
    print(f"   {nr:>6}  {A*B*nr:>6}   {r[0]['int']:>8.4f}   {r[1]['int']:>8.4f}"
          f"   {r[2]['int']:>8.4f}   {r[3]['int']:>8.4f}")

print(f"\n   칸마다 {NREPS[0]} 개일 때 HC0 의 교호작용 1종오류가 "
      f"{RES[0]['int'][0]:.3f} 로 이름값의 {RES[0]['int'][0]/ALPHA:.1f} 배다.")
print(f"   같은 자리에서 HC3 는 {RES[3]['int'][0]:.3f} 다.")
ok = [i for i, v in enumerate(RES[3]["int"]) if v <= 0.07]
if ok:
    print(f"   HC3 가 0.07 아래로 들어오는 것은 칸 반복 {NREPS[ok[0]]} "
          f"(전체 {TOT[ok[0]]}) 부터다.")

print("\n   주효과 쪽")
print("   칸 반복     HC0 주효과   HC3 주효과")
for i, nr in enumerate(NREPS):
    print(f"   {nr:>6}     {RES[0]['a'][i]:>10.4f}   {RES[3]['a'][i]:>10.4f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {0: "#d62728", 1: "#ff7f0e", 2: "#9467bd", 3: "#1f77b4"}
NAME = {0: "HC0", 1: "HC1", 2: "HC2", 3: "HC3"}

for k in range(4):
    ax1.plot(NREPS, RES[k]["int"], "o-", color=COL[k], lw=2.6, ms=9,
             label=NAME[k])
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax1.text(18, ALPHA * 1.18, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax1.set_xlabel("칸마다의 반복 수")
ax1.set_ylabel("교호작용 검정의 $1$ 종오류")
ax1.set_title("① 칸이 작으면 HC0 가 무너진다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25)

for k in (0, 3):
    ax2.plot(NREPS, RES[k]["a"], "o-", color=COL[k], lw=2.8, ms=10,
             label=f"{NAME[k]} 주효과")
ax2.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax2.text(18, ALPHA * 1.18, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax2.set_xlabel("칸마다의 반복 수")
ax2.set_ylabel("요인 A 주효과의 $1$ 종오류")
ax2.set_title("② 주효과에서도 같다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

for k in range(4):
    ax3.semilogx(TOT, RES[k]["int"], "o-", color=COL[k], lw=2.6, ms=9,
                 label=NAME[k])
ax3.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax3.axvline(250, color="#2ca02c", lw=2.2, ls=":")
ax3.text(255, max(RES[0]["int"]) * 0.75, "롱과 어빈이\n말한 $250$",
         fontsize=10.5, color="#2ca02c")
ax3.set_xlabel("전체 표본 수")
ax3.set_ylabel("교호작용 검정의 $1$ 종오류")
ax3.set_xticks([20, 100, 270])
ax3.set_xticklabels(["$20$", "$100$", "$270$"])
ax3.minorticks_off()
ax3.set_title("③ 전체 표본 수로 보면", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch11/anova_welch/img/hc_small_cells.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
