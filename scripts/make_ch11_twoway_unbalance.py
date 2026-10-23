r"""이원배치에서 이분산이 어느 쪽으로 틀어지는가 — 짝에 달려 있다.

이 쪽은 「전통적인 이원배치 분산분석과 달리 이분산을 조정한다」를 내세운다. 그러면
전통적인 쪽은 정확히 어떻게 틀어지는가. 일원배치에 대해서는 「균형이면 로버스트,
불균형이면 문제」라는 경험칙이 책 여러 곳에 적혀 있지만, 이원배치에서 **어느 쪽으로**
틀어지는지는 짝에 달려 있다.

2x2 설계, 참 효과는 모두 0. 칸마다 분산을 다르게 주고 세 가지로 나눈다.
- 균형: 칸마다 같은 수
- 불리한 짝: 큰 분산 칸에 **작은** 표본
- 유리한 짝: 큰 분산 칸에 **큰** 표본

① 교호작용 검정의 1종오류.
② 요인 A 주효과 — 여기서는 균형도 보수적이 된다.
③ 불리한 짝에서 HC3 월드 검정이 고쳐 주는가 (옆 쪽의 방법).

만드는 파일: docs/ch11/anova_welch/img/twoway_unbalance.png
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
ALPHA = 0.05
REP = 6000
RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]      # 가장 큰 표준편차 / 가장 작은 것
CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def cell_sd(r):
    """칸마다의 표준편차 — 대각선으로 커진다."""
    return np.array([1.0, np.sqrt(r), np.sqrt(r), r])


def cell_n(kind, base=8):
    if kind == "균형":
        return np.array([base, base, base, base])
    if kind == "불리":            # 큰 분산에 작은 n
        return np.array([base * 2, base, base, base // 2])
    return np.array([base // 2, base, base, base * 2])   # 유리


def design(ns):
    rows, cid = [], []
    for ci, (i, j) in enumerate(CELLS):
        for _ in range(ns[ci]):
            rows.append([1.0, float(i), float(j), float(i * j)])
            cid.append(ci)
    return np.array(rows), np.array(cid)


def run(kind, r, rep=REP, robust=False):
    ns = cell_n(kind)
    X, cid = design(ns)
    sd = cell_sd(r)[cid]
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    h = np.clip(np.diag(X @ XtXi @ X.T), 0, 1 - 1e-9)
    R_int = np.array([[0.0, 0, 0, 1]])
    R_a = np.array([[0.0, 1, 0, 0]])
    hit = {"int": 0, "a": 0}
    for _ in range(rep):
        y = RNG.normal(0.0, sd)
        beta = XtXi @ (X.T @ y)
        e = y - X @ beta
        if robust:
            w = e ** 2 / (1 - h) ** 2                 # HC3
            V = XtXi @ (X.T * w) @ X @ XtXi
        else:
            V = XtXi * (e @ e) / (n - p)              # 고전적 OLS 분산
        for nm, Rm in (("int", R_int), ("a", R_a)):
            q = Rm.shape[0]
            M = Rm @ V @ Rm.T
            stat = float((Rm @ beta) @ np.linalg.solve(M, Rm @ beta)) / q
            if stat > stats.f.ppf(1 - ALPHA, q, n - p):
                hit[nm] += 1
    return hit["int"] / rep, hit["a"] / rep


KINDS = ["균형", "불리", "유리"]
print("2x2 설계, 참 효과는 모두 0, 칸 표본 (균형 8,8,8,8 / 불리 16,8,8,4 / "
      "유리 4,8,8,16)")
print(f"이름값 {ALPHA}, 반복 {REP:,}\n")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   분산비    균형 교호   불리 교호   유리 교호")
INT = {k: [] for k in KINDS}
MAIN = {k: [] for k in KINDS}
for r in RATIOS:
    row = []
    for k in KINDS:
        a, b = run(k, r)
        INT[k].append(a); MAIN[k].append(b)
        row.append(a)
    print(f"   {r:>7.1f}   {row[0]:>9.4f}   {row[1]:>9.4f}   {row[2]:>9.4f}")

print(f"\n   분산비 {RATIOS[-1]:.0f} 에서 균형은 {INT['균형'][-1]:.4f} 로 "
      f"거의 이름값인데,")
print(f"   불리한 짝은 {INT['불리'][-1]:.4f} 로 느슨해지고 "
      f"유리한 짝은 {INT['유리'][-1]:.4f} 로 보수적이 된다.")
print("   **같은 이분산인데 부호가 반대다. 짝이 방향을 정한다.**")

print("\n   주효과 쪽")
print("   분산비    균형 주효과   불리 주효과   유리 주효과")
for i, r in enumerate(RATIOS):
    print(f"   {r:>7.1f}   {MAIN['균형'][i]:>11.4f}   {MAIN['불리'][i]:>11.4f}"
          f"   {MAIN['유리'][i]:>11.4f}")

# === ③ HC3 가 고쳐 주는가 ===========================================
print("\n③ 불리한 짝에서 HC3 월드 검정으로 바꾸면 (교호작용)")
print("   분산비   고전적 OLS   HC3")
ROB = []
for r in RATIOS:
    a_cl, _ = run("불리", r)
    a_rb, _ = run("불리", r, robust=True)
    ROB.append(a_rb)
    print(f"   {r:>6.1f}   {a_cl:>10.4f}   {a_rb:>6.4f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {"균형": "#2ca02c", "불리": "#d62728", "유리": "#1f77b4"}
LBL = {"균형": "균형 (칸마다 같은 수)",
       "불리": "큰 분산에 작은 표본",
       "유리": "큰 분산에 큰 표본"}

for k in KINDS:
    ax1.plot(RATIOS, INT[k], "o-", color=COL[k], lw=2.8, ms=10, label=LBL[k])
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax1.text(1.1, ALPHA * 1.14, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax1.set_xlabel("가장 큰 표준편차 / 가장 작은 것")
ax1.set_ylabel("교호작용 검정의 $1$ 종오류")
ax1.set_title("① 같은 이분산, 반대 방향", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

for k in KINDS:
    ax2.plot(RATIOS, MAIN[k], "o-", color=COL[k], lw=2.8, ms=10, label=LBL[k])
ax2.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
# ylim 밖에 두면 bbox_inches="tight" 가 그림을 찌그러뜨린다
ax2.text(4.4, ALPHA * 0.90, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax2.set_xlabel("가장 큰 표준편차 / 가장 작은 것")
ax2.set_ylabel("요인 A 주효과의 $1$ 종오류")
ax2.set_title("② 주효과는 모양이 다르다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(RATIOS, INT["불리"], "o-", color="#d62728", lw=2.8, ms=10,
         label="고전적 OLS $F$")
ax3.plot(RATIOS, ROB, "s-", color="#1f77b4", lw=2.8, ms=10,
         label="HC3 월드 검정")
ax3.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
ax3.text(1.1, ALPHA * 1.14, "이름값 $0.05$", fontsize=10.5, color="#333333")
ax3.set_xlabel("가장 큰 표준편차 / 가장 작은 것")
ax3.set_ylabel("교호작용 검정의 $1$ 종오류")
ax3.set_title("③ 불리한 짝에서 바꿔 끼우면", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch11/anova_welch/img/twoway_unbalance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
