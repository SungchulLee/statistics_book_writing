r"""다중공선성의 두 번째 청구서 — 자릿수.

롱리($1967$)의 미국 거시경제 자료는 $16$ 행 $7$ 열인데 $\kappa(X^\top X) = 2.4\times10^{19}$ 로
**배정도의 한계를 넘는다.** 정규방정식을 그대로 풀면 맞는 자릿수가 몇 개나 남는가.

참값은 자료가 유한소수이므로 **유리수 산술로 정확히** 구한다.

① 조건수 사다리와 부동소수의 한계.
② 푸는 방법에 따라 남는 자릿수.
③ 중심화·표준화만 해도 조건수가 얼마나 떨어지는가.

만드는 파일: docs/ch13/diagnostics/img/longley.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from fractions import Fraction  # noqa: E402
import statsmodels.api as sm  # noqa: E402

d = sm.datasets.longley.load_pandas().data
NAMES = ["상수", "GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
Xf = np.column_stack(
    [np.ones(len(d))] + [d[c].to_numpy() for c in
                         ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]]
)
yf = d["TOTEMP"].to_numpy()


# === 참값: 유리수로 정확히 푼다 ====================================
def to_frac(a):
    return [[Fraction(str(round(float(v), 1))) for v in row] for row in np.atleast_2d(a)]


def exact_ols(X, y):
    Xr, yr = to_frac(X), [Fraction(str(round(float(v), 1))) for v in y]
    n, p = len(Xr), len(Xr[0])
    A = [[sum(Xr[i][j] * Xr[i][k] for i in range(n)) for k in range(p)] for j in range(p)]
    b = [sum(Xr[i][j] * yr[i] for i in range(n)) for j in range(p)]
    # 가우스 소거 (완전 유리수)
    M = [A[j] + [b[j]] for j in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(p):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[j][p] for j in range(p)]


BETA = exact_ols(Xf, yf)
beta_true = np.array([float(b) for b in BETA])


# === 여러 방법으로 풀어 본다 ========================================
def digits(est):
    rel = np.abs((np.asarray(est, dtype=float) - beta_true) / beta_true)
    rel = np.where(rel == 0, 1e-17, rel)
    return np.clip(-np.log10(rel), 0, 16)


def ne_solve(X, y, dt):
    X, y = X.astype(dt), y.astype(dt)
    return np.linalg.solve(X.T @ X, X.T @ y)


def ne_inv(X, y, dt):
    X, y = X.astype(dt), y.astype(dt)
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def qr_solve(X, y, dt):
    X, y = X.astype(dt), y.astype(dt)
    q, r = np.linalg.qr(X)
    return np.linalg.solve(r, q.T @ y)


METHODS = [
    ("정규방정식\n역행렬·배정도", ne_inv(Xf, yf, np.float64)),
    ("정규방정식\nsolve·배정도", ne_solve(Xf, yf, np.float64)),
    ("QR 분해\n배정도", qr_solve(Xf, yf, np.float64)),
    ("QR 분해\n단정도", qr_solve(Xf, yf, np.float32)),
    ("정규방정식\nsolve·단정도", ne_solve(Xf, yf, np.float32)),
]

# === ③ 중심화·표준화 ===============================================
Xc = Xf.copy()
Xc[:, 1:] = Xf[:, 1:] - Xf[:, 1:].mean(axis=0)
Xs = Xc.copy()
Xs[:, 1:] = Xc[:, 1:] / Xc[:, 1:].std(axis=0, ddof=0)
SCALINGS = [
    ("원래 자료", Xf),
    ("중심화", Xc),
    ("중심화 + 표준화", Xs),
]

def sci(v, d=1):
    """문자열 치환 대신 지수를 직접 꺼내 mathtext 로 적는다."""
    m, e = f"{v:.{d}e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

ladder = [
    ("$\\kappa(X)$", np.linalg.cond(Xf), "#1f77b4"),
    ("$\\kappa(X^\\top X)$", np.linalg.cond(Xf.T @ Xf), "#d62728"),
]
for i, (lab, v, c) in enumerate(ladder):
    ax1.barh([1 - i], [v], color=c, height=0.45, zorder=3)
    ax1.text(v * 1.6, 1 - i, sci(v), va="center", fontsize=12, color=c)
for lim, lab, c in [
    (1 / np.finfo(np.float32).eps, "단정도의 한계", "#ff7f0e"),
    (1 / np.finfo(np.float64).eps, "배정도의 한계", "#555555"),
]:
    ax1.axvline(lim, color=c, lw=1.6, ls="--")
    ax1.text(lim * 0.80, 1.42, lab, rotation=90, ha="right", va="top",
             fontsize=9.5, color=c)
ax1.set_xscale("log")
ax1.set_yticks([1, 0])
ax1.set_yticklabels(["$X$", "$X^\\top X$"], fontsize=13)
ax1.set_xlim(1e2, 1e22)
ax1.set_ylim(-0.75, 1.5)
ax1.set_xticks([1e5, 1e10, 1e15, 1e20])
ax1.set_xticklabels(["$10^{5}$", "$10^{10}$", "$10^{15}$", "$10^{20}$"])
ax1.minorticks_off()
ax1.set_xlabel("조건수")
ax1.set_title("① 정규방정식은 조건수를 제곱한다", fontsize=12)
ax1.grid(alpha=0.25, axis="x")

labs = [m[0] for m in METHODS]
worst = [float(digits(m[1]).min()) for m in METHODS]
cols = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
ax2.bar(range(len(labs)), worst, color=cols, width=0.62, zorder=3)
for i, v in enumerate(worst):
    ax2.text(i, v + 0.2, f"${v:.1f}$", ha="center", fontsize=11, fontweight="bold")
ax2.set_xticks(range(len(labs)))
ax2.set_xticklabels(labs, fontsize=8.5)
ax2.set_ylabel("가장 나쁜 계수의 맞는 자릿수")
ax2.set_title("② 같은 자료, 같은 식, 다른 답", fontsize=12)
ax2.set_ylim(0, 12)
ax2.grid(alpha=0.25, axis="y")

names = [s[0] for s in SCALINGS]
kx = [np.linalg.cond(s[1]) for s in SCALINGS]
kxx = [np.linalg.cond(s[1].T @ s[1]) for s in SCALINGS]
w = 0.36
xs = np.arange(len(names))
ax3.bar(xs - w / 2, kx, w, color="#1f77b4", label="$\\kappa(X)$", zorder=3)
ax3.bar(xs + w / 2, kxx, w, color="#d62728", label="$\\kappa(X^\\top X)$", zorder=3)
ax3.set_yscale("log")
ax3.axhline(1 / np.finfo(np.float64).eps, color="#555555", lw=1.5, ls="--")
ax3.text(2.45, 1 / np.finfo(np.float64).eps * 1.6, "배정도의 한계", ha="right",
         fontsize=9.5, color="#555555")
ax3.set_xticks(xs)
ax3.set_xticklabels(names, fontsize=10.5)
ax3.set_ylim(1, 1e22)
ax3.set_yticks([1e0, 1e5, 1e10, 1e15, 1e20])
ax3.set_yticklabels(["$1$", "$10^{5}$", "$10^{10}$", "$10^{15}$", "$10^{20}$"])
ax3.minorticks_off()
ax3.set_ylabel("조건수")
ax3.set_title("③ 중심화만 해도 달라진다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch13/diagnostics/img/longley.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n정확한 해 (유리수 산술)")
for n, b in zip(NAMES, beta_true):
    print(f"  {n:>8}: {b: .10f}")

print(f"\n조건수:  κ(X) = {np.linalg.cond(Xf):.4e}   κ(XtX) = {np.linalg.cond(Xf.T @ Xf):.4e}")
print(f"  배정도 한계 1/eps = {1 / np.finfo(np.float64).eps:.3e}")
print(f"  단정도 한계 1/eps = {1 / np.finfo(np.float32).eps:.3e}")

print("\n방법별 맞는 자릿수 (가장 나쁜 계수 / 평균)")
for lab, est in METHODS:
    dg = digits(est)
    print(f"  {lab.replace(chr(10), ' '):<28}{dg.min():6.2f}{dg.mean():9.2f}")

print("\n중심화·표준화")
for nm, Xv in SCALINGS:
    print(f"  {nm:<16}κ(X) = {np.linalg.cond(Xv):.3e}   κ(XtX) = {np.linalg.cond(Xv.T @ Xv):.3e}")

print("\n단정도 정규방정식이 내놓는 YEAR 계수")
i = NAMES.index("YEAR")
print(f"  참값 {beta_true[i]:.6f}")
for lab, est in METHODS:
    print(f"  {lab.replace(chr(10), ' '):<28}{float(est[i]):18.6f}")
