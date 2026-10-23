r"""멱등인데 대칭이 아니면 R^2 가 음수가 된다 — 2SLS 의 모자 행렬.

이 쪽의 절 4 표에 「제곱합의 피타고라스 분해: 빗사영은 아니다」라는 줄이 있다.
그 줄이 실무에서 어디를 무는지 재어 본다.

2SLS 의 적합값은 yhat = H y, H = X (Xhat' X)^{-1} Xhat' (Xhat = P_Z X) 인데
  - H^2 = H          (멱등이다 — 정의 1)
  - tr(H) = rank(H) = p   (정리 2 가 그대로 성립한다)
  - H' != H          (대칭이 아니다 — 빗사영이다)
라서 TSS = MSS + RSS 가 깨지고 R^2 = 1 - RSS/TSS 가 음수가 될 수 있다.

① 모의실험마다 TSS 와 RSS 를 찍는다. 최소제곱은 45도 선 아래에만, 2SLS 는 위에도.
② R^2 의 분포.
③ 무엇이 좌우하는가 — 내생성 rho 가 아니라 도구의 세기 pi 다.

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/oblique_2sls.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

RNG = np.random.default_rng(0)
N, BETA, PI = 200, 1.0, 0.6
PI_W = 0.1                 # 되풀이에 쓰는 약한 도구


def draw(rng, n=N, rho=0.8, pi=PI):
    """y = beta0 + beta*x + u, x = pi*z + v, corr(u, v) = rho."""
    z = rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1.0, rho], [rho, 1.0]], size=n)
    u, v = e[:, 0], e[:, 1]
    x = pi * z + v
    y = 0.5 + BETA * x + u
    X = np.column_stack([np.ones(n), x])
    Z = np.column_stack([np.ones(n), z])
    return y, X, Z


def draw0(rng, n=N, rho=0.9, pi=PI):
    """beta = 0 — x 는 아무 효과가 없는데 u 와 상관되어 있다."""
    z = rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1.0, rho], [rho, 1.0]], size=n)
    u, v = e[:, 0], e[:, 1]
    x = pi * z + v
    y = 0.5 + 0.0 * x + u
    return y, np.column_stack([np.ones(n), x]), np.column_stack([np.ones(n), z])


def hat_ols(X):
    return X @ np.linalg.solve(X.T @ X, X.T)


def hat_2sls(X, Z):
    """yhat = H y 의 H. 멱등이지만 대칭이 아니다."""
    Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)      # Xhat = P_Z X
    return X @ np.linalg.solve(Xh.T @ X, Xh.T)


def sums(y, H):
    yh = H @ y
    r = y - yh
    tss = float(((y - y.mean()) ** 2).sum())
    rss = float((r ** 2).sum())
    mss = float(((yh - y.mean()) ** 2).sum())
    return tss, mss, rss


# === ① 한 자료에서 H 를 들여다본다 ==================================
y1, X1, Z1 = draw(RNG)
H_O, H_2 = hat_ols(X1), hat_2sls(X1, Z1)
print(f"n = {N}, 모수 {X1.shape[1]} 개 (상수항 포함)")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   모자 행렬      ||H^2-H||     ||H-H'||      tr(H)    rank(H)")
for name, H in (("최소제곱", H_O), ("2SLS", H_2)):
    print(f"   {name:<10}  {np.linalg.norm(H @ H - H):>10.2e}  "
          f"{np.linalg.norm(H - H.T):>10.2e}  {np.trace(H):>9.6f}  "
          f"{np.linalg.matrix_rank(H):>8d}")
print("   둘 다 멱등이고 tr = rank = 2 다. 갈리는 것은 대칭성뿐이다.")

for name, H in (("최소제곱", H_O), ("2SLS", H_2)):
    t, m, r = sums(y1, H)
    print(f"\n   [{name}] TSS = {t:.3f},  MSS = {m:.3f},  RSS = {r:.3f}")
    print(f"      MSS + RSS = {m + r:.3f}   (TSS 와의 차 {m + r - t:+.3f})")
    print(f"      R^2 = 1 - RSS/TSS = {1 - r / t:+.4f}")

# === ② 되풀이 =======================================================
REP = 2000
T_O, R_O, T_2, R_2 = [], [], [], []
for _ in range(REP):
    y, X, Z = draw(RNG, rho=0.9, pi=PI_W)
    t, _, r = sums(y, hat_ols(X))
    T_O.append(t); R_O.append(r)
    t, _, r = sums(y, hat_2sls(X, Z))
    T_2.append(t); R_2.append(r)
T_O, R_O, T_2, R_2 = map(np.array, (T_O, R_O, T_2, R_2))
RSQ_O, RSQ_2 = 1 - R_O / T_O, 1 - R_2 / T_2
print(f"\n② 되풀이 {REP:,} 번 (rho = 0.9, 약한 도구 pi = {PI_W})")
print("   방법          R^2 중앙값   가장 작은 R^2   R^2 < 0 인 비율   RSS > TSS")
for name, rsq in (("최소제곱", RSQ_O), ("2SLS", RSQ_2)):
    print(f"   {name:<10}  {np.median(rsq):>10.4f}  {rsq.min():>13.4f}  "
          f"{(rsq < 0).mean():>15.3f}  {(rsq < 0).sum():>10d}")

# === ③ 내생성을 키우며 ==============================================
PIS = np.linspace(0.05, 1.0, 20)
RHOS = [(0.3, "$\\rho = 0.3$"), (0.6, "$\\rho = 0.6$"), (0.9, "$\\rho = 0.9$")]
CURVE = {}
NB = 400
for rho, _ in RHOS:
    CURVE[rho] = np.array([
        float(np.mean([
            (lambda tr: 1 - tr[2] / tr[0])(
                sums(*(lambda d: (d[0], hat_2sls(d[1], d[2])))(
                    draw(RNG, rho=rho, pi=pi)))) < 0
            for _ in range(NB)]))
        for pi in PIS])
# beta = 0 — x 가 실제로는 아무 효과가 없는데 내생성 때문에 최소제곱이 속는 경우
ZERO = np.array([
    float(np.mean([
        (lambda tr: 1 - tr[2] / tr[0])(
            sums(*(lambda d: (d[0], hat_2sls(d[1], d[2])))(
                draw0(RNG, pi=pi)))) < 0
        for _ in range(NB)]))
    for pi in PIS])
print("\n③ 도구의 세기 pi 를 키우며 R^2 < 0 이 되는 비율")
print("      pi   rho=0.3   rho=0.6   rho=0.9   beta=0 일 때")
for i in (0, 2, 4, 8, 12, 19):
    print(f"   {PIS[i]:>5.2f}   {CURVE[0.3][i]:>7.3f}   {CURVE[0.6][i]:>7.3f}   "
          f"{CURVE[0.9][i]:>7.3f}   {ZERO[i]:>11.3f}")
print("   **세 rho 곡선이 거의 겹친다 — 좌우하는 것은 내생성이 아니라 도구의 세기다.**")
print(f"   beta = 0 이면 pi 와 무관하게 절반쯤 음수다 "
      f"(중앙값 {np.median(ZERO):.3f}).")

# === 그림 ============================================================
def logticks(ax, which, lo, hi, step=2):
    """로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다."""
    e = list(range(lo, hi + 1, step))
    t = [10.0 ** k for k in e]
    lab = [f"$10^{{{k}}}$" for k in e]
    (ax.set_xticks if which == "x" else ax.set_yticks)(t)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(lab)
    ax.minorticks_off()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_O, C_2 = "#2ca02c", "#d62728"

# ① 한 자료에서 피타고라스 분해가 맞아떨어지는가
TSS1, M_O, RS_O = sums(y1, H_O)
_, M_2, RS_2 = sums(y1, H_2)
ax1.bar([0], [M_O], width=0.6, color="#1f77b4", label="$\\mathrm{MSS}$")
ax1.bar([0], [RS_O], width=0.6, bottom=[M_O], color="#ff7f0e",
        label="$\\mathrm{RSS}$")
ax1.bar([1], [M_2], width=0.6, color="#1f77b4")
ax1.bar([1], [RS_2], width=0.6, bottom=[M_2], color="#ff7f0e")
ax1.axhline(TSS1, color="#333333", lw=2.4, ls="--",
            label=f"$\\mathrm{{TSS}} = {TSS1:.1f}$")
ax1.annotate("", xy=(1.42, TSS1), xytext=(1.42, M_2 + RS_2),
             arrowprops=dict(arrowstyle="<->", lw=2.0, color=C_2))
ax1.text(1.47, (TSS1 + M_2 + RS_2) / 2, f"모자란 몫\n${TSS1-M_2-RS_2:.1f}$",
         fontsize=11, color=C_2, va="center")
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["최소제곱\n(대칭 멱등)", "2SLS\n(대칭이 아닌 멱등)"])
ax1.set_xlim(-0.6, 2.5)
ax1.set_ylabel("제곱합")
ax1.set_title("① $\\mathrm{TSS} = \\mathrm{MSS} + \\mathrm{RSS}$ 인가",
              fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

LO = -2.0
n_off = int((RSQ_2 < LO).sum())
print(f"   ② 그림은 R^2 >= {LO} 만 그린다 — {n_off} 개가 왼쪽 밖이다 "
      f"(가장 작은 것 {RSQ_2.min():.1f}).")
bins = np.linspace(LO, 1.0, 80)
ax2.hist(RSQ_O, bins=bins, color=C_O, alpha=0.65, label="최소제곱")
ax2.hist(RSQ_2, bins=bins, color=C_2, alpha=0.65, label="2SLS")
ax2.axvline(0.0, color="#333333", lw=2.2, ls="--")
ax2.axvspan(bins[0], 0.0, color="#333333", alpha=0.08)
ax2.set_xlim(LO, 1.0)
ax2.text(LO * 0.96, ax2.get_ylim()[1] * 0.93,
         f"$R^2 < 0$ ({(RSQ_2 < 0).mean()*100:.1f}$\\%$)",
         fontsize=11, color="#333333")
ax2.text(LO * 0.96, ax2.get_ylim()[1] * 0.84,
         f"← {n_off} 개는 더 왼쪽", fontsize=9.5, color="#333333")
ax2.set_xlabel("$R^2 = 1 - \\mathrm{RSS}/\\mathrm{TSS}$")
ax2.set_ylabel("도수")
ax2.set_title(f"② $R^2$ 의 분포 ({REP:,} 번)", fontsize=12)
ax2.legend(loc="center left", fontsize=10)
ax2.grid(alpha=0.25)

for (rho, lab), c, mk in zip(RHOS, ("#1f77b4", C_2, "#9467bd"), ("o", "s", "^")):
    ax3.plot(PIS, CURVE[rho], mk + "-", color=c, lw=2.6, ms=6, label=lab)
ax3.plot(PIS, ZERO, "d--", color="#333333", lw=2.4, ms=6,
         label="$\\beta = 0$ ($\\rho = 0.9$)")
ax3.set_xlabel("도구의 세기 $\\pi$")
ax3.set_ylabel("$R^2 < 0$ 이 나온 비율")
ax3.set_ylim(-0.03, 0.75)
ax3.set_title("③ 내생성이 아니라 도구의 세기가 좌우한다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/square_matrices/img/oblique_2sls.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
