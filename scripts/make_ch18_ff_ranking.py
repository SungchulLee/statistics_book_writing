r"""프랭크와 프리드먼이 1993 년에 매긴 차례.

이 쪽과 이웃 쪽들은 「예측이 목표라면 PLS 가 우선 선택」이라고 적는다. 통계학
쪽에서 화학계량학의 도구들을 들여다본 논문이 하나 있고, 그 논문이 매긴 차례는
RR > PLS > PCR > VSS > OLS 였다. 능형이 맨 앞이다.

같은 논문이 까닭도 적는다. 셋 다 「표본 퍼짐이 작은 방향에서 멀어지도록」 계수를
줄이는데, PCR 과 PLS 는 RR 보다 더 세게 줄인다. 그리고 PLS 는 선형 축소 추정량이
아니라서, K 번째 주성분 방향에서는 오히려 **늘리기도** 한다.

① 분광학을 흉내 낸 p >> n 자료에서 다섯 방법의 시험 MSE.
② 주성분 방향마다의 축소 인자 — 능형 · PCR · PLS.
③ PLS 의 축소 인자가 1 을 넘는 자리.

만드는 파일: docs/ch18/pcr_pls/img/ff_ranking.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.cross_decomposition import PLSRegression          # noqa: E402
from sklearn.decomposition import PCA                          # noqa: E402
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression  # noqa: E402
from sklearn.model_selection import KFold                      # noqa: E402

RNG = np.random.default_rng(0)


def spectra(rng, n, p=200, k=4, snr=4.0):
    """분광학을 흉내 낸다 — 잠재요인 k 개가 매끈한 봉우리로 p 개 채널을 만든다."""
    t = np.linspace(0, 1, p)
    L = np.stack([np.exp(-0.5 * ((t - c) / w) ** 2)
                  for c, w in zip(np.linspace(0.15, 0.85, k),
                                  np.linspace(0.05, 0.12, k))])
    S = rng.normal(size=(n, k)) * np.array([3.0, 2.0, 1.2, 0.7])
    X = S @ L + 0.05 * rng.normal(size=(n, p))
    beta_s = np.array([1.0, -1.5, 0.0, 2.0])      # 셋째 요인은 y 와 무관
    mu = S @ beta_s
    y = mu + rng.normal(size=n) * mu.std() / snr
    return X, y


def fit_eval(Xtr, ytr, Xte, yte, kmax=12):
    """다섯 방법을 교차검증으로 맞추고 시험 MSE 를 돌려준다."""
    mx, sx, my = Xtr.mean(0), Xtr.std(0) + 1e-12, ytr.mean()
    A, B = (Xtr - mx) / sx, (Xte - mx) / sx
    c = ytr - my
    out = {}
    out["OLS"] = np.mean((yte - (B @ np.linalg.pinv(A) @ c + my)) ** 2)
    r = RidgeCV(alphas=np.logspace(-3, 5, 60)).fit(A, c)
    out["능형"] = np.mean((yte - (r.predict(B) + my)) ** 2)
    lo = LassoCV(n_alphas=60, cv=5, max_iter=50_000, random_state=0).fit(A, c)
    out["라쏘"] = np.mean((yte - (lo.predict(B) + my)) ** 2)
    kf = KFold(5, shuffle=True, random_state=0)
    for name, mk in (("PCR", "pcr"), ("PLS", "pls")):
        best, bk = np.inf, 1
        for k in range(1, kmax + 1):
            e = []
            for i_tr, i_va in kf.split(A):
                if mk == "pcr":
                    pc = PCA(n_components=k).fit(A[i_tr])
                    Z = pc.transform(A[i_tr])
                    g = np.linalg.lstsq(Z, c[i_tr], rcond=None)[0]
                    pred = pc.transform(A[i_va]) @ g
                else:
                    pl = PLSRegression(n_components=k, scale=False)
                    pl.fit(A[i_tr], c[i_tr])
                    pred = pl.predict(A[i_va]).ravel()
                e.append(np.mean((c[i_va] - pred) ** 2))
            if np.mean(e) < best:
                best, bk = np.mean(e), k
        if mk == "pcr":
            pc = PCA(n_components=bk).fit(A)
            g = np.linalg.lstsq(pc.transform(A), c, rcond=None)[0]
            pred = pc.transform(B) @ g
        else:
            pl = PLSRegression(n_components=bk, scale=False).fit(A, c)
            pred = pl.predict(B).ravel()
        out[name] = np.mean((yte - (pred + my)) ** 2)
        out[name + "_k"] = bk
    return out


# === ① p >> n 에서의 시험 MSE =======================================
NTR, NTE, P, REP = 50, 2000, 200, 60
METH = ["OLS", "능형", "PLS", "PCR", "라쏘"]
RES = {m: [] for m in METH}
KS_PCR, KS_PLS = [], []
for _ in range(REP):
    X, y = spectra(RNG, NTR + NTE, p=P)
    o = fit_eval(X[:NTR], y[:NTR], X[NTR:], y[NTR:])
    for m in METH:
        RES[m].append(o[m])
    KS_PCR.append(o["PCR_k"]); KS_PLS.append(o["PLS_k"])
MED = {m: float(np.median(RES[m])) for m in METH}

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"① 분광학을 흉내 낸 자료 — 훈련 {NTR}, 시험 {NTE}, 변수 {P}, "
      f"되풀이 {REP}")
print("   방법     시험 MSE 중앙값   능형 대비")
base = MED["능형"]
for m in sorted(METH, key=lambda z: MED[z]):
    print(f"   {m:<7}   {MED[m]:>14.4f}   {MED[m]/base:>8.2f}")
print(f"   PCR 이 고른 성분 수 중앙값 {np.median(KS_PCR):.1f}, "
      f"PLS 는 {np.median(KS_PLS):.1f} (참 요인은 4, 그 가운데 y 와 관련된 것은 3)")
order = " > ".join(sorted(METH, key=lambda z: MED[z]))
print(f"   이 자료의 차례: {order}")

# === ②③ 축소 인자 ==================================================
# 축소 인자는 OLS 를 기준으로 재므로 p < n 인 설계를 따로 쓴다
N2, P2 = 120, 20
X2, y2 = spectra(np.random.default_rng(5), N2, p=P2, k=4)
X2 = (X2 - X2.mean(0)) / (X2.std(0) + 1e-12)
y2 = y2 - y2.mean()
U, D, Vt = np.linalg.svd(X2, full_matrices=False)
b_ols = np.linalg.lstsq(X2, y2, rcond=None)[0]
proj_ols = Vt @ b_ols                      # 주성분 방향마다의 OLS 성분


def z(v, tol=5e-4):
    """-1e-17 이 «-0.000» 으로 찍히는 것을 막는다."""
    return 0.0 if abs(v) < tol else v


def shrink(beta):
    return (Vt @ beta) / np.where(np.abs(proj_ols) < 1e-12, np.nan, proj_ols)


LAM = 20.0
b_rr = np.linalg.solve(X2.T @ X2 + LAM * np.eye(P2), X2.T @ y2)
KPC = 4
pc = PCA(n_components=KPC).fit(X2)
g = np.linalg.lstsq(pc.transform(X2), y2, rcond=None)[0]
b_pcr = pc.components_.T @ g
F_RR, F_PCR = shrink(b_rr), shrink(b_pcr)
F_PLS = {}
for k in (2, 4, 6):
    pl = PLSRegression(n_components=k, scale=False).fit(X2, y2)
    F_PLS[k] = shrink(pl.coef_.ravel())

print(f"\n② 주성분 방향마다의 축소 인자 (n = {N2}, p = {P2}, "
      f"능형 lambda = {LAM:g}, 성분 {KPC} 개)")
print("    j   특잇값 d_j    능형     PCR    PLS(4)")
for j in range(min(8, P2)):
    print(f"   {j+1:>2}   {D[j]:>9.2f}   {z(F_RR[j]):>6.3f}   {z(F_PCR[j]):>6.3f}   "
          f"{z(F_PLS[4][j]):>7.3f}")
print("   능형은 매끄럽게 줄이고 PCR 은 1 과 0 으로 끊는다.")

print("\n③ PLS 의 축소 인자가 1 을 넘는 자리")
for k in (2, 4, 6):
    f = F_PLS[k]
    over = np.where(f > 1.0)[0] + 1
    print(f"   성분 {k} 개: 1 을 넘는 방향 {list(over[:6])}, "
          f"가장 큰 값 {np.nanmax(f):.3f}")
print("   **PLS 는 선형 축소 추정량이 아니라서 어떤 방향은 늘린다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
CMAP = {"OLS": "#7f7f7f", "능형": "#d62728", "PLS": "#2ca02c",
        "PCR": "#1f77b4", "라쏘": "#9467bd"}

ordm = sorted(METH, key=lambda z: MED[z])
bp = ax1.boxplot([RES[m] for m in ordm], labels=ordm, showfliers=False,
                 patch_artist=True, widths=0.6)
for patch, m in zip(bp["boxes"], ordm):
    patch.set_facecolor(CMAP[m]); patch.set_alpha(0.65)
for med in bp["medians"]:
    med.set_color("#222222"); med.set_linewidth(2.2)
ax1.axhline(MED["능형"], color="#d62728", lw=1.6, ls=":")
ax1.set_ylabel("시험 MSE")
ax1.set_title(f"① $p = {P} \\gg n = {NTR}$ 에서 ({REP} 번)", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

jj = np.arange(1, P2 + 1)
ax2.plot(jj, F_RR, "o-", color=CMAP["능형"], lw=2.8, ms=8, label="능형")
ax2.plot(jj, F_PCR, "s-", color=CMAP["PCR"], lw=2.8, ms=8,
         label=f"PCR ($K = {KPC}$)")
ax2.plot(jj, F_PLS[4], "^-", color=CMAP["PLS"], lw=2.8, ms=8,
         label="PLS ($K = 4$)")
ax2.axhline(1.0, color="#333333", lw=2.0, ls="--")
ax2.set_xlabel("주성분 방향 $j$ (퍼짐이 큰 쪽부터)")
ax2.set_ylabel("축소 인자")
ax2.set_xlim(0.5, 10.5)
ax2.set_title("② 어느 방향을 얼마나 줄이는가", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

for k, c, mk in zip((2, 4, 6), ("#1f77b4", "#2ca02c", "#9467bd"),
                    ("o", "^", "s")):
    ax3.plot(jj, F_PLS[k], mk + "-", color=c, lw=2.6, ms=8,
             label=f"PLS ($K = {k}$)")
ax3.axhline(1.0, color="#333333", lw=2.4, ls="--")
ax3.text(3.4, 2.45, "$1$ — 줄이지도 늘리지도 않음", fontsize=10.5,
         color="#333333")
_jpk = int(np.nanargmax(F_PLS[6]))
ax3.annotate(f"${np.nanmax(F_PLS[6]):.2f}$ 배로 **늘린다**".replace("**", ""),
             (jj[_jpk], F_PLS[6][_jpk]), xytext=(0.7, -1.2),
             textcoords="offset fontsize", fontsize=11, color="#9467bd")
ax3.set_xlabel("주성분 방향 $j$")
ax3.set_ylabel("축소 인자")
ax3.set_xlim(0.5, 10.5)
ax3.set_title("③ PLS 는 늘리기도 한다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/pcr_pls/img/ff_ranking.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
