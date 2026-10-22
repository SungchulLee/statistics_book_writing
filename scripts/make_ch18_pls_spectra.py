r"""PLS 가 통계학 바깥에서 태어난 까닭 — 화학이 갖고 있던 문제의 모양.

분광 보정 문제를 흉내 낸다. 파장 $240$ 곳에서 잰 스펙트럼 $60$ 개로 분석물의
농도를 맞혀야 한다. 바탕선 흔들림·산란·다른 화합물이 신호보다 훨씬 커서,
**찾는 성분은 전체 분산의 만분의 일**에 지나지 않는다.

① 스펙트럼 $60$ 개와, 그 안에 든 분석물 기여분(눈금을 $100$ 배 키워야 보인다).
② 주성분마다 (담은 분산, $y$ 와의 상관)을 찍는다. 둘이 전혀 같은 순서가 아니다.
③ 성분 수에 따른 시험 RMSE — PCR 과 PLS.

만드는 파일: docs/ch18/pcr_pls/img/pls_spectra.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.cross_decomposition import PLSRegression   # noqa: E402
from sklearn.decomposition import PCA                   # noqa: E402
from sklearn.linear_model import LinearRegression       # noqa: E402

P, NTR, NTE = 240, 60, 800
SD = 0.015
RNG = np.random.default_rng(11)

w = np.arange(P)
gauss = lambda m, s: np.exp(-0.5 * ((w - m) / s) ** 2)   # noqa: E731
S_AN = gauss(70, 4.0)                       # 재려는 분석물의 좁은 봉우리
NUIS = [np.ones(P), (w / P - 0.5), (w / P - 0.5) ** 2,
        np.exp(-w / 70), np.exp(-(P - w) / 70),
        gauss(150, 22), gauss(30, 16), gauss(200, 30)]
AMP = [3.0, 2.4, 2.0, 1.7, 1.5, 1.2, 1.0, 0.9]
AMP_AN = 0.30


def make(n):
    c = RNG.uniform(0.2, 1.0, n)
    X = np.outer(c, AMP_AN * S_AN) + RNG.normal(0, SD, (n, P))
    for s, a in zip(NUIS, AMP):
        X = X + np.outer(RNG.normal(0, 1, n), a * s)
    return X, c


Xtr, ytr = make(NTR)
Xte, yte = make(NTE)
mu = Xtr.mean(axis=0)
Xtr0, Xte0 = Xtr - mu, Xte - mu
ym = ytr.mean()
rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))   # noqa: E731

print(f"① 설계행렬 {Xtr.shape} — 변수가 표본의 {P // NTR} 배")
share = 100 * np.outer(ytr, AMP_AN * S_AN).var() / Xtr0.var()
print(f"   분석물 봉우리가 X 의 전체 분산에서 차지하는 몫 : {share:.4f}%")

pca = PCA(n_components=NTR - 1).fit(Xtr0)
Ztr, Zte = pca.transform(Xtr0), pca.transform(Xte0)
ev = pca.explained_variance_ratio_
cor = np.array([abs(np.corrcoef(Ztr[:, i], ytr)[0, 1]) for i in range(NTR - 1)])
k = int(cor.argmax())

print(f"\n② 주성분 — 분산 순위와 쓸모가 어긋난다")
print(f"{'성분':>5}{'담은 분산':>12}{'y 와의 상관':>13}")
for i in list(range(6)) + [k, k + 1]:
    mark = "   <- 신호" if i == k else ""
    print(f"{i + 1:>5}{100 * ev[i]:>11.4f}%{cor[i]:>13.3f}{mark}")
print(f"   신호를 담은 것은 {k + 1} 번째 성분이고 "
      f"분산 순위로는 꼴찌에 가깝다 ({100 * ev[k]:.4f}%).")
print(f"   그 앞의 일곱 성분이 분산의 "
      f"{100 * ev[:k].sum():.4f}% 를 쓸어 가는데 상관은 모두 "
      f"{cor[:k].max():.2f} 이하다.")

print(f"\n③ 성분 수에 따른 시험 RMSE (y 를 평균으로 두면 "
      f"{rmse(np.full(NTE, ym), yte):.4f})")
MS = np.arange(1, 21)
e_pcr, e_pls = [], []
for M in MS:
    lr = LinearRegression().fit(Ztr[:, :M], ytr)
    e_pcr.append(rmse(lr.predict(Zte[:, :M]), yte))
    pl = PLSRegression(n_components=M, scale=False).fit(Xtr0, ytr - ym)
    e_pls.append(rmse(pl.predict(Xte0).ravel() + ym, yte))
e_pcr, e_pls = np.array(e_pcr), np.array(e_pls)
print(f"{'M':>3}{'PCR':>10}{'PLS':>10}")
for i, M in enumerate(MS):
    if M <= 10 or M % 5 == 0:
        print(f"{M:>3}{e_pcr[i]:>10.4f}{e_pls[i]:>10.4f}")
m_pcr = int(MS[np.argmax(e_pcr < 0.1)])
m_pls = int(MS[np.argmax(e_pls < 0.1)])
print(f"   절벽 — PCR 은 {m_pcr} 개, PLS 는 {m_pls} 개에서 떨어진다.")
print("   PLS 도 공분산(상관 x 표준편차)을 최대로 하므로, 분산이 큰 쪽에")
print("   똑같이 끌려간다. 여기서는 둘이 같은 자리에서 떨어진다.")

# === 그림 ============================================================
fig = plt.figure(figsize=(15.6, 5.1))
gs = fig.add_gridspec(1, 3, wspace=0.27)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

for i in range(NTR):
    ax1.plot(w, Xtr[i], color="#1f77b4", lw=0.6, alpha=0.45)
ax1.set_xlabel("파장 (칸 번호)")
ax1.set_ylabel("흡광도")
ax1.set_title(f"① 스펙트럼 ${NTR}$ 개 · 파장 ${P}$ 곳", fontsize=12)
ax1.grid(alpha=0.25)
ins = ax1.inset_axes([0.46, 0.60, 0.50, 0.36])
for i in range(NTR):
    ins.plot(w, ytr[i] * AMP_AN * S_AN, color="#d62728", lw=0.7, alpha=0.6)
ins.set_xlim(30, 115)
ins.set_xticks([50, 70, 100])
ins.set_xticklabels(["$50$", "$70$", "$100$"], fontsize=8)
ins.tick_params(labelsize=8)
ins.set_title("답이 든 자리만 떼어 본 것", fontsize=9, pad=2)
ax1.axvline(70, color="#d62728", lw=1.2, ls=":")
ax1.text(78, ax1.get_ylim()[0] * 0.86,
         f"답이 든 파장 — 여기가\n전체 분산의 ${share:.4f}\\%$ 다",
         fontsize=10, color="#d62728", ha="left", va="bottom")

ax2.scatter(100 * ev[:30], cor[:30], s=70, color="#999999", zorder=3)
ax2.scatter([100 * ev[k]], [cor[k]], s=190, color="#d62728", zorder=4,
            edgecolor="white", linewidth=1.4)
for i in range(3):
    ax2.annotate(f"${i + 1}$", (100 * ev[i], cor[i]), fontsize=10,
                 xytext=(0, 1.0), textcoords="offset fontsize", ha="center")
ax2.annotate(f"${k + 1}$ 번째 성분\n분산 ${100 * ev[k]:.4f}\\%$, "
             f"상관 ${cor[k]:.2f}$",
             xy=(100 * ev[k], cor[k]), xytext=(2.4e-3, 0.46),
             textcoords="data", fontsize=10.5, color="#d62728", ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
ax2.set_xscale("log")
# 로그 눈금의 마이너스는 mathtext 라 unicode_minus 로 막을 수 없다 — 직접 적는다
ax2.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
ax2.set_xticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$",
                     "$1$", "$10$", "$100$"], fontsize=9.5)
ax2.minorticks_off()
ax2.set_xlim(2e-5, 4e2)
ax2.set_xlabel("그 성분이 담은 분산 ($\\%$)")
ax2.set_ylabel("$y$ 와의 상관의 절댓값")
ax2.set_ylim(-0.04, 0.93)
ax2.set_title("② 분산 순위와 쓸모가 어긋난다", fontsize=12)
ax2.grid(alpha=0.25)

ax3.plot(MS, e_pcr, "o-", color="#1f77b4", lw=2.3, ms=6, label="PCR")
ax3.plot(MS, e_pls, "s-", color="#d62728", lw=2.3, ms=6, label="PLS")
ax3.axhline(rmse(np.full(NTE, ym), yte), color="black", lw=1.3, ls="--")
ax3.text(20, rmse(np.full(NTE, ym), yte) * 1.07, "$y$ 를 평균으로 두면",
         fontsize=10, ha="right")
ax3.axvline(k + 1, color="#2ca02c", lw=1.4, ls=":")
ax3.annotate(f"${k + 1}$ 개째에서 절벽", xy=(k + 1, 0.06), xytext=(11.5, 0.115),
             textcoords="data", fontsize=10.5, color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1))
ax3.set_yscale("log")
ax3.set_xticks([1, 5, 8, 10, 15, 20])
ax3.set_xticklabels(["$1$", "$5$", "$8$", "$10$", "$15$", "$20$"])
ax3.set_yticks([0.02, 0.05, 0.1, 0.2])
ax3.set_yticklabels(["$0.02$", "$0.05$", "$0.1$", "$0.2$"])
ax3.minorticks_off()
ax3.set_xlabel("쓴 성분의 수 $M$")
ax3.set_ylabel("시험 RMSE")
ax3.set_title("③ 절벽은 여덟 개째에 온다", fontsize=12)
ax3.legend(loc="center right", fontsize=10.5)
ax3.grid(alpha=0.25)

out = "docs/ch18/pcr_pls/img/pls_spectra.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
