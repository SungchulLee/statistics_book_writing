r"""고르느라 애쓴 손잡이보다 돌리지 않은 손잡이가 크다.

이 쪽은 King County 집값에 라쏘 경로를 그리고 lambda 를 교차검증으로 고른다.
절 8 은 「능형과 라쏘의 예측정확도가 비슷하다」고 적는다. 비슷한 정도가 아니다 —
셋(최소제곱·능형·라쏘)이 사실상 같은 자리에 있다.

그동안 이 쪽이 건드리지 않은 손잡이가 하나 있다. **반응변수를 달러로 둘 것인가
로그로 둘 것인가.** 집값은 오른쪽으로 심하게 치우친 양이다(13 장의 진단 쪽들이
이미 재어 두었다). 로그로 적합하고 두안의 스미어링(14 장)으로 되돌리면 어떻게
되는지 같은 자료에서 잰다.

① 다섯 모형의 시험 MAE — 척도 안의 차이와 척도 사이의 차이.
② lambda 경로 전체 — 달러 쪽 최선이 로그 쪽 최선에 닿지 못한다.
③ 집값 구간별 상대오차 — 왜 그런가.

만드는 파일: docs/ch18/lasso/img/scale_beats_lambda.png
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.linear_model import (LinearRegression, LassoCV,  # noqa: E402
                                  RidgeCV, lasso_path)
from sklearn.model_selection import train_test_split          # noqa: E402

URL = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/"
       "8a6d3bb6468e979c861d4b37215e1413702dfdfa/data/house_sales.csv")
PRED = ["SqFtTotLiving", "SqFtLot", "Bathrooms", "Bedrooms", "BldgGrade",
        "PropertyType", "NbrLivingUnits", "SqFtFinBasement", "YrBuilt",
        "YrRenovated", "NewConstruction"]

house = pd.read_csv(URL, sep="\t")
X = pd.get_dummies(house[PRED], drop_first=True)
X["NewConstruction"] = X["NewConstruction"].astype(int)
COLS = list(X.columns)
X = X.astype(float).values
y = house["AdjSalePrice"].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr)
A, B = sc.transform(Xtr), sc.transform(Xte)
LOGTR = np.log(ytr)


def score(pred):
    e = yte - pred
    return (float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
            float(np.median(np.abs(e / yte)) * 100))


def smear(model, Atr, Btr):
    """로그 척도 적합을 달러로 되돌린다 — 두안의 스미어링."""
    r = LOGTR - model.predict(Atr)
    return np.exp(model.predict(Btr)) * np.mean(np.exp(r))


RES = {}
RES["OLS"] = score(LinearRegression().fit(A, ytr).predict(B))
rg = RidgeCV(alphas=np.logspace(-2, 6, 60)).fit(A, ytr)
RES["능형"] = score(rg.predict(B))
la = LassoCV(n_alphas=60, cv=5, max_iter=100_000, random_state=0).fit(A, ytr)
RES["라쏘"] = score(la.predict(B))
ol = LinearRegression().fit(A, LOGTR)
SM = float(np.mean(np.exp(LOGTR - ol.predict(A))))
RES["OLS (로그)"] = score(smear(ol, A, B))
la2 = LassoCV(n_alphas=60, cv=5, max_iter=100_000,
              random_state=0).fit(A, LOGTR)
RES["라쏘 (로그)"] = score(smear(la2, A, B))

NZ1 = int((la.coef_ != 0).sum())
NZ2 = int((la2.coef_ != 0).sum())
DROP1 = [c for c, b in zip(COLS, la.coef_) if b == 0]
DROP2 = [c for c, b in zip(COLS, la2.coef_) if b == 0]

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"n = {len(y):,}, 가변수를 만든 뒤 p = {len(COLS)}. "
      "훈련 70 % / 시험 30 %, 표준화는 훈련에서만 했다.")
print(f"두안의 스미어링 인자 = {SM:.4f}\n")
print("① 시험자료에서 잰 값")
print("   모형              RMSE($)     MAE($)   중앙 절대퍼센트오차")
for k in ("OLS", "능형", "라쏘", "OLS (로그)", "라쏘 (로그)"):
    r = RES[k]
    print(f"   {k:<14} {r[0]:>10,.0f} {r[1]:>10,.0f} {r[2]:>19.2f} %")

d_reg = RES["OLS"][1] - RES["라쏘"][1]
d_scale = RES["라쏘"][1] - RES["라쏘 (로그)"][1]
print(f"\n② 두 손잡이의 값어치 (MAE 기준)")
print(f"   달러 척도 안에서 최소제곱 -> 라쏘:  {d_reg:>10,.0f} 달러 "
      f"({d_reg/RES['OLS'][1]*100:.2f} %)")
print(f"   라쏘를 달러에서 로그로 옮기면:      {d_scale:>10,.0f} 달러 "
      f"({d_scale/RES['라쏘'][1]*100:.2f} %)")
print(f"   **척도를 바꾸는 쪽이 {d_scale/d_reg:.0f} 배 크다.**")
print(f"   RMSE 로 보면 {RES['라쏘'][0]-RES['라쏘 (로그)'][0]:,.0f} 달러 "
      f"({(RES['라쏘'][0]-RES['라쏘 (로그)'][0])/RES['라쏘'][0]*100:.1f} %) 로 "
      "훨씬 작다 — 아주 비싼 집 몇 채가 제곱오차를 지배하기 때문이다.")

print(f"\n③ 라쏘가 버리는 변수")
print(f"   달러 척도: {NZ1}/{len(COLS)} 개를 남기고 버린 것은 {DROP1}")
print(f"   로그 척도: {NZ2}/{len(COLS)} 개를 남기고 버린 것은 "
      f"{DROP2 if DROP2 else '없음'}")
print("   **척도를 고치면 「해석하기 좋은 희소 모형」이 사라진다.**")

# === lambda 경로 =====================================================
LAMS1 = np.logspace(np.log10(5e4), np.log10(50), 40)
LAMS2 = np.logspace(np.log10(0.5), np.log10(5e-4), 40)
_, C1, _ = lasso_path(A, ytr, alphas=LAMS1)
_, C2, _ = lasso_path(A, LOGTR, alphas=LAMS2)
MAE1 = np.array([score(B @ C1[:, i] + ytr.mean() - A.mean(0) @ C1[:, i])[1]
                 for i in range(C1.shape[1])])
R2 = LOGTR[:, None] - (A @ C2 + LOGTR.mean() - A.mean(0) @ C2)
SMs = np.exp(R2).mean(0)
MAE2 = np.array([score(np.exp(B @ C2[:, i] + LOGTR.mean()
                              - A.mean(0) @ C2[:, i]) * SMs[i])[1]
                 for i in range(C2.shape[1])])
print(f"\n④ lambda 경로 전체에서 (두 경로 모두 큰 lambda 쪽 끝에서는 영모형이다)")
print(f"   달러 쪽 MAE 의 최솟값  {MAE1.min():>10,.0f} 달러")
print(f"   로그 쪽 MAE 의 최솟값  {MAE2.min():>10,.0f} 달러")
print(f"   **lambda 를 아무리 잘 골라도 달러 쪽은 로그 쪽 최선에 "
      f"{MAE1.min()-MAE2.min():,.0f} 달러 못 미친다.**")

# === 집값 구간별 ======================================================
pred_d = la.predict(B)
pred_l = smear(la2, A, B)
qs = np.quantile(yte, np.linspace(0, 1, 11))
MID, APE_D, APE_L = [], [], []
for i in range(10):
    m = (yte >= qs[i]) & (yte < qs[i + 1] if i < 9 else yte <= qs[i + 1])
    MID.append(np.median(yte[m]))
    APE_D.append(np.median(np.abs((yte[m] - pred_d[m]) / yte[m])) * 100)
    APE_L.append(np.median(np.abs((yte[m] - pred_l[m]) / yte[m])) * 100)
print("\n⑤ 집값 십분위별 중앙 절대퍼센트오차")
print("   십분위 중앙값($)   달러 모형   로그 모형")
for i in (0, 2, 4, 7, 9):
    print(f"   {MID[i]:>14,.0f} {APE_D[i]:>11.1f} % {APE_L[i]:>10.1f} %")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_D, C_L = "#d62728", "#2ca02c"

KEYS = ["OLS", "능형", "라쏘", "OLS (로그)", "라쏘 (로그)"]
vals = [RES[k][1] for k in KEYS]
cols = [C_D, C_D, C_D, C_L, C_L]
ax1.bar(range(5), vals, color=cols, alpha=0.8, width=0.62)
ax1.set_xticks(range(5))
ax1.set_xticklabels(["OLS", "능형", "라쏘", "OLS\n(로그)", "라쏘\n(로그)"],
                    fontsize=10.5)
ax1.set_ylabel("시험자료 MAE (달러)")
ax1.set_ylim(0, max(vals) * 1.22)
for i, v in enumerate(vals):
    ax1.text(i, v * 1.02, f"${v:,.0f}$", ha="center", fontsize=10)
ax1.annotate("", xy=(2, vals[2] * 1.12), xytext=(0, vals[0] * 1.12),
             arrowprops=dict(arrowstyle="<->", lw=1.8, color="#333333"))
ax1.text(1, vals[0] * 1.15, f"${d_reg:,.0f}$", ha="center", fontsize=10.5)
_h = max(vals) * 1.17
ax1.annotate("", xy=(4, _h), xytext=(2, _h),
             arrowprops=dict(arrowstyle="<->", lw=2.4, color=C_L))
ax1.text(3, _h * 1.01, f"${d_scale:,.0f}$", ha="center", fontsize=12,
         color=C_L)
ax1.set_title("① 어느 손잡이가 큰가", fontsize=12)
ax1.grid(alpha=0.25, axis="y")

ax2.semilogx(LAMS1 / LAMS1[0], MAE1, "o-", color=C_D, lw=2.6, ms=6,
             label="달러 척도")
ax2.semilogx(LAMS2 / LAMS2[0], MAE2, "s-", color=C_L, lw=2.6, ms=6,
             label="로그 척도 (스미어링)")
ax2.axhline(MAE2.min(), color=C_L, lw=1.6, ls=":")
ax2.set_ylim(top=min(MAE1.max(), MAE2.max()) * 1.02)
ax2.set_xlabel("$\\lambda / \\lambda_{\\max}$ (척도마다 따로 잰다)")
ax2.set_ylabel("시험자료 MAE (달러)")
# 로그 축의 음수 지수는 mathtext 를 거치므로 눈금을 직접 적는다
_xt = [1e-3, 1e-2, 1e-1, 1.0]
ax2.set_xticks(_xt)
ax2.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
ax2.minorticks_off()
ax2.set_title("② 경로 전체를 훑어도 닿지 않는다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.plot(MID, APE_D, "o-", color=C_D, lw=2.8, ms=9, label="달러 척도 라쏘")
ax3.plot(MID, APE_L, "s-", color=C_L, lw=2.8, ms=9, label="로그 척도 라쏘")
ax3.set_xscale("log")
ax3.set_xlabel("집값 십분위의 중앙값 (달러)")
ax3.set_ylabel("중앙 절대퍼센트오차 (%)")
ax3.set_title("③ 어디서 갈리는가", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/lasso/img/scale_beats_lambda.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
