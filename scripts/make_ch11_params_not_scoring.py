"""11.7절 GAM — 같은 자료에서 잰 R² 로 줄을 세우면 모수 세기가 된다.

선형(모수 6) · 이차 다항(7) · GAM(18) 의 훈련 R² 가 0.7954 → 0.8058 → 0.8305 로
좋아진다. 그런데 훈련 R² 는 모수를 더하면 오르기만 하므로 이 순서는 모수 개수의
순서와 같을 수밖에 없다.

그것을 수로 보이는 길은 **아무 뜻 없는 열**이다. 선형모형에 난수 열만 더해
모수를 18 까지 늘리면 훈련 R² 가 얼마나 오르는지를 회색 띠로 깔고, 세 모형을
그 위에 얹는다. 띠보다 위에 있으면 유연함만으로 설명되지 않는 몫이 있다는
뜻이고, 그 몫이 얼마인지는 **본 적 없는 자료에서만** 알 수 있다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = "docs/ch11/splines_gams/img/"
URL = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/"
       "8a6d3bb6468e979c861d4b37215e1413702dfdfa/data/house_sales.csv")
PRED = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
REP = 300
rng = np.random.default_rng(11)

house = pd.read_csv(URL, sep='\t')
h = house.loc[house.ZipCode == 98105, :]
y = h['AdjSalePrice'].to_numpy(float)
X0 = h[PRED].assign(const=1).to_numpy(float)
n, k0 = len(y), X0.shape[1]

r2_linear = sm.OLS(y, X0).fit().rsquared

# 쪽이 적어 둔 세 모형 — 선형은 위에서 다시 계산해 맞춰 본다
MODELS = [("선형", 6, r2_linear, "tab:blue"),
          ("이차 다항", 7, 0.8058, "tab:green"),
          ("GAM", 18, 0.8305, "tab:red")]

# 아무 뜻 없는 열만 더했을 때의 훈련 R²
ks = np.arange(k0, 25)
null_mean, null_lo, null_hi = [], [], []
for k in ks:
    vals = []
    for _ in range(REP):
        Z = np.hstack([X0, rng.standard_normal((n, k - k0))]) if k > k0 else X0
        vals.append(sm.OLS(y, Z).fit().rsquared)
    v = np.array(vals)
    null_mean.append(v.mean()); null_lo.append(np.quantile(v, .05))
    null_hi.append(np.quantile(v, .95))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5),
                               gridspec_kw={"width_ratios": [1.3, 1]})

# ── (좌) 유연함만으로 오르는 몫 ─────────────────────────────────────
ax1.fill_between(ks, null_lo, null_hi, color="0.75", alpha=0.45,
                 label="뜻 없는 난수 열만 더했을 때 (가운데 90%)")
ax1.plot(ks, null_mean, "-", color="0.45", lw=2.0)
for name, k, r2, c in MODELS:
    ax1.scatter([k], [r2], s=180, c=c, zorder=5, edgecolor="white", lw=1.8)
    # 왼쪽 끝 점은 가운데 맞춤이면 잘린다
    dx, ha = ((14, "left") if k == MODELS[0][1] else (0, "center"))
    ax1.annotate(f"{name}\n모수 {k} · R² {r2:.4f}", (k, r2),
                 textcoords="offset points", xytext=(dx, 14), ha=ha,
                 fontsize=10.5, color=c, fontweight="bold")
ax1.plot([m[1] for m in MODELS], [m[2] for m in MODELS], "--", color="0.6",
         lw=1.4, zorder=3)

ax1.set_xlabel("모수의 개수")
ax1.set_ylabel("같은 자료에서 잰 훈련 R²")
ax1.set_title(f"관측 {n} 개 — 모수를 더하면 훈련 R² 는 오르기만 한다",
              fontsize=11.5)
ax1.set_ylim(0.785, 0.858)
ax1.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 모수 하나가 떠맡는 관측의 수 ───────────────────────────────
x = np.arange(len(MODELS))
per = [n / m[1] for m in MODELS]
ax2.bar(x, per, width=0.52, color=[m[3] for m in MODELS], alpha=0.85)
for xi, (m, v) in zip(x, zip(MODELS, per)):
    ax2.text(xi, v + 1.4, f"{v:.0f} 개", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color=m[3])
    ax2.text(xi, 2.0, f"모수 {m[1]}", ha="center", va="bottom", fontsize=10,
             color="white", fontweight="bold")
ax2.set_xticks(x, [m[0] for m in MODELS], fontsize=11)
ax2.set_ylabel("모수 하나가 떠맡는 관측의 수")
ax2.set_ylim(0, 62)
ax2.set_title("순서는 같고 방향은 반대다", fontsize=11.5)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "params_not_scoring.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"n = {n}")
print(f"  선형 R² = {r2_linear:.4f}  (쪽: 0.7954)")
for name, k, r2, _ in MODELS:
    j = int(np.where(ks == k)[0][0]) if k in ks else None
    nm = f"{null_mean[j]:.4f}" if j is not None else "—"
    print(f"  {name:8s} 모수 {k:>2}  R² {r2:.4f}   뜻 없는 열만으로는 {nm}   "
          f"모수당 관측 {n / k:.0f}")
print(f"  선형 → GAM 의 상승 {MODELS[2][2] - MODELS[0][2]:.4f} · "
      f"뜻 없는 열만으로 오르는 몫 {null_mean[ks.tolist().index(18)] - r2_linear:.4f}")
