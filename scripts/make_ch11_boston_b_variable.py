r"""13.9절 「그림으로 보는 통계학」 — 그 자료를 부르면 오류 메시지가 뜬다.

Boston 주택 자료의 열 하나는 $B = 1000(\mathrm{Bk} - 0.63)^2$ 이다($\mathrm{Bk}$ 는
읍의 흑인 인구 비율). 제곱이라 **되돌릴 수 없고**, 회귀에 그대로 넣으면 집값이
$\mathrm{Bk} = 0.63$ 에서 가장 낮고 양쪽으로 오른다는 **가정**을 설계행렬에 새겨 넣는다.
scikit-learn 은 $1.2$ 판에서 이 자료의 적재 함수를 지웠다.

① 변환 — $B$ 하나에 $\mathrm{Bk}$ 가 둘.
② 자료의 $B$ 분포 — 네 읍에 하나가 한 점에 몰려 있다.
③ 그 열이 회귀에서 뜻하는 것 — 추정한 것이 아니라 가정한 모양.

만드는 파일: docs/ch13/package_usage/img/boston_b_variable.png
실행:  python3 scripts/make_ch13_boston_b_variable.py   (저장소 최상위에서)
필요:  인터넷(쪽과 같은 고정 커밋의 Rdatasets 사본을 내려받는다).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/package_usage/img/boston_b_variable.png"
URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
       "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/MASS/Boston.csv")

df = pd.read_csv(URL)
B = df["black"].to_numpy()
y = df["medv"].to_numpy()
cols = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis",
        "rad", "tax", "ptratio", "black", "lstat"]
X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])

beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta
s2 = res @ res / (len(y) - X.shape[1])
se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
j = 1 + cols.index("black")
bB, seB = beta[j], se[j]
print(f"자료 {df.shape[0]} 행.  B 계수 = {bB:.5f} (SE {seB:.5f}, t = {bB/seB:.2f})")

TOP = 1000 * 0.63 ** 2          # Bk = 0 일 때의 값 = 396.9
AMBIG = 1000 * (1 - 0.63) ** 2  # Bk = 1 일 때의 값 = 136.9
n_top = int(np.isclose(B, TOP).sum())
n_amb = int((B <= AMBIG).sum())
print(f"B 의 최댓값 {B.max():.1f} (= Bk 0),  거기 몰린 읍 {n_top} 개 "
      f"({100*n_top/len(B):.1f}%)")
print(f"Bk 가 둘 다 가능한 구간 B ≤ {AMBIG:.1f} 에 든 읍 {n_amb} 개 "
      f"({100*n_amb/len(B):.1f}%)")
print(f"medv 가 정확히 50.0 인 행 {int(np.isclose(y, 50).sum())} 개")

for bk in (0.0, 0.2, 0.4, 0.63, 0.8, 1.0):
    print(f"  Bk = {bk:.2f} → B = {1000*(bk-0.63)**2:6.1f} → "
          f"집값 기여 {bB*1000*(bk-0.63)**2:+.3f} 천달러")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2 = "#1f6f8b", "#c0532b"
bk = np.linspace(0, 1, 400)
curve = 1000 * (bk - 0.63) ** 2

ax = axes[0]
ax.plot(bk, curve, color=C1, lw=2.2)
LEV = 100.0
r1, r2 = 0.63 - np.sqrt(LEV / 1000), 0.63 + np.sqrt(LEV / 1000)
ax.axhline(LEV, color=C2, lw=1.3, ls="--")
ax.plot([r1, r2], [LEV, LEV], "o", color=C2, ms=7)
ax.annotate(f"$B = {LEV:.0f}$ 이면\nBk 는 {r1:.2f} 이거나 {r2:.2f} 다",
            xy=(r2, LEV), xytext=(0.52, 250), color=C2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C2))
ax.axhspan(0, AMBIG, color="0.85", alpha=0.6)
ax.text(0.98, AMBIG * 0.45, f"두 값이 모두 가능한 구간\n(읍 {n_amb} 개)",
        fontsize=9, color="0.35", ha="right")
ax.set_xlabel("Bk — 읍의 흑인 인구 비율")
ax.set_ylabel("자료에 실린 값  $B = 1000(\\mathrm{Bk} - 0.63)^2$")
ax.set_title("① 제곱한 뒤에는 되돌릴 수 없다", fontsize=11)

ax = axes[1]
ax.hist(B, bins=60, color="0.72", edgecolor="white")
ax.axvline(TOP, color=C2, lw=2)
ax.annotate(f"$B = {TOP:.1f}$ 에 읍 {n_top} 개\n(전체의 {100*n_top/len(B):.0f}%, Bk = 0)",
            xy=(TOP, n_top * 0.75), xytext=(60, n_top * 0.78), color=C2, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C2))
ax.set_xlabel("자료에 실린 값 B")
ax.set_ylabel("읍 수")
ax.set_title("② 네 읍에 하나가 한 점에 있다", fontsize=11)

ax = axes[2]
ax.plot(bk, bB * curve, color=C1, lw=2.2)
ax.axvline(0.63, color="0.45", ls=":", lw=1.3)
ax.text(0.645, bB * 1000 * 0.63 ** 2 * 0.80,
        "모형이 집값을 가장 낮게\n보는 자리 — Bk = 0.63", fontsize=9, color="0.35")
for p, dx, dy in ((0.0, 0.07, -0.30), (1.0, -0.33, 0.28)):
    v = bB * 1000 * (p - 0.63) ** 2
    ax.plot([p], [v], "o", color=C2, ms=7)
    ax.annotate(f"{v:+.2f} 천달러", xy=(p, v), xytext=(p + dx, v + dy),
                color=C2, fontsize=9)
ax.set_xlabel("Bk — 읍의 흑인 인구 비율")
ax.set_ylabel("집값 중앙값에 더해지는 몫 (천달러)")
ax.set_title("③ 추정한 것이 아니라 가정한 모양", fontsize=11)
ax.text(0.30, 0.52, f"$\\hat\\beta_B = {bB:.5f}$   $t = {bB/seB:.2f}$",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
