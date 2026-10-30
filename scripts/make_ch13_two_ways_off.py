"""13.6절 피셔의 정확검정 — 두 근사가 양쪽으로 비껴간다.

기대도수가 5 에 못 미치는 2x2 표에서 정확값은 5/143 = 0.034965 다.
보정하지 않은 카이제곱은 0.013425 로 **너무 작아** 기각하는 쪽으로 기울고,
예이츠 보정을 건 쪽은 0.050962 로 **너무 커서** 하필 0.05 를 0.001 차이로
넘어 결론이 뒤집힌다.

세 값을 알파 선 위에 올려놓으면 정확값이 가운데 있고 두 근사가 양쪽으로
벌어진 모습이 한눈에 보인다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch13/practice/img/"
ALPHA = 0.05

observed = np.array([[1, 5], [8, 2]])

p_exact = stats.fisher_exact(observed)[1]
chi_raw = stats.chi2_contingency(observed, correction=False)
chi_yat = stats.chi2_contingency(observed, correction=True)
p_raw, p_yat = chi_raw[1], chi_yat[1]
expected = chi_raw[3]

METHODS = [("보정하지 않은 χ²", p_raw, "tab:red"),
           ("피셔의 정확검정", p_exact, "tab:blue"),
           ("예이츠 보정 χ²", p_yat, "tab:orange")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.2),
                               gridspec_kw={"width_ratios": [1.5, 1]})

# ── (좌) 세 p 값과 알파 ─────────────────────────────────────────────
y = np.arange(len(METHODS))[::-1]
ax1.axvspan(0, ALPHA, color="#eef6ee", zorder=0)
ax1.axvspan(ALPHA, 0.075, color="#fdf0ee", zorder=0)
ax1.text(ALPHA - 0.002, len(METHODS) - 0.42, "기각", ha="right", va="bottom",
         fontsize=10.5, color="0.4")
ax1.text(ALPHA + 0.002, len(METHODS) - 0.42, "기각 못 함", ha="left",
         va="bottom", fontsize=10.5, color="0.4")

for yy, (name, p, c) in zip(y, METHODS):
    ax1.hlines(yy, p_exact, p, color=c, lw=2.6, alpha=0.5)
    ax1.scatter([p], [yy], s=190, c=c, zorder=5, edgecolor="white", lw=1.8)
    ratio = p / p_exact
    extra = "" if name.startswith("피셔") else f"   정확값의 {ratio:.2f} 배"
    # 알파 선을 넘어가지 않도록 오른쪽 점은 왼쪽으로 적는다
    ha = "right" if p > ALPHA else "center"
    off = -0.002 if p > ALPHA else 0.0
    ax1.text(p + off, yy + 0.21, f"{p:.6f}{extra}", ha=ha, va="bottom",
             fontsize=11, fontweight="bold", color=c)

ax1.axvline(ALPHA, color="black", ls="--", lw=1.8)
ax1.text(ALPHA, -0.38, f"α = {ALPHA}", ha="center", va="top", fontsize=11,
         bbox=dict(fc="white", ec="none", pad=1.5))

ax1.set_yticks(y, [m[0] for m in METHODS], fontsize=11)
ax1.set_xlim(0, 0.075)
ax1.set_ylim(-0.62, len(METHODS) - 0.2)
ax1.set_xlabel("양측 p 값")
ax1.set_title("정확값이 가운데, 두 근사가 양쪽으로", fontsize=12)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 왜 근사가 안 되는가 — 기대도수 ─────────────────────────────
x = np.arange(4)
ex = expected.ravel()
ob = observed.ravel().astype(float)
labs = ["(1,1)", "(1,2)", "(2,1)", "(2,2)"]
w = 0.36
ax2.bar(x - w / 2, ob, w, color="0.6", alpha=0.85, label="관측도수")
ax2.bar(x + w / 2, ex, w, color="tab:purple", alpha=0.85, label="기대도수")
ax2.axhline(5, color="tab:red", ls="--", lw=1.8)
ax2.text(-0.45, 5.15, "흔히 쓰는 조건 5", ha="left", va="bottom", fontsize=10,
         color="tab:red")
for xi, (o, e) in zip(x, zip(ob, ex)):
    ax2.text(xi + w / 2, e + 0.12, f"{e:.2f}", ha="center", va="bottom",
             fontsize=10, color="tab:purple", fontweight="bold")

ax2.set_xticks(x, labs, fontsize=11)
ax2.set_ylabel("도수")
ax2.set_ylim(0, 9.4)
ax2.set_title(f"기대도수 넷 가운데 {int((ex < 5).sum())} 개가 5 에 못 미친다",
              fontsize=12)
ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "two_ways_off.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"표 {observed.tolist()}")
print(f"  피셔 정확    p = {p_exact:.6f}  (쪽: 0.034965 = 5/143 = {5/143:.6f})")
print(f"  비보정 χ²    p = {p_raw:.6f}  (쪽: 0.013425)  정확값의 {p_raw/p_exact:.2f} 배")
print(f"  예이츠 χ²    p = {p_yat:.6f}  (쪽: 0.050962)  정확값의 {p_yat/p_exact:.2f} 배")
print(f"  기대도수 {np.round(ex, 4).tolist()}  — 5 미만이 {(ex < 5).sum()} 개")
print(f"  결론: 비보정 {'기각' if p_raw < ALPHA else '기각 못 함'} · "
      f"정확 {'기각' if p_exact < ALPHA else '기각 못 함'} · "
      f"예이츠 {'기각' if p_yat < ALPHA else '기각 못 함'}")
