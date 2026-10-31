"""9.6절 가족단위 오류율 — 검정 하나하나는 약속을 지켰다.

들머리의 신약 보기를 모의로 잰다. 효과가 전혀 없는 약을 혈압·콜레스테롤·
체중 세 평가변수로 재고 「어느 하나라도 유의하면 성공」이라 선언한다.
변수별 기각률은 저마다 0.05 를 지키는데 선언 전체의 성공률은 0.143 이다.

쪽의 fwer_growth 는 검정 수에 따른 FWER 곡선이고 fwer_bound_dependence 는
합집합 한계와 상관의 영향을 다룬다. 여기서는 곡선 대신 **한 연구를
되풀이해 센 결과**를 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/multiple_testing/img/"
M, R, A = 3, 200_000, 0.05
rng = np.random.default_rng(93)
p = rng.random((R, M))                 # 효과가 없으면 p 는 균등분포다
each = (p < A).mean(axis=0)
any_ = (p < A).any(axis=1).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# ── (좌) 두 평가변수의 p 값을 그대로 찍는다 ────────────────────────
s = p[:4000, :2]
hit = (s < A).any(axis=1)
ax1.plot(s[~hit, 0], s[~hit, 1], ".", ms=2.4, color="0.72", alpha=0.7)
ax1.plot(s[hit, 0], s[hit, 1], ".", ms=3.4, color="tab:red", alpha=0.9)
ax1.axvline(A, color="tab:red", lw=1.6)
ax1.axhline(A, color="tab:red", lw=1.6)
ax1.text(A + 0.012, 0.92, f"$p_1 < {A}$", fontsize=10.5, color="tab:red")
ax1.text(0.97, A + 0.015, f"$p_2 < {A}$", va="bottom", ha="right",
         fontsize=10.5, color="tab:red")
ax1.text(0.52, 0.52, "두 변수 다 조용한 자리", ha="center", fontsize=11.5,
         color="0.45")
ax1.text(0.03, 0.74, "ㄱ 자 모양의 이 띠가\n「어느 하나라도」다",
         fontsize=11.5, color="tab:red", fontweight="bold")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_xlabel("첫째 평가변수의 $p$ 값", fontsize=11)
ax1.set_ylabel("둘째 평가변수의 $p$ 값", fontsize=11)
ax1.set_title("효과가 없는 약을 되풀이해 재면 (연구 4,000 개)", fontsize=12.5)

# ── (우) 변수별 기각률과 선언 전체의 성공률 ────────────────────────
names = [f"{n} 평가변수" for n in ("첫째", "둘째", "셋째")] + \
    ["「어느 하나라도」\n= 연구 전체"]
vals = list(each) + [any_]
cols = ["0.55"] * M + ["tab:red"]
y = np.arange(len(names))[::-1]
ax2.barh(y, vals, height=0.56, color=cols, alpha=0.88)
for yy, v in zip(y, vals):
    ax2.text(v + 0.004, yy, f"{v:.4f}", va="center", fontsize=12,
             fontweight="bold", color="0.2")
ax2.axvline(A, color="0.25", ls="--", lw=1.8)
ax2.text(A + 0.004, len(names) - 1.42, f"약속한 {A}", ha="left", fontsize=11,
         color="0.25", fontweight="bold")
ax2.set_yticks(y, names, fontsize=11)
ax2.set_xlim(0, 0.175)
ax2.set_xlabel("잘못 기각한 비율", fontsize=11)
ax2.set_title(f"되풀이 {R:,} 번으로 센 결과", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("검정 하나하나는 약속을 지켰는데 연구 전체는 그 세 배를 썼다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "each_kept_its_promise.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  변수별 기각률 {np.round(each, 4).tolist()}")
print(f"  「어느 하나라도」 {any_:.4f}   (이론 {1-(1-A)**M:.4f})")
