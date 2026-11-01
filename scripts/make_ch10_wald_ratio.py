"""10.5절 도구변수 — 그 13% 는 교육의 값이 아니다.

들머리의 첫 두 문단을 그린다. 교육 연수와 로그임금에 직선을 그으면
기울기가 0.130 인데, 그 가운데 0.030 은 「원래 그런 사람이었다」는 몫이다.
도구변수는 그 몫을 걷어내고 0.100 을 집어낸다. 나눗셈 한 번이 전부다 ---
임금 차이 0.080 을 교육 차이 0.800 으로 나눈다.

자료는 이 책이 만든 것이고, 쪽 연습문제 5 의 표에 **정확히** 맞도록
짜 넣었다. 네 수 (13.60, 2.200, 12.80, 2.120) 이 표본에서 소수 자리까지
그대로 나오고 최소제곱 기울기도 정확히 0.130 이다. 그렇게 만드는 길은
셋이다.

  ① 능력 U 와 교육의 나머지 변동을 **거주 집단 안에서** 평균 0 으로
     맞춘다 --- 그러면 교육 연수의 집단별 평균이 설계값 그대로 나온다.
  ② 임금 잡음을 1 · z · X 셋 **모두에** 직교시킨다. 1 과 z 에 직교하면
     집단별 평균이 0 이 되어 임금의 집단별 평균이 설계값대로 나오고,
     X 에도 직교해야 최소제곱 기울기가 딱 떨어진다. 집단 안에서만
     중심화한 뒤 X 에 직교시키면 중심화가 다시 깨진다.
  ③ 능력의 임금 효과 delta 를 표본 공분산에서 거꾸로 풀어 기울기가
     정확히 0.130 이 되게 한다.

쪽의 wright_identification 은 수요·공급 교점에서 기울기가 식별되지 않는
모습을, iv_vs_ols_strength 는 도구가 약해질 때의 손익을 그린다. 여기서는
그 둘이 건드리지 않는 것 --- 월드 추정량이 그림에서 어떤 선인지 --- 만
본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch10/causation/img/"

N = 1200                     # 짝수, 두 집단에 절반씩
BETA = 0.100                 # 교육 한 해의 참 인과효과
OLS_TARGET = 0.130           # 쪽이 적은 최소제곱 기울기
X_FAR, X_NEAR = 12.80, 13.60      # 쪽 연습문제 5 의 교육 연수
Y_FAR, Y_NEAR = 2.120, 2.200      # 그 표의 로그임금
CONST = Y_FAR - BETA * X_FAR      # 설계가 요구하는 절편

rng = np.random.default_rng(31)
z = np.repeat([0, 1], N // 2)                      # 1 = 대학 가까이 산다


def center_within(v):
    """거주 집단 안에서 표본평균을 정확히 0 으로 맞춘다."""
    out = v.astype(float).copy()
    for g in (0, 1):
        m = z == g
        out[m] -= out[m].mean()
    return out


u = center_within(rng.normal(0, 1, N))             # 재지 않은 능력
ex = center_within(rng.normal(0, 1.2, N))          # 교육의 나머지 변동
x = np.where(z == 1, X_NEAR, X_FAR) + u + ex       # 교육 연수

# 임금 잡음: 1, z, x 셋 모두에 직교시킨다. 1 과 z 에 직교하면 거주 집단별
# 평균이 정확히 0 이 되고, x 에도 직교해야 최소제곱 기울기가 딱 떨어진다.
# (집단 안에서만 중심화한 뒤 x 에 직교시키면 중심화가 다시 깨진다)
ey = rng.normal(0, 0.25, N)
D = np.column_stack([np.ones(N), z.astype(float), x])
ey = ey - D @ np.linalg.lstsq(D, ey, rcond=None)[0]
xc = x - x.mean()

# 최소제곱 기울기가 정확히 OLS_TARGET 이 되도록 능력의 임금 효과를 푼다
uc = u - u.mean()
delta = (OLS_TARGET - BETA) * (xc @ xc) / (uc @ xc)
y = CONST + BETA * x + delta * u + ey              # 로그임금

ols = (xc @ (y - y.mean())) / (xc @ xc)
xn, xf = x[z == 1].mean(), x[z == 0].mean()
yn, yf = y[z == 1].mean(), y[z == 0].mean()
iv = (yn - yf) / (xn - xf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 두 직선 ───────────────────────────────────────────────────
for g, col, name in ((0, "tab:orange", "대학에서 멀리 산다"),
                     (1, "tab:blue", "대학 가까이 산다")):
    m = z == g
    ax1.scatter(x[m], y[m], s=13, color=col, alpha=0.35,
                edgecolors="none", label=name)
gx = np.linspace(x.min() - 0.3, x.max() + 0.3, 2)
ax1.plot(gx, y.mean() + ols * (gx - x.mean()), "-", lw=3.0, color="tab:red",
         label=f"최소제곱 — 기울기 {ols:.3f}")
ax1.plot(gx, yf + iv * (gx - xf), "-", lw=3.0, color="tab:green",
         label=f"도구변수 — 기울기 {iv:.3f}")
for xm, ym, col, dy, va in ((xf, yf, "tab:orange", -0.08, "top"),
                            (xn, yn, "tab:blue", 0.08, "bottom")):
    ax1.plot([xm], [ym], "o", ms=15, color="white", mec=col, mew=3.0,
             zorder=6)
    ax1.text(xm, ym + dy, f"({xm:.2f},  {ym:.3f})", ha="center", va=va,
             fontsize=11, color=col, fontweight="bold")
ax1.set_xlim(9.0, 17.4)
ax1.set_ylim(1.55, 2.80)
ax1.set_xlabel("교육 연수", fontsize=11)
ax1.set_ylabel("로그임금", fontsize=11)
ax1.set_title(f"같은 자료에 그은 두 직선 (n = {N:,})", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)
ax1.text(0.985, 0.03, "초록 직선은 동그라미 두 개를 이은 것이다\n"
         "— 그것이 나눗셈 한 번의 뜻이다", transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=11, fontweight="bold",
         color="tab:green",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="tab:green", lw=1.0))

# ── (우) 나눗셈 한 번과 그 차이 ────────────────────────────────────
ax2.bar([0], [iv], width=0.5, color="tab:green", alpha=0.85,
        label=f"교육이 준 몫 {iv:.3f}")
ax2.bar([0], [ols - iv], width=0.5, bottom=[iv], color="tab:red", alpha=0.55,
        label=f"「원래 그런 사람이었다」는 몫 {ols - iv:.3f}")
ax2.text(0, ols + 0.004, f"최소제곱이 보고하는 {ols:.3f}", ha="center",
         va="bottom", fontsize=12, fontweight="bold", color="0.15")
ax2.text(0, iv / 2, f"{iv:.3f}", ha="center", va="center", fontsize=14,
         fontweight="bold", color="white")
ax2.text(0, iv + (ols - iv) / 2, f"{ols - iv:.3f}", ha="center", va="center",
         fontsize=13, fontweight="bold", color="0.15")
ax2.text(0.42, iv / 2, f"교육 한 해에\n임금 {100*iv:.0f}%", ha="left",
         va="center", fontsize=11.5, color="tab:green", fontweight="bold")
ax2.text(0.42, iv + (ols - iv) / 2,
         f"참값의 {100*(ols - iv)/iv:.0f}% 를\n더 얹은 것이다", ha="left",
         va="center", fontsize=11.5, color="tab:red", fontweight="bold")
ax2.text(0, -0.016, f"임금 차이 {yn - yf:.3f}\n나누기 교육 차이 "
         f"{xn - xf:.3f}\n= {iv:.3f}", ha="center", va="top", fontsize=12,
         fontweight="bold", color="tab:green",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="tab:green", lw=1.2))
ax2.set_xlim(-0.55, 1.15)
ax2.set_ylim(-0.085, 0.165)
ax2.set_xticks([])
ax2.set_yticks([0.0, 0.05, 0.10, 0.15])
ax2.set_ylabel("교육 한 해에 붙는 로그임금", fontsize=11)
ax2.set_title("나눗셈 한 번이 걷어내는 것", fontsize=12.5)
ax2.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("그 13% 는 교육의 값이 아니다 — 안에 3 할이 섞여 있다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "wald_ratio.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n이 책이 만든 자료 n = {N:,} (쪽 연습문제 5 의 표에 맞춰 짰다)")
print(f"  대학 가까이: 교육 {xn:.4f} 년 · 로그임금 {yn:.4f}  "
      f"(표의 {X_NEAR}, {Y_NEAR})")
print(f"  대학에서 멀리: 교육 {xf:.4f} 년 · 로그임금 {yf:.4f}  "
      f"(표의 {X_FAR}, {Y_FAR})")
print(f"  최소제곱 기울기 {ols:.6f}   (쪽이 적은 {OLS_TARGET})")
print(f"  도구변수 = ({yn:.4f} - {yf:.4f}) / ({xn:.4f} - {xf:.4f}) "
      f"= {yn - yf:.4f} / {xn - xf:.4f} = {iv:.6f}")
print(f"  섞여 있던 몫 {ols - iv:.6f} = 참값의 "
      f"{(ols - iv) / iv:.4f} 배")
print(f"  (자료를 그렇게 만들었다: 참 인과효과 {BETA}, 능력의 임금 효과 "
      f"{delta:.4f})")
