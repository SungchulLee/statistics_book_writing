r"""ch20 사례연구 절의 개념 그림을 그린다.

만드는 파일:
  ch20/case_studies/img/averaging_macro_vs_micro.png   거시·미시·가중 평균이 갈라지는 방식

실행:  python3 scripts/make_ch20_case_studies_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
OUT = "docs/ch20/case_studies/img/"
os.makedirs(OUT, exist_ok=True)

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"


# === 본문 보기의 혼동행렬 (행 = 실제, 열 = 예측) =========================
CM = np.array([[760, 30, 10],
               [40, 100, 10],
               [25, 5, 20]], dtype=float)
NAMES = ["A", "B", "C"]

TP = np.diag(CM)
FP = CM.sum(axis=0) - TP
FN = CM.sum(axis=1) - TP
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (P + R)
n_c = CM.sum(axis=1)
n = CM.sum()

P_macro, R_macro, F_macro = P.mean(), R.mean(), F1.mean()
acc = TP.sum() / n                      # = 미시 정밀도 = 미시 재현율 = 미시 F1
P_w = float(np.dot(n_c / n, P))
F_w = float(np.dot(n_c / n, F1))


def sweep(p_rare):
    """소수 범주 C의 지지도 비율을 p_rare 로 바꿀 때의 세 평균.

    범주별 정밀도·재현율은 고정하고 범주 크기만 바꾼다.
    A:B 비율은 원래대로 800:150 으로 유지한다.
    """
    w = np.empty((len(p_rare), 3))
    w[:, 2] = p_rare
    w[:, 0] = (1 - p_rare) * 800 / 950
    w[:, 1] = (1 - p_rare) * 150 / 950
    prec_w = w @ P
    acc_s = w @ R                       # 가중 재현율 = 정확도 = 미시 정밀도
    return prec_w, acc_s


# === 그림: 거시 vs 미시 vs 가중 ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

# --- 왼쪽: 범주별 지표와 세 평균 -----------------------------------------
x = np.arange(3)
bw = 0.26
ax1.bar(x - bw, P, bw, color=BLUE_F, edgecolor=BLUE, linewidth=1.3, label="정밀도")
ax1.bar(x, R, bw, color=ORANGE_F, edgecolor=ORANGE, linewidth=1.3, label="재현율")
ax1.bar(x + bw, F1, bw, color=GREEN_F, edgecolor=GREEN, linewidth=1.3, label="F1")

for xi, (p, r, f) in enumerate(zip(P, R, F1)):
    for dx, v in ((-bw, p), (0.0, r), (bw, f)):
        ax1.text(xi + dx, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=8.5, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                           alpha=0.9))

ax1.axhline(F_macro, color=PURPLE, ls="--", lw=1.6)
ax1.axhline(acc, color=RED, ls="-.", lw=1.6)
ax1.text(2.48, F_macro - 0.055, f"거시 F1 = {F_macro:.3f}", ha="right", va="top",
         fontsize=9.5, color=PURPLE, fontweight="bold")
ax1.text(2.48, acc + 0.02, f"미시 F1 = 정확도 = {acc:.3f}", ha="right", va="bottom",
         fontsize=9.5, color=RED, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels([f"범주 {m}\n(지지도 {int(s)})" for m, s in zip(NAMES, n_c)],
                    fontsize=10)
ax1.set_ylim(0, 1.16)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.set_ylabel("지표 값", fontsize=10.5)
ax1.set_title("범주별 지표 — 드문 범주 C가 거시평균을 끌어내린다", fontsize=11.5,
              color=INK, pad=10)
ax1.legend(loc="upper center", ncol=3, fontsize=9.5, framealpha=0.95,
           bbox_to_anchor=(0.5, 1.0))
ax1.grid(axis="y", alpha=0.25, ls=":")
ax1.set_axisbelow(True)

# --- 오른쪽: 불균형을 바꿀 때의 궤적 -------------------------------------
pr = np.linspace(0.01, 0.40, 400)
prec_w, acc_s = sweep(pr)

ax2.axhline(P_macro, color=PURPLE, ls="--", lw=2.0,
            label=f"거시 정밀도 = {P_macro:.3f} (고정)")
ax2.plot(pr, prec_w, color=BLUE, lw=2.0, label="가중 정밀도")
ax2.plot(pr, acc_s, color=RED, lw=2.0, ls="-.", label="미시 정밀도 (= 정확도)")

# 교차점
i_cross = int(np.argmin(np.abs(acc_s - P_macro)))
ax2.plot(pr[i_cross], P_macro, "o", color=INK, ms=6, zorder=5)
ax2.annotate(f"교차: 지지도 {pr[i_cross]*100:.0f}%",
             xy=(pr[i_cross], P_macro), xytext=(0.225, 0.742),
             fontsize=9, color=INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

# 본문 보기 위치 (C 의 지지도 5%)
ax2.axvline(0.05, color=MUTED, lw=1.2, ls=":")
ax2.plot([0.05, 0.05], [P_w, acc], "o", color=INK, ms=5, zorder=5)
ax2.annotate(f"본문 보기\n가중 {P_w:.3f} / 미시 {acc:.3f}",
             xy=(0.05, acc), xytext=(0.105, 0.925),
             fontsize=9, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

ax2.set_xlim(0.0, 0.42)
ax2.set_ylim(0.66, 0.99)
ax2.set_xticks(np.arange(0.0, 0.41, 0.10))
ax2.set_xticklabels([f"{v:.0%}" for v in np.arange(0.0, 0.41, 0.10)])
ax2.set_xlabel("드문 범주 C의 지지도 비율", fontsize=10.5)
ax2.set_ylabel("정밀도", fontsize=10.5)
ax2.set_title("범주 크기만 바꿔도 보고되는 성능은 갈린다", fontsize=11.5,
              color=INK, pad=10)
ax2.legend(loc="lower left", fontsize=9.2, framealpha=0.95)
ax2.grid(alpha=0.25, ls=":")
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "averaging_macro_vs_micro.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

print("거시 P/R/F1 =", f"{P_macro:.4f} {R_macro:.4f} {F_macro:.4f}")
print("미시(정확도) =", f"{acc:.4f}")
print("가중 P/F1    =", f"{P_w:.4f} {F_w:.4f}")
print("교차 지지도  =", f"{pr[i_cross]:.4f}")
print("지지도 1% 에서 가중/미시 =", f"{prec_w[0]:.4f} {acc_s[0]:.4f}")
print("지지도 40% 에서 가중/미시 =", f"{prec_w[-1]:.4f} {acc_s[-1]:.4f}")
print("wrote", OUT + "averaging_macro_vs_micro.png")
