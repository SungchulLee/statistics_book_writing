r"""적률법이 초당 수십억 번 돌고 있다 — Adam 의 1차·2차 적률과 그 편향 보정.

이 쪽은 표본 적률을 모집단 적률에 맞추어 모수를 얻는다. 같은 일이 딥러닝 최적화기
안에서 돌아간다. Adam 은 기울기의 1차 적률과 2차 원적률을 지수이동평균으로 추정하고,
그 추정이 **0 으로 치우쳐 있다**는 것을 알아 나눗셈 한 번으로 바로잡는다.

기울기가 상수 g 로 일정하다고 두면 t 걸음째의 이동평균은
    m_t = (1 - beta1^t) g,    v_t = (1 - beta2^t) g^2
이므로 쌓인 몫이 각각 1-beta1^t, 1-beta2^t 다. 보정은 그것으로 나누는 일이다.

① 쌓인 몫이 1 에 닿기까지 — beta2 = 0.999 는 수천 걸음이 걸린다.
② 보정을 빼먹으면 걸음 크기가 몇 배가 되는가. 정점이 1 이 아니라 여섯 배가 넘는다.
③ beta2 를 1 에 붙일수록 그 편향이 커진다 — 논문이 적어 둔 바를 수로.

만드는 파일: docs/ch06/estimation_methods/img/adam_moments.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
# 한글 글꼴에는 mathtext 의 U+2212 가 없다. 수식 글꼴을 따로 지정한다
plt.rcParams["mathtext.fontset"] = "dejavusans"

B1, B2, EPS = 0.9, 0.999, 1e-8      # 논문의 기본값
T = np.arange(1, 20001)

# === ① 쌓인 몫 ======================================================
acc1 = 1 - B1 ** T
acc2 = 1 - B2 ** T


def reach(beta, frac):
    """1 - beta^t 가 frac 에 닿는 가장 이른 t."""
    return int(np.ceil(np.log(1 - frac) / np.log(beta)))


print(f"① 기울기가 일정할 때 t 걸음째까지 쌓인 몫 (beta1 = {B1}, beta2 = {B2})")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("        t    1차 적률    2차 적률")
for t in (1, 2, 10, 100, 1000, 5000):
    print(f"   {t:>6}   {1 - B1 ** t:>9.4f}   {1 - B2 ** t:>9.4f}")
for f in (0.5, 0.9, 0.99):
    print(f"   {int(f*100)}% 에 닿는 걸음: 1차 {reach(B1, f):>5}, "
          f"2차 {reach(B2, f):>6}")
print(f"   **2 차 적률은 {reach(B2, 0.9):,} 걸음이 지나야 참값의 90% 에 닿는다.**")

# === ② 보정을 빼먹으면 ==============================================
ratio = acc1 / np.sqrt(acc2)        # (보정 없음) / (보정 있음)
imax = int(np.argmax(ratio))
print(f"\n② 보정 없는 걸음 / 보정 있는 걸음 = (1-b1^t) / sqrt(1-b2^t)")
print("        t      배수")
for t in (1, 5, 10, 20, 50, 100, 1000, 10000):
    print(f"   {t:>6}   {ratio[t-1]:>7.3f}")
print(f"   정점은 t = {T[imax]} 에서 {ratio[imax]:.3f} 배다.")
print(f"   첫 걸음에서도 {ratio[0]:.3f} 배다 — "
      f"(1-{B1})/sqrt(1-{B2}) = {(1-B1)/np.sqrt(1-B2):.3f}.")
print("   **보정을 빼면 학습 초반의 걸음이 여섯 배까지 커진다.**")

# === ③ beta2 가 1 에 가까울수록 편향이 커진다 ======================
# 논문: "beta2 가 1 에 가까우면 ... 초기화 편향이 더 커진다"
B2S = [0.9, 0.99, 0.999, 0.9999, 0.99999]
print("\n③ beta2 를 바꿔 가며 정점 배수 (beta1 = 0.9 고정)")
print("      beta2    정점 배수   그때의 t   90% 에 닿는 걸음")
PEAK, PEAK_T, REACH90 = [], [], []
for b2 in B2S:
    r = (1 - B1 ** T) / np.sqrt(1 - b2 ** T)
    i = int(np.argmax(r))
    PEAK.append(float(r[i]))
    PEAK_T.append(int(T[i]))
    REACH90.append(reach(b2, 0.9))
    print(f"   {b2:>9}   {PEAK[-1]:>9.2f}   {PEAK_T[-1]:>8}   {REACH90[-1]:>14,}")
print("   **beta2 를 1 에 붙일수록 보정을 빼먹은 값이 더 크게 틀린다** — "
      "논문이 적어 둔 그대로다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"

ax1.semilogx(T, acc1, lw=2.8, color=C1, label=f"$1-\\beta_1^t$  ($\\beta_1 = {B1}$)")
ax1.semilogx(T, acc2, lw=2.8, color=C2, label=f"$1-\\beta_2^t$  ($\\beta_2 = {B2}$)")
ax1.axhline(0.9, color="#333333", lw=1.6, ls="--")
for b, c in ((B1, C1), (B2, C2)):
    t90 = reach(b, 0.9)
    ax1.plot([t90], [0.9], "o", color=c, ms=12, zorder=5)
    ax1.annotate(f"$t = {t90:,}$".replace(",", "{,}"), (t90, 0.9),
                 xytext=(0.4, -1.8), textcoords="offset fontsize",
                 fontsize=10.5, color=c)
ax1.set_xlabel("걸음 $t$")
ax1.set_ylabel("참 적률 가운데 쌓인 몫")
ax1.set_xticks([1, 100, 10000])
ax1.set_xticklabels(["$1$", "$10^{2}$", "$10^{4}$"])
ax1.minorticks_off()
ax1.set_ylim(0, 1.08)
ax1.set_title("① 2 차 적률은 아주 느리게 찬다", fontsize=12)
ax1.legend(loc="lower right", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.semilogx(T, ratio, lw=2.8, color=C2)
ax2.axhline(1.0, color=C3, lw=2.2, ls="--")
ax2.text(1200, 1.25, "보정을 하면 여기", fontsize=10.5, color=C3)
ax2.plot([T[imax]], [ratio[imax]], "o", color="#333333", ms=13, zorder=5)
ax2.annotate(f"정점 ${ratio[imax]:.2f}$ 배\n($t = {T[imax]}$)",
             (T[imax], ratio[imax]), xytext=(0.6, -0.6),
             textcoords="offset fontsize", fontsize=10.5, color="#333333")
ax2.set_xlabel("걸음 $t$")
ax2.set_ylabel("보정 없는 걸음 / 보정 있는 걸음")
ax2.set_xticks([1, 100, 10000])
ax2.set_xticklabels(["$1$", "$10^{2}$", "$10^{4}$"])
ax2.minorticks_off()
ax2.set_ylim(0, ratio.max() * 1.15)
ax2.set_title("② 보정을 빼먹으면 걸음이 커진다", fontsize=12)
ax2.grid(alpha=0.25)

xs3 = np.arange(len(B2S))
ax3.plot(xs3, PEAK, "o-", color=C2, lw=2.8, ms=13)
ax3.axhline(1.0, color=C3, lw=2.2, ls="--")
ax3.text(0.12, 13.0, "보정을 하면 여기", fontsize=10.5, color=C3)
ax3.annotate("", xy=(0.05, 1.0), xytext=(0.3, 11.5),
             arrowprops=dict(arrowstyle="->", color=C3, lw=1.8))
i999 = B2S.index(0.999)
ax3.plot([xs3[i999]], [PEAK[i999]], "o", color="#333333", ms=15, zorder=5)
ax3.annotate(f"기본값 $\\beta_2 = 0.999$\n${PEAK[i999]:.2f}$ 배",
             (xs3[i999], PEAK[i999]), xytext=(0.7, 0.2),
             textcoords="offset fontsize", fontsize=10.5, color="#333333")
for i, v in enumerate(PEAK):
    if i > i999:          # 작은 값 둘은 눈금 글자와 겹쳐서 생략한다
        ax3.annotate(f"${v:.2f}$", (xs3[i], v), xytext=(0.0, 0.9),
                     textcoords="offset fontsize", fontsize=10,
                     color=C2, ha="center")
ax3.set_xticks(xs3)
ax3.set_xticklabels(["$0.9$", "$0.99$", "$0.999$", "$0.9999$", "$0.99999$"],
                    fontsize=10)
ax3.set_xlabel("$\\beta_2$")
ax3.set_ylabel("정점에서의 배수")
ax3.set_xlim(-0.4, len(B2S) - 0.3)
ax3.set_ylim(0, max(PEAK) * 1.18)
ax3.set_title("③ $\\beta_2$ 가 1 에 가까울수록 커진다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/estimation_methods/img/adam_moments.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
