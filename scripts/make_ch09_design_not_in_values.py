"""9.4절 대응 대 이표본 — 짝인지 아닌지는 값에 적혀 있지 않다.

들머리의 요점 둘을 그린다. 왼쪽은 자료표 둘이다 --- 하나는 같은 사람을
전후로 잰 것이고 하나는 두 부서의 사람들인데, 수만 보아서는 어느 쪽이
대응인지 가릴 수 없다. 오른쪽은 틀렸을 때 치르는 값으로, 방향이 반대인
두 잘못이 둘 다 손해라는 것을 보인다.

쪽의 paired_vs_unpaired 는 한 자료를 두 가지로 분석한 결과를 보이고
twins_pairing 은 상관에 따른 필요 표본과 검정력을 그린다. 여기서는
「무엇을 보고 정하는가」와 「틀리면 어떻게 되는가」만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/paired_sample_tests/img/"
N, R, A = 12, 20_000, 0.05
rng = np.random.default_rng(1990)

# ── 보여 줄 자료표 둘 ─────────────────────────────────────────────
z = rng.standard_normal(N)
A1 = np.round(28 + 4 * z + rng.standard_normal(N) * 1.2, 1)       # 전
A2 = np.round(A1 - 2.4 + rng.standard_normal(N) * 1.5, 1)         # 후 (같은 사람)
B1 = np.round(rng.normal(31, 4.3, N), 1)                          # A 부서
B2 = np.round(rng.normal(28.6, 4.3, N), 1)                        # B 부서

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8),
                         gridspec_kw={"width_ratios": [1, 1, 1.5]})
for ax, (c1, c2, title) in zip(axes[:2],
                               [(A1, A2, "자료 ㄱ"), (B1, B2, "자료 ㄴ")]):
    ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=14, fontweight="bold", color="0.15")
    ax.text(0.28, 0.95, "첫째 열", transform=ax.transAxes, ha="center",
            fontsize=11.5, color="0.4")
    ax.text(0.72, 0.95, "둘째 열", transform=ax.transAxes, ha="center",
            fontsize=11.5, color="0.4")
    for i, (v1, v2) in enumerate(zip(c1, c2)):
        y = 0.89 - 0.072 * i
        if i % 2 == 0:
            ax.axhspan(y - 0.034, y + 0.036, xmin=0.08, xmax=0.92,
                       color="0.93")
        ax.text(0.28, y, f"{v1:.1f}", transform=ax.transAxes, ha="center",
                fontsize=12, color="0.15")
        ax.text(0.72, y, f"{v2:.1f}", transform=ax.transAxes, ha="center",
                fontsize=12, color="0.15")
    ax.axis("off")
axes[0].text(0.5, -0.02, "같은 사람을 전후로 잰 것", transform=axes[0].transAxes,
             ha="center", va="top", fontsize=11.5, color="tab:blue",
             fontweight="bold")
axes[1].text(0.5, -0.02, "두 부서의 서로 다른 사람들",
             transform=axes[1].transAxes, ha="center", va="top",
             fontsize=11.5, color="tab:red", fontweight="bold")
fig.text(0.34, 0.035, "값만 보아서는 가를 수 없다 — 가르는 것은 "
         "자료를 어떻게 모았는가다", ha="center", fontsize=12.5,
         fontweight="bold", color="0.2")

# ── (우) 틀리면 양쪽으로 값을 치른다 ──────────────────────────────
def power(paired_truth, analyze_paired, rho=0.7, delta=0.9):
    z0 = rng.standard_normal((R, N))
    z1 = rng.standard_normal((R, N))
    if paired_truth:
        x = z0
        y = delta + rho * z0 + np.sqrt(1 - rho ** 2) * z1
    else:
        x, y = z0, delta + z1
    if analyze_paired:
        d = y - x
        t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(N))
        p = 2 * stats.t.sf(np.abs(t), N - 1)
    else:
        sp = np.sqrt(((N - 1) * x.var(1, ddof=1) +
                      (N - 1) * y.var(1, ddof=1)) / (2 * N - 2))
        t = (y.mean(1) - x.mean(1)) / (sp * np.sqrt(2 / N))
        p = 2 * stats.t.sf(np.abs(t), 2 * N - 2)
    return float((p < A).mean())


CASES = [("짝인 자료를\n대응으로", True, True, "tab:blue"),
         ("짝인 자료를\n풀어서", True, False, "0.6"),
         ("짝 아닌 자료를\n이표본으로", False, False, "tab:red"),
         ("짝 아닌 자료를\n묶어서", False, True, "0.6")]
vals = [power(a, b) for _, a, b, _ in CASES]
ax = axes[2]
x = np.arange(4)
ax.bar(x, [100 * v for v in vals], width=0.6,
       color=[c for *_, c in CASES], alpha=0.88)
for xi, v in zip(x, vals):
    ax.text(xi, 100 * v + 1.6, f"{100*v:.1f}%", ha="center", fontsize=12,
            fontweight="bold", color="0.2")
for i, ybase in ((0, 72), (2, 36)):
    ax.annotate("", xy=(i + 0.74, ybase), xytext=(i + 0.26, ybase),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8))
    ax.text(i + 0.5, ybase + 2.0,
            f"{100*(vals[i]-vals[i+1]):+.1f}%p", ha="center", fontsize=11,
            color="0.2", fontweight="bold")
ax.set_xticks(x, [n for n, *_ in CASES], fontsize=10.5)
ax.set_ylim(0, 105)
ax.set_ylabel("참 차이를 잡아낼 확률 (%)", fontsize=11)
ax.set_title("틀리면 양쪽으로 값을 치른다", fontsize=12.5)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("짝인지 아닌지는 자료의 값 어디에도 적혀 있지 않다",
             fontsize=14, y=1.0)
fig.subplots_adjust(bottom=0.16)
path = OUT + "design_not_in_values.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for (n, a_, b_, _), v in zip(CASES, vals):
    print(f"  {n.replace(chr(10),' '):<22s} 검정력 {v:.4f}")
print(f"  짝인데 풀면 {100*(vals[0]-vals[1]):+.1f} %p · "
      f"짝 아닌데 묶으면 {100*(vals[2]-vals[3]):+.1f} %p")
