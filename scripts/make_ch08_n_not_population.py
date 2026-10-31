"""8.5절 표본크기 계산 — ±3.1%p 라는 이름을 정하는 것은 n 이다.

들머리의 문장 하나를 그린다. 「±3.1%p 는 1,000 명의 다른 이름이다」가
참이려면, 그 수가 **모집단이 얼마나 큰지와 무관**해야 한다. 실제로
그렇다 --- 유한모집단 보정 sqrt((N-n)/(N-1)) 은 n 이 N 에 견주어 작은
동안 1 에 붙어 있다.

왼쪽은 n = 1,000 을 못 박고 모집단 크기만 바꾼 것이고, 오른쪽은 같은
±3.1%p 를 약속하는 데 드는 사람 수를 모집단별로 센 것이다. 쪽의
inverse_square_law 가 n 과 오차한계를 맞바꾸는 두 곡선을 이미 그리고
poll_margin 이 1/sqrt(n) 감소를 그리므로, 여기서는 그 둘이 건드리지
않는 축 --- 모집단 크기 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/sample_size/img/"

N_SAMPLE = 1000
P = 0.5
Z = stats.norm.ppf(0.975)
TARGET = 0.031          # 발표문이 적는 오차한계


def moe(N, n=N_SAMPLE):
    """유한모집단 보정을 넣은 오차한계 (퍼센트포인트)."""
    fpc = np.sqrt((N - n) / (N - 1))
    return 100 * Z * np.sqrt(P * (1 - P) / n) * fpc


def need(N):
    """그 모집단에서 오차한계 TARGET 을 약속하는 데 드는 사람 수."""
    n0 = Z * Z * P * (1 - P) / TARGET ** 2
    return n0 / (1 + (n0 - 1) / N)


MARK = [(2_000, "회사 하나"), (20_000, "작은 군"), (800_000, "큰 도시"),
        (44_000_000, "대한민국 성인")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7))

# ── (좌) 모집단이 백 배 커져도 달라지지 않는다 ──────────────────────
g = np.logspace(np.log10(N_SAMPLE * 1.02), np.log10(1e8), 600)
ax1.plot(g, moe(g), color="tab:blue", lw=2.4)
ax1.axhline(moe(1e9), color="tab:red", ls="--", lw=1.6)
ax1.text(7e7, moe(1e9) + 0.045, "모집단이 아주 클 때의 값 3.10", ha="right",
         fontsize=10.5, color="tab:red", fontweight="bold")
for N, name in MARK:
    ax1.plot([N], [moe(N)], "o", ms=9, color="0.25")
    ax1.annotate(f"{name}\n{moe(N):.2f}", xy=(N, moe(N)),
                 xytext=(N, moe(N) - 0.42), ha="center", fontsize=10,
                 color="0.25", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
ax1.set_xscale("log")
ax1.set_xlim(N_SAMPLE * 1.02, 1e8)
ax1.set_ylim(0, 3.45)
ax1.set_xlabel("모집단 크기 N (로그 눈금)", fontsize=11)
ax1.set_ylabel("오차한계 (퍼센트포인트)", fontsize=11)
ax1.set_title(f"n = {N_SAMPLE:,} 을 못 박고 모집단만 바꾸면", fontsize=12.5)
ax1.grid(alpha=0.25, which="both")

# ── (우) 같은 약속에 드는 사람 수 ──────────────────────────────────
names = [m[1] for m in MARK] + ["무한히 큰 모집단"]
vals = [need(m[0]) for m in MARK] + [Z * Z * P * (1 - P) / TARGET ** 2]
y = np.arange(len(names))[::-1]
ax2.barh(y, vals, height=0.52, color="tab:purple", alpha=0.85)
for yy, v in zip(y, vals):
    ax2.text(v + 14, yy, f"{v:,.0f} 명", va="center", fontsize=11.5,
             fontweight="bold", color="tab:purple")
ax2.set_yticks(y, names, fontsize=11)
ax2.set_xlim(0, 1280)
ax2.set_xlabel("오차한계 ±3.1 퍼센트포인트를 약속하는 데 드는 사람 수",
               fontsize=11)
ax2.set_title("모집단이 만 배 달라도 거의 같은 수가 든다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("「±3.1 퍼센트포인트」라는 이름을 정하는 것은 사람 수이지 "
             "모집단이 아니다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "n_not_population.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N_SAMPLE:,} 을 못 박았을 때의 오차한계")
for N, name in MARK:
    print(f"  {name:<12s} N = {N:>12,}  →  ±{moe(N):.4f} 퍼센트포인트")
print(f"  {'아주 큰 모집단':<12s}                →  ±{moe(1e9):.4f}")
print(f"\n오차한계 ±{TARGET*100:.1f} 퍼센트포인트를 약속하는 데 드는 사람 수")
for N, name in MARK:
    print(f"  {name:<12s} N = {N:>12,}  →  {need(N):.1f} 명")
print(f"  {'무한히 큰 모집단':<12s}              →  "
      f"{Z*Z*P*(1-P)/TARGET**2:.1f} 명")
