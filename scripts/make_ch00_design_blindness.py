r"""기울기를 가장 잘 재는 설계가 가장 눈먼 설계다 — x 를 어디에 둘 것인가.

이 쪽의 절 5 는 x 를 넓게 벌릴수록 Var(beta1) = sigma^2/Sxx 가 작아짐을 보인다.
그 말을 끝까지 밀면 답이 하나로 정해진다 — **양 끝에만 몰아라.** 같은 범위에서
Sxx 를 최대로 만드는 배치다.

그런데 x 가 두 값뿐이면 직선 말고는 아무것도 맞춰 볼 수 없다. 참 관계가 휘어 있어도
**휘었다는 것을 알 길이 없다.**

① 같은 n, 같은 범위에서 배치만 바꿀 때의 기울기 표준오차.
② 참 관계가 이차일 때 휘어 있음을 알아채는 힘.
③ 양 끝에 두는 몫을 바꿔 가며 — 둘을 함께 보면.

만드는 파일: docs/ch00/linalg_regression/linalg_statistics/img/design_blindness.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
N, SIGMA, A, B = 20, 1.0, 0.0, 10.0
REP = 20_000
BETA0, BETA1 = 1.0, 2.0


def designs():
    mid = (A + B) / 2
    return {
        "양 끝에만": np.r_[np.full(N // 2, A), np.full(N // 2, B)],
        "고르게": np.linspace(A, B, N),
        "가운데로": np.linspace(mid - 1, mid + 1, N),
    }


def slope_se(x):
    """이론값 sigma / sqrt(Sxx)."""
    return SIGMA / np.sqrt(((x - x.mean()) ** 2).sum())


print(f"n = {N}, sigma = {SIGMA}, 범위 [{A:.0f}, {B:.0f}], 참 직선 "
      f"y = {BETA0} + {BETA1}x")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n① 배치만 바꿀 때 (범위와 n 은 같다)")
print("   배치         서로 다른 x   Sxx      기울기 SE   양끝 대비")
D = designs()
SE = {}
for name, x in D.items():
    sxx = ((x - x.mean()) ** 2).sum()
    SE[name] = slope_se(x)
    print(f"   {name:<12} {len(np.unique(x)):>9}   {sxx:>7.1f}   {SE[name]:>9.4f}   "
          f"{SE[name]/SE['양 끝에만']:>8.2f}x")
print(f"   양 끝에만 두는 것이 기울기를 가장 정밀하게 잰다 — "
      f"고르게 두는 것보다 {SE['고르게']/SE['양 끝에만']:.2f} 배 좁다.")

# === ② 참 관계가 휘어 있을 때 =======================================
CURV = [0.0, 0.02, 0.05, 0.10, 0.20]
print(f"\n② 참 관계가 y = {BETA0} + {BETA1}x + c(x-5)^2 일 때 "
      f"「휘었다」를 알아채는 힘")
print("   곡률 c      양 끝에만   고르게   가운데로")
POW = {k: [] for k in D}
for c in CURV:
    row = []
    for name, x in D.items():
        if len(np.unique(x)) < 3:
            POW[name].append(np.nan)     # 이차항을 맞출 수조차 없다
            row.append(None)
            continue
        Xq = np.c_[np.ones(N), x, (x - 5) ** 2]
        XtXi = np.linalg.pinv(Xq.T @ Xq)
        hits = 0
        for _ in range(2000):
            y = BETA0 + BETA1 * x + c * (x - 5) ** 2 + RNG.normal(0, SIGMA, N)
            b = XtXi @ (Xq.T @ y)
            e = y - Xq @ b
            s2 = (e @ e) / (N - 3)
            se = np.sqrt(s2 * XtXi[2, 2])
            t = b[2] / se
            hits += abs(t) > stats.t.ppf(0.975, N - 3)
        POW[name].append(hits / 2000)
        row.append(hits / 2000)
    cells = "   ".join("    없음" if v is None else f"{v:>8.3f}" for v in row)
    print(f"   {c:>6.2f}   {cells}")
print("   **양 끝에만 두면 이차항을 적합할 수조차 없다** — 서로 다른 x 가 둘뿐이라")
print("   계획행렬의 계수가 모자란다. 휘어 있어도 영영 알 수 없다.")

# === ③ 양 끝에 두는 몫을 바꿔 가며 ==================================
FRACS = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
C_TRUE = 0.10
print(f"\n③ 점의 일부만 양 끝에 두고 나머지를 가운데에 (곡률 {C_TRUE})")
print(f"   (끝의 두 극단은 뺐다 — 모두 가운데면 Sxx = 0 이고 모두 끝이면 곡률을 못 본다)")
print("   양끝 몫   기울기 SE   곡률 탐지력")
SE3, PW3 = [], []
for f in FRACS:
    k = int(round(N * f / 2)) * 2
    rest = N - k
    x = np.r_[np.full(k // 2, A), np.full(k // 2, B),
              np.full(rest, (A + B) / 2)] if rest else \
        np.r_[np.full(k // 2, A), np.full(k // 2, B)]
    SE3.append(slope_se(x))
    if len(np.unique(x)) < 3:
        PW3.append(np.nan)
    else:
        Xq = np.c_[np.ones(N), x, (x - 5) ** 2]
        XtXi = np.linalg.pinv(Xq.T @ Xq)
        hits = 0
        for _ in range(2000):
            y = BETA0 + BETA1 * x + C_TRUE * (x - 5) ** 2 + RNG.normal(0, SIGMA, N)
            b = XtXi @ (Xq.T @ y)
            e = y - Xq @ b
            s2 = (e @ e) / (N - 3)
            t = b[2] / np.sqrt(s2 * XtXi[2, 2])
            hits += abs(t) > stats.t.ppf(0.975, N - 3)
        PW3.append(hits / 2000)
    p = "없음" if np.isnan(PW3[-1]) else f"{PW3[-1]:.3f}"
    print(f"   {f:>7.1f}   {SE3[-1]:>9.4f}   {p:>11}")
print(f"   양 끝에만 둘 때의 SE 가 {SE['양 끝에만']:.4f} 인데, 열에 아홉을 끝에 두면")
print(f"   {SE3[-1]:.4f} 로 거의 같으면서 곡률 탐지력이 {PW3[-1]:.3f} 다.")
print("   **조금만 가운데에 남겨 두면 모형을 확인할 눈을 거의 공짜로 얻는다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {"양 끝에만": "#d62728", "고르게": "#2ca02c", "가운데로": "#1f77b4"}

for i, (name, x) in enumerate(D.items()):
    ax1.plot(x, np.full_like(x, i), "o", color=COL[name], ms=11, alpha=0.75)
    ax1.text(B + 0.4, i, f"SE $= {SE[name]:.3f}$", va="center", fontsize=11,
             color=COL[name])
ax1.set_yticks(range(len(D)))
ax1.set_yticklabels(list(D), fontsize=11)
ax1.set_xlabel("$x$ 를 놓은 자리")
ax1.set_xlim(A - 0.8, B + 4.2)
ax1.set_ylim(-0.6, len(D) - 0.4)
ax1.invert_yaxis()
ax1.set_title(f"① 같은 $n = {N}$, 같은 범위", fontsize=12)
ax1.grid(alpha=0.25, axis="x")

for name in D:
    v = POW[name]
    if np.all(np.isnan(v)):
        ax2.plot([], [], "o-", color=COL[name], lw=2.6, ms=9,
                 label=f"{name} (적합 불가)")
    else:
        ax2.plot(CURV, v, "o-", color=COL[name], lw=2.6, ms=9, label=name)
ax2.axhline(0.05, color="#333333", lw=1.6, ls=":")
ax2.text(0.005, 0.60, "양 끝에만 두면\n곡선이 아예 없다 —\n이차항을 맞출 수 없다",
         fontsize=10.5, color="#d62728")
ax2.set_xlabel("참 곡률 $c$")
ax2.set_ylabel("「휘었다」를 알아챌 확률")
ax2.set_ylim(0, 1.03)
ax2.set_title("② 모형이 틀렸음을 알아채는 힘", fontsize=12)
ax2.legend(loc="center right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.plot(FRACS, SE3, "o-", color="#d62728", lw=2.8, ms=10, label="기울기 SE")
ax3.set_xlabel("양 끝에 둔 점의 비율")
ax3.set_ylabel("기울기 SE", color="#d62728")
ax3.tick_params(axis="y", labelcolor="#d62728")
ax3b = ax3.twinx()
ax3b.plot(FRACS, PW3, "s--", color="#1f77b4", lw=2.8, ms=10,
          label="곡률 탐지력")
ax3b.set_ylabel("곡률 탐지력", color="#1f77b4")
ax3b.tick_params(axis="y", labelcolor="#1f77b4")
ax3b.set_ylim(0, 1.05)
ax3.set_title(f"③ 조금만 가운데에 두면 (곡률 ${C_TRUE}$)", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/linalg_statistics/img/design_blindness.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
