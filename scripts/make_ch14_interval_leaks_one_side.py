"""14.7절 회귀 잔차 — 전체 포함률은 멀쩡한데 한쪽으로만 샌다.

잔차가 오른쪽으로 늘어진 자료에서 정규를 믿은 95% 예측구간은 전체 포함률이
0.9582 로 약속에 가깝다. 그런데 아래로 샌 것이 0.0000, 위로 샌 것이 0.0418 로
「양쪽에 2.5% 씩」이라는 약속이 통째로 깨져 있다.

같은 자료의 잔차에서 분위수를 직접 읽어 만든 구간은 0.0250 과 0.0222 로
양쪽이 고르다. 폭은 오히려 좁다(8.93 대 9.75).

쪽의 코드(seed 14)를 그대로 다시 돌린다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch14/applications/img/"

rng = np.random.default_rng(14)
n_tr, n_cal, n_te = 1_000, 2_000, 20_000
x = rng.uniform(0, 10, n_tr + n_cal + n_te)
y = 2 + 3 * x + (np.exp(rng.normal(0, 1.0, x.size)) - np.exp(0.5)) * 1.2
tr, cal, te = slice(0, n_tr), slice(n_tr, n_tr + n_cal), slice(n_tr + n_cal, None)
b1, b0 = np.polyfit(x[tr], y[tr], 1)
r = y - (b0 + b1 * x)
s = r[tr].std(ddof=2)
lv = np.ceil((n_cal + 1) * np.array([0.025, 0.975])) / n_cal

BANDS = [("정규를 믿는 구간", (-1.959964 * s, 1.959964 * s), "tab:red"),
         ("잔차가 정하는 구간", tuple(np.quantile(r[cal], lv)), "tab:blue")]
res = []
for name, (lo, hi), c in BANDS:
    u, o = float(np.mean(r[te] < lo)), float(np.mean(r[te] > hi))
    res.append((name, lo, hi, 1 - u - o, u, o, c))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5),
                               gridspec_kw={"width_ratios": [1.4, 1]})

# ── (좌) 잔차의 분포 위에 두 구간 ───────────────────────────────────
ax1.hist(r[te], bins=np.linspace(-6, 14, 160), color="0.75", alpha=0.85)
for i, (name, lo, hi, cov, u, o, c) in enumerate(res):
    yy = ax1.get_ylim()[1] * (0.86 - 0.16 * i)
    ax1.hlines(yy, lo, hi, color=c, lw=7, alpha=0.8)
    ax1.plot([lo, lo], [0, yy], ":", color=c, lw=1.2, alpha=0.6)
    ax1.plot([hi, hi], [0, yy], ":", color=c, lw=1.2, alpha=0.6)
    ax1.text((lo + hi) / 2, yy, f"{name}   폭 {hi - lo:.2f}", ha="center",
             va="bottom", fontsize=10.5, color=c, fontweight="bold")
    ax1.text(lo, yy, f"{lo:+.2f} ", ha="right", va="center", fontsize=9.5, color=c)
    ax1.text(hi, yy, f" {hi:+.2f}", ha="left", va="center", fontsize=9.5, color=c)

ax1.axvline(0, color="0.4", lw=1)
ax1.set_xlim(-6, 14)
ax1.set_xlabel("잔차")
ax1.set_ylabel(f"시험 자료 {n_te:,} 개의 빈도")
ax1.set_title("잔차는 오른쪽으로 늘어져 있는데 한 구간만 대칭이다", fontsize=11.5)
ax1.grid(axis="y", alpha=0.2)

# ── (우) 어느 쪽으로 샜는가 ─────────────────────────────────────────
yb = np.arange(len(res))[::-1]
h = 0.3
for yy, (name, lo, hi, cov, u, o, c) in zip(yb, res):
    ax2.barh(yy + h / 2, u, h, color="tab:purple", alpha=0.85,
             label="아래로 샘" if yy == yb[0] else None)
    ax2.barh(yy - h / 2, o, h, color="tab:orange", alpha=0.85,
             label="위로 샘" if yy == yb[0] else None)
    ax2.text(u + 0.001, yy + h / 2, f"{u:.4f}" + ("  (한 번도 안 샜다)" if u == 0 else ""),
             va="center", ha="left", fontsize=10.5)
    ax2.text(o + 0.001, yy - h / 2, f"{o:.4f}", va="center", ha="left", fontsize=10.5)
    # 막대 위에 적는다 — 막대 안에 넣으면 주황과 겹친다
    ax2.text(0.0005, yy + h + 0.13, f"전체 포함률 {cov:.4f}", va="bottom", ha="left",
             fontsize=10.5, color=c, fontweight="bold")

ax2.axvline(0.025, color="black", ls="--", lw=1.6)
ax2.text(0.0255, len(res) - 0.42, "약속한 2.5%", ha="left", va="bottom", fontsize=10.5)
ax2.set_yticks(yb, [r_[0].replace(" ", "\n", 1) for r_ in res], fontsize=10.5)
ax2.set_xlim(0, 0.052)
ax2.set_ylim(-0.62, len(res) - 0.12)
ax2.set_xlabel("약속을 벗어난 비율")
ax2.set_title("전체는 맞는데 양쪽이 고르지 않다", fontsize=11.5)
ax2.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "interval_leaks_one_side.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for name, lo, hi, cov, u, o, _ in res:
    print(f"{name:18s} 폭 {hi-lo:5.2f}  [{lo:+6.2f}, {hi:+6.2f}]   "
          f"포함률 {cov:.4f}   아래 {u:.4f}   위 {o:.4f}")
print("쪽: 0.9582 / 0.0000 / 0.0418  ·  0.9527 / 0.0250 / 0.0222")
