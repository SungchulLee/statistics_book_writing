"""9.2절 모분산의 카이제곱 검정 — 로그를 취하면 수준이 제자리로 온다.

오른쪽으로 늘어진 자료에 이 검정을 원자료 그대로 걸면 명목 5% 가 지켜지지
않는다. 변동계수가 커질수록 더 나빠지는데, 로그를 취한 뒤에 걸면 어느
변동계수에서도 0.05 근처다. 로그정규의 로그가 바로 정규이기 때문이다.
수는 쪽의 모의 출력 그대로다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch09/one_sample_tests/img/"
NOMINAL = 0.05

# 자료 · 변동계수 · 원자료 그대로 · 로그를 취한 뒤 (None = 쪽의 「—」)
ROWS = [("정규",     0.15, 0.054, None),
        ("로그정규", 0.15, 0.068, 0.047),
        ("로그정규", 0.30, 0.124, 0.050),
        ("로그정규", 0.60, 0.342, 0.048)]

labels = [f"{n}\n변동계수 {cv:.2f}" for n, cv, _, _ in ROWS]
raw = np.array([r[2] for r in ROWS])
logged = [r[3] for r in ROWS]
x = np.arange(len(ROWS))
w = 0.34

fig, ax = plt.subplots(figsize=(10.4, 4.2))

b1 = ax.bar(x - w / 2, raw, w, color="tab:red", alpha=0.85,
            label="원자료 그대로")
b2 = ax.bar([xi + w / 2 for xi, v in zip(x, logged) if v is not None],
            [v for v in logged if v is not None], w,
            color="tab:blue", alpha=0.85, label="로그를 취한 뒤")

ax.axhline(NOMINAL, color="black", ls="--", lw=1.3)
# 0.054 막대 꼬리표와 겹치지 않도록 기준선 아래에 둔다
ax.text(-0.45, NOMINAL - 0.009, "명목 0.05", ha="left", va="top", fontsize=10)

for xi, v in zip(x, raw):
    lab = f"{v:.3f}" if v <= 0.15 else f"{v:.3f}"
    ax.text(xi - w / 2, v + 0.008, lab, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="tab:red")
    if 0.06 < v <= 0.15:
        ax.text(xi - w / 2, v + 0.030, f"명목의 {v / NOMINAL:.1f}배", ha="center",
                va="bottom", fontsize=9.5, color="0.45")
    if v > 0.15:
        ax.text(xi - w / 2, (v + NOMINAL) / 2, f"명목의\n{v / NOMINAL:.1f}배", ha="center",
                va="center", fontsize=9.5, color="white", fontweight="bold")

for xi, v in zip(x, logged):
    if v is None:
        ax.text(xi + w / 2, 0.006, "해당 없음\n(이미 정규다)", ha="center",
                va="bottom", fontsize=9, color="0.5")
        continue
    ax.text(xi + w / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="tab:blue")

ax.set_xticks(x, labels, fontsize=10.5)
ax.set_ylim(0, 0.40)
ax.set_ylabel("명목 5% 검정의 실제 기각률")
ax.set_title("오른쪽으로 늘어진 자료에 분산 검정을 걸면", fontsize=12)
ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "log_scale_rescues.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for n, cv, r, lg in ROWS:
    s = "—" if lg is None else f"{lg:.3f}"
    print(f"{n:>8} cv {cv:.2f}   원자료 {r:.3f} (명목의 {r / NOMINAL:.1f}배)"
          f"   로그 {s}")
