"""11.1절 부분상관 — 무엇을 통제했는지 적지 않으면 읽을 수 없다.

같은 (X, Y) 한 쌍에 통제집합만 바꾸어 본다. Z 는 둘을 함께 미는
교란변수이고 C 는 둘이 함께 미는 충돌자다. 통제 없이 재면 양수, Z 를
통제하면 0 근처, C 를 통제하면 오히려 음수가 된다. 셈은 네 번 다 맞다.

"측정한 변수는 모두 통제하는 것이 안전하다" 가 왜 틀렸는지도 여기 있다 ---
맨 오른쪽 막대가 그 규칙을 따른 결과다.

쪽의 partial_map 은 자료마다 부분상관이 어떻게 달라지는지를,
simpson_and_partial 은 합치기와 통제를 견준다. 여기서는 **자료를 고정하고
통제집합만** 바꾼다. 자료는 이 책이 만든 것이다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

OUT = "docs/ch11/correlation/img/"
N = 5000
rng = np.random.default_rng(10)
Z = rng.standard_normal(N)                       # 교란변수
X = 0.8 * Z + rng.standard_normal(N)
Y = 0.8 * Z + rng.standard_normal(N)             # X 와 Y 사이에 직접 연결이 없다
C = X + Y + rng.standard_normal(N)               # 충돌자


def resid(a, M):
    if M.size == 0:
        return a - a.mean()
    A = np.column_stack([np.ones(N)] + [M[:, j] for j in range(M.shape[1])])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


def pcorr(ctrl):
    M = np.column_stack(ctrl) if ctrl else np.empty((N, 0))
    return float(np.corrcoef(resid(X, M), resid(Y, M))[0, 1])


CASES = [("통제 없음", [], "0.6"),
         ("$Z$ 를 통제\n(교란변수)", [Z], "tab:blue"),
         ("$C$ 를 통제\n(충돌자)", [C], "tab:red"),
         ("$Z$ 와 $C$ 를 모두\n(「다 넣으면 안전」)", [Z, C], "tab:red")]
vals = [pcorr(c) for _, c, _ in CASES]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1, 1.35]})

# ── (좌) 자료를 만든 구조 ─────────────────────────────────────────
pos = {"Z": (0.5, 0.86), "X": (0.14, 0.50), "Y": (0.86, 0.50),
       "C": (0.5, 0.14)}
for a, b in (("Z", "X"), ("Z", "Y"), ("X", "C"), ("Y", "C")):
    x0, y0 = pos[a]; x1, y1 = pos[b]
    ax1.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color="0.45", lw=2.2,
                                 shrinkA=20, shrinkB=20))
for k, (xx, yy) in pos.items():
    col = {"Z": "tab:blue", "C": "tab:red"}.get(k, "0.25")
    ax1.plot([xx], [yy], "o", ms=40, color=col, alpha=0.85)
    ax1.text(xx, yy, k, ha="center", va="center", fontsize=17,
             fontweight="bold", color="white")
ax1.text(0.5, 0.50, "직접 연결 없음", ha="center", va="center",
         fontsize=12, color="0.45")
ax1.text(0.5, 0.97, "$Z$ 가 둘을 함께 민다", ha="center", fontsize=11,
         color="tab:blue")
ax1.text(0.5, -0.04, "$C$ 는 둘이 함께 민다", ha="center", fontsize=11,
         color="tab:red")
ax1.set_xlim(0, 1); ax1.set_ylim(-0.12, 1.06)
ax1.axis("off")
ax1.set_title("자료를 만든 구조", fontsize=12.5)

x = np.arange(len(CASES))
ax2.bar(x, vals, width=0.58, color=[c for *_, c in CASES], alpha=0.88)
for xi, v in zip(x, vals):
    ax2.text(xi, v + (0.025 if v >= 0 else -0.035), f"{v:+.4f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=13,
             fontweight="bold", color="0.15")
ax2.axhline(0, color="0.2", lw=1.6)
ax2.set_xticks(x, [n for n, *_ in CASES], fontsize=10.5)
ax2.set_ylim(-0.68, 0.52)
ax2.set_ylabel("$X$ 와 $Y$ 의 (부분)상관", fontsize=11)
ax2.set_title("자료는 하나다 — 통제집합만 바꾸었다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("같은 쌍에서 네 수가 나오고 셈은 네 번 다 맞다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "which_control.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for (n, _, _), v in zip(CASES, vals):
    print(f"  {n.replace(chr(10), ' '):<30s} {v:+.6f}")
