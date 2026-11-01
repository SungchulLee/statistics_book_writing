"""10장 개요 — 맞춰 보는 일은 우리가 잰 것까지만 듣는다.

들머리의 한가운데 문장을 그린다. 「연구자들이 게을렀던 것도 아니다.
나이도 흡연도 소득도 꼼꼼히 맞춰 보았다. 다만 맞춰 보는 일은 우리가
잰 것까지만 듣는다. 「건강에 신경 쓰는 마음」은 아무도 재지 못했고,
재지 못한 것은 맞출 수도 없다.」

처치가 다섯 가지 특성에 끌려 정해지는 자료를 만들었다. 넷은 재어
두었고 하나 --- 건강 의식 --- 는 끝내 재지 못한다. 맞추는 변수를 하나씩
늘려 가면 추정값이 참값 쪽으로 다가가다가 **멈춘다.** 넷을 다 맞춘 뒤에도
남는 거리가 재지 못한 하나의 몫이고, 그 거리는 아무리 꼼꼼해도 줄지
않는다.

오른쪽은 같은 자료생성과정에서 배정만 동전으로 바꾼 것이다. 아무것도
맞추지 않은 첫 칸이 이미 참값이고, 그 뒤로 무엇을 넣든 참값에 머문다.
동전이 재지 못한 변수까지 함께 섞어 주기 때문이다.

자료생성과정은 10.5절 실험과 인과 쪽의 것과 같은 꼴이다 --- 공변량 다섯은
표준정규, 결과는 각 공변량에 2.0 씩, 참 처치효과는 5.0.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch10/img/"

N = 400
REPS = 3000
TAU = 5.0
BETA_Y = 2.0
NAMES = ["나이", "흡연", "소득", "운동량", "건강 의식"]
HIDDEN = 4                       # 이 하나는 끝내 재지 못한다
rng = np.random.default_rng(1010)


def trial(design, gen):
    """다섯 특성이 처치를 끌어당기는 자료 한 벌."""
    Z = gen.normal(0, 1, (N, 5))
    if design == "self":
        p = 1 / (1 + np.exp(-1.5 * Z.sum(axis=1) / np.sqrt(5)))
        x = (gen.random(N) < p).astype(int)
    else:
        x = (gen.random(N) < 0.5).astype(int)
    y = 50 + Z @ np.full(5, BETA_Y) + TAU * x + gen.normal(0, 5, N)
    return Z, x, y


def adjusted(Z, x, y, k):
    """앞의 k 개 공변량만 넣고 회귀했을 때의 처치 계수."""
    cols = [np.ones(N), x.astype(float)] + [Z[:, j] for j in range(k)]
    M = np.column_stack(cols)
    return np.linalg.lstsq(M, y, rcond=None)[0][1]


out = {}
for design in ("self", "rand"):
    acc = np.empty((REPS, 5))
    for i in range(REPS):
        Z, x, y = trial(design, rng)
        for k in range(5):
            acc[i, k] = adjusted(Z, x, y, k)
    out[design] = (acc.mean(0), acc.std(0, ddof=1))

STEPS = ["아무것도\n맞추지 않음", "+ 나이", "+ 흡연", "+ 소득", "+ 운동량"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), sharey=True)

for ax, design, title, col in (
        (ax1, "self", "사람들이 스스로 정했다", "tab:red"),
        (ax2, "rand", "동전이 정했다", "tab:blue")):
    m, s = out[design]
    xs = np.arange(5)
    ax.errorbar(xs, m, yerr=s, fmt="o-", ms=10, lw=2.6, capsize=6,
                color=col)
    ax.axhline(TAU, color="0.15", lw=2.2, ls="--")
    for xx, v, sd in zip(xs, m, s):
        ax.text(xx, v + sd + 0.14, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, color=col, fontweight="bold")
    ax.set_xticks(xs, STEPS, fontsize=10.5)
    ax.set_xlim(-0.5, 4.5)
    ax.set_title(title, fontsize=13, color=col)
    ax.grid(axis="y", alpha=0.25)

m_self = out["self"][0]
ax1.text(-0.45, TAU, f"참 효과 {TAU:.0f}", ha="left", va="center",
         fontsize=11.5, color="0.15", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95,
                   edgecolor="none"))
ax1.annotate(f"넷을 다 맞춘 뒤에도\n{m_self[4] - TAU:+.2f} 가 남는다\n"
             f"— 재지 못한 「{NAMES[HIDDEN]}」의 몫이다",
             xy=(4, m_self[4]), xytext=(0.55, 4.85), ha="left", va="top",
             fontsize=11, color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax1.set_ylabel("처치효과 추정값", fontsize=11.5)
ax1.set_ylim(3.6, 10.6)
ax2.text(0.5, 0.95, "아무것도 맞추지 않은 첫 칸이 이미 참값이고\n"
         "그 뒤로 무엇을 넣든 참값에 머문다", transform=ax2.transAxes,
         ha="center", va="top", fontsize=11.5, fontweight="bold",
         color="tab:blue",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="tab:blue", lw=1.0))

fig.suptitle("맞춰 보는 일은 우리가 잰 것까지만 듣는다 — "
             f"같은 자료를 {REPS:,} 번 되풀이한 평균", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "adjusting_runs_out.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N} · {REPS:,} 번 · 참 처치효과 {TAU} · "
      f"재지 못한 변수 「{NAMES[HIDDEN]}」")
print("  맞춘 변수            스스로 정함        동전")
for k in range(5):
    lab = "없음" if k == 0 else " + ".join(NAMES[:k])
    print(f"  {lab:<22s} {out['self'][0][k]:6.3f}        "
          f"{out['rand'][0][k]:6.3f}")
print(f"\n  넷을 다 맞춘 뒤에도 남는 거리: "
      f"{out['self'][0][4] - TAU:+.3f}")
print(f"  동전 쪽은 아무것도 맞추지 않아도 "
      f"{out['rand'][0][0] - TAU:+.3f}")
