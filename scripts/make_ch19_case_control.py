r"""19.1절 「그림으로 보는 통계학」 — 오즈비만 살아남는 까닭.

이 쪽은 오즈비가 확률비가 아니라고 바르게 경고한다. 그러면 왜 굳이
오즈비를 쓰는가. 더 읽기 쉬운 위험비를 두고.

답은 1951 년에 콘필드가 냈다. **오즈비는 결과를 보고 표본을 뽑아도
변하지 않는 유일한 양이다.**

드문 병을 연구한다고 하자. 무작위로 만 명을 뽑으면 환자가 몇 명 안
나온다. 그래서 역학에서는 환자-대조 설계를 쓴다 — 환자를 먼저 다 모으고
대조군을 따로 모으는 것이다. 이때 「위험」은 복구할 수 없다. 환자의
비율을 **연구자가 정했기** 때문이다.

그런데 오즈비는 그대로 남는다. 모집단을 만들어 확인하면

  대조군 표집률     오즈비    위험비    위험차
     1.0 (전수)     3.124     2.996    0.0400
     0.05           3.140     1.939    0.2718
     0.01           3.084     1.284    0.1910
     0.002          3.067     1.063    0.0572
     0.0005         3.305     1.017    0.0161

표집률을 2000 배 바꾸는 동안 오즈비는 3.1 에 머물고 위험비는 3.0 에서
1.02 로 무너진다.

로지스틱 회귀로 보면 더 깔끔하다. 기울기는 그대로이고 **절편만**
-log(표집률)만큼 밀린다.

    모집단        절편 -3.8888   기울기 +1.1390
    표집률 0.05   절편 -0.8936   기울기 +1.1449   (예측 -0.8931)
    표집률 0.002  절편 +2.3233   기울기 +1.1784   (예측 +2.3258)

로지스틱 회귀가 역학의 일꾼이 된 까닭이 이것이다. 그 계수가 로그
오즈비이고, 로그 오즈비는 환자-대조 표집을 견딘다.

만드는 파일:

  ch19/logistic_regression/img/case_control.png

실행:  python3 scripts/make_ch19_case_control.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/logistic_regression/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, PX, P0, P1, SEED = 4_000_000, 0.30, 0.02, 0.06, 1951
RATES = (1.0, 0.2, 0.05, 0.01, 0.004, 0.002, 0.001, 0.0005)


def summ(x, y):
    a = ((x == 1) & (y == 1)).sum(); b = ((x == 1) & (y == 0)).sum()
    c = ((x == 0) & (y == 1)).sum(); d = ((x == 0) & (y == 0)).sum()
    r1, r0 = a / (a + b), c / (c + d)
    return (a / b) / (c / d), r1 / r0, r1 - r0


def logit_fit(x, y):
    Xd = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(60):
        p = 1 / (1 + np.exp(-Xd @ b))
        W = p * (1 - p)
        step = np.linalg.solve((Xd * W[:, None]).T @ Xd, Xd.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-11:
            break
    return b


def main():
    rng = np.random.default_rng(SEED)
    X = (rng.random(N) < PX).astype(int)
    Y = (rng.random(N) < np.where(X == 1, P1, P0)).astype(int)
    o0, r0_, d0 = summ(X, Y)
    b0 = logit_fit(X, Y)
    print(f"    모집단  오즈비 {o0:.4f}  위험비 {r0_:.4f}  위험차 {d0:.5f}")
    print(f"            절편 {b0[0]:+.4f}  기울기 {b0[1]:+.4f}")

    cases = np.where(Y == 1)[0]
    ctrl = np.where(Y == 0)[0]
    ors, rrs, rds, b1s, b0s = [], [], [], [], []
    print(f"\n{'대조 표집률':>11}{'표본':>11}{'오즈비':>9}{'위험비':>9}"
          f"{'위험차':>10}{'절편':>10}{'기울기':>9}")
    for f in RATES:
        idx = np.concatenate([cases, ctrl[rng.random(ctrl.size) < f]])
        o, r, d = summ(X[idx], Y[idx])
        b = logit_fit(X[idx], Y[idx])
        ors.append(o); rrs.append(r); rds.append(d)
        b0s.append(b[0]); b1s.append(b[1])
        print(f"{f:>11.4f}{idx.size:>11,}{o:>9.4f}{r:>9.4f}{d:>10.5f}"
              f"{b[0]:>10.4f}{b[1]:>9.4f}")

    rr = np.array(RATES)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 세 측도가 표집률에 어떻게 반응하나 ──
    ax = axes[0]
    ax.axhline(1.0, color=INK, lw=2.0, ls="--")
    ax.text(0.9, 1.12, "모집단 값과 같다", fontsize=10.8, color=INK,
            ha="right")
    ax.plot(rr, np.array(ors) / o0, lw=3.2, color=TEAL, marker="o", ms=8,
            label="오즈비")
    ax.plot(rr, np.array(rrs) / r0_, lw=2.8, color=RED, marker="s", ms=7,
            label="위험비")
    ax.plot(rr, np.array(rds) / d0, lw=2.8, color=AMBER, marker="^", ms=7,
            label="위험차")
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1 (전수)"])
    ax.minorticks_off()
    ax.annotate(f"표집률을 $2000$ 배 바꾸어도\n오즈비는 "
                f"${min(ors):.2f}$–${max(ors):.2f}$ 에 머문다",
                xy=(0.004, np.array(ors)[RATES.index(0.004)] / o0),
                xytext=(0.03, 2.6), fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate("위험비는 $1$ 로 무너진다 — 환자의 비율을\n"
                "연구자가 정했으니 「위험」이 남아 있을 리 없다",
                xy=(0.0012, np.array(rrs)[RATES.index(0.001)] / r0_),
                xytext=(0.006, 0.22), fontsize=10.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("대조군을 뽑은 비율", fontsize=11.5)
    ax.set_ylabel("모집단 값으로 나눈 추정값", fontsize=11.5)
    ax.set_title("결과를 보고 표본을 뽑으면", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 7.2)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 로지스틱 계수 ──
    ax = axes[1]
    ax.axhline(b0[1], color=TEAL, lw=2.0, ls="--")
    ax.plot(rr, b1s, lw=3.2, color=TEAL, marker="o", ms=8,
            label=f"기울기 — 모집단에서 ${b0[1]:+.3f}$")
    ax.plot(rr, b0s, lw=3.4, color=RED, marker="s", ms=8,
            label="절편 — 표집률만큼 밀린다")
    fine = np.logspace(np.log10(4e-4), 0, 200)
    ax.plot(fine, b0[0] - np.log(fine), lw=2.2, color=INK, ls=":", zorder=6)
    ax.annotate("기울기는 꿈쩍도 하지 않는다 —\n이것이 로그 오즈비다",
                xy=(0.02, b1s[RATES.index(0.02)] if 0.02 in RATES else b0[1]),
                xytext=(0.05, 3.3), fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate("검은 점선이 $\\beta_0 - \\log f$ —\n절편의 밀림이 식과 맞는다",
                xy=(0.0012, b0[0] - np.log(0.0012)), xytext=(0.03, -2.2),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1 (전수)"])
    ax.minorticks_off()
    ax.set_xlabel("대조군을 뽑은 비율", fontsize=11.5)
    ax.set_ylabel("로지스틱 회귀 계수", fontsize=11.5)
    ax.set_title("밀리는 것은 절편뿐이다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(-5.2, 4.6)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("결과를 보고 표본을 뽑아도 살아남는 것", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"사람 {N//10**6} 백만 명을 만들었다. 노출률 ${PX}$, 노출군의 "
             f"위험 ${P1}$, 비노출군 ${P0}$ 이다. 환자는 모두 쓰고 대조군만 "
             "가로축의 비율로 뽑았다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "case_control.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
