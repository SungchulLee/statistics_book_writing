r"""19.2절 「그림으로 보는 통계학」 — 한 해에 둘을 내놓고.

1972 년에 존 넬더와 로버트 웨더번이 『JRSS-A』에 「일반화선형모형」을
냈다. 그때까지 따로따로 배우고 따로따로 계산하던 것들 — 선형회귀,
로지스틱 회귀, 프로빗, 포아송 회귀, 분할표의 로그선형모형 — 이 사실은
세 조각으로 이루어진 하나의 틀이라는 것이었다.

  선형예측자  eta = X beta
  연결함수    g(mu) = eta
  분산함수    Var(Y) = phi V(mu)

바꾸는 것은 g 와 V 뿐이고, 맞추는 절차는 전부 같다 — 반복 가중최소제곱
(IRLS). 이 쪽의 이탈도도 그 논문에서 나온 말이며, 정규분포에서는
잔차제곱합과 같아진다.

여기서는 IRLS 를 한 번만 적어 놓고 세 가족에 그대로 돌린다.

  가족        연결      분산함수     반복    statsmodels 와의 최대 차이
  정규        항등      1            2 회    1.2e-15
  이항        로짓      mu(1-mu)     6 회    8.9e-16
  포아송      로그      mu          12 회    8.4e-14

같은 코드다. 바꾼 것은 함수 두 개뿐이다.

로버트 웨더번은 1974 년에 준가능도까지 내놓고 1975 년에 스물여덟
살로 세상을 떠났다. 야외에서 벌레에 쏘인 뒤의 과민성 쇼크였다.

만드는 파일:

  ch19/estimation_inference/img/glm_unification.png

실행:  python3 scripts/make_ch19_glm_unification.py   (저장소 최상위에서)
필요:  numpy, statsmodels, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/estimation_inference/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, SEED, MAXIT, TOL = 600, 1972, 25, 1e-12

#      이름      연결 g       역연결        분산함수 V      색
FAM = (("정규", "항등 $\\mu$", lambda e: e, lambda m: np.ones_like(m), TEAL),
       ("이항", "로짓 $\\log\\frac{\\mu}{1-\\mu}$",
        lambda e: 1 / (1 + np.exp(-e)), lambda m: m * (1 - m), RED),
       ("포아송", "로그 $\\log \\mu$", np.exp, lambda m: m, AMBER))


def sci(v):
    """2.7e-14 를 $2.7 \\times 10^{-14}$ 꼴로 적는다."""
    m, e = f"{v:.1e}".split("e")
    return f"{m} \\times 10^{{{int(e)}}}"


def irls(X, y, linkinv, var, maxit=MAXIT, tol=TOL):
    """어느 가족이든 이 고리 하나로 맞춘다. 바꾸는 것은 linkinv 와 var 뿐이다."""
    beta = np.zeros(X.shape[1])
    path = []
    for _ in range(maxit):
        eta = X @ beta
        mu = linkinv(eta)
        w = np.maximum(var(mu), 1e-12)        # 정준연결이면 dmu/deta = V(mu)
        z = eta + (y - mu) / w
        new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
        path.append(float(np.abs(new - beta).max()))
        beta = new
        if path[-1] < tol:
            break
    return beta, path


def main():
    rng = np.random.default_rng(SEED)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    X = np.c_[np.ones(N), x1, x2]
    truth = np.array([0.3, 0.9, -0.6])
    eta = X @ truth

    data = {"정규": eta + rng.normal(0, 1, N),
            "이항": (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(float),
            "포아송": rng.poisson(np.exp(eta)).astype(float)}
    smfam = {"정규": sm.families.Gaussian(), "이항": sm.families.Binomial(),
             "포아송": sm.families.Poisson()}

    print(f"{'가족':>8}{'반복':>6}{'statsmodels 와의 최대 차이':>26}")
    paths, gaps = {}, {}
    for nm, _, linkinv, var, _ in FAM:
        b, path = irls(X, data[nm], linkinv, var)
        ref = sm.GLM(data[nm], X, family=smfam[nm]).fit().params
        paths[nm] = path
        gaps[nm] = float(np.abs(b - ref).max())
        print(f"{nm:>8}{len(path):>6}{gaps[nm]:>26.2e}")
        print(f"{'':>8}  계수 {np.array2string(b, precision=4)}")

    # 이탈도가 정규에서 잔차제곱합과 같은지 확인
    g = sm.GLM(data["정규"], X, family=sm.families.Gaussian()).fit()
    rss = float(((data["정규"] - g.fittedvalues) ** 2).sum())
    print(f"\n  정규에서 이탈도 {g.deviance:.6f} = 잔차제곱합 {rss:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 같은 고리, 다른 함수 두 개 ──
    ax = axes[0]
    es = np.linspace(-4, 4, 400)
    for nm, lab, linkinv, var, col in FAM:
        m = linkinv(es)
        ax.plot(es, m, lw=3.4, color=col, label=f"{nm} — {lab}")
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.axvline(0, color=MUTED, lw=1.2)
    ax.annotate("같은 $X\\beta$ 를 받아\n저마다 다른 평균으로 보낸다",
                xy=(1.78, np.exp(1.78)), xytext=(2.15, -2.75), fontsize=10.8,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("선형예측자 $\\eta = X\\beta$", fontsize=11.5)
    ax.set_ylabel("평균 $\\mu$", fontsize=11.5)
    ax.set_title("바뀌는 것은 연결함수와 분산함수뿐", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(-4, 4); ax.set_ylim(-4.2, 7.2)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 같은 고리로 다 맞는다 ──
    ax = axes[1]
    for nm, _, _, _, col in FAM:
        p = np.maximum(np.array(paths[nm]), 1e-16)
        ax.plot(np.arange(1, len(p) + 1), p, lw=3.2, color=col, marker="o",
                ms=7,
                label=f"{nm} — {len(p)} 회, 오차 ${sci(gaps[nm])}$")
    ax.axhline(TOL, color=INK, lw=2.0, ls="--")
    ax.text(9.6, TOL * 1.9, "멈추는 기준", fontsize=10.6, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("반복 횟수", fontsize=11.5)
    ax.set_ylabel("계수가 움직인 폭", fontsize=11.5)
    ax.set_title("같은 고리를 돌려 셋 다 맞춘다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0.6, 13.4)
    ax.set_xticks(range(1, 14, 2))
    ax.set_ylim(1e-14, 20)
    ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
    ax.set_yticklabels(["$10^{-12}$", "$10^{-9}$", "$10^{-6}$", "$10^{-3}$",
                        "$1$"])
    ax.minorticks_off()
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("한 해에 둘을 내놓고", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"관측 ${N}$ 개, 설명변수 둘, 참 계수 $(0.3,\\ 0.9,\\ -0.6)$. "
             "세 가족 모두 같은 반복 가중최소제곱 고리로 맞췄고, "
             "「오차」는 statsmodels 가 내놓은 답과의 최대 차이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "glm_unification.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
