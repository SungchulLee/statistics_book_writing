r"""18.3절 「그림으로 보는 통계학」 — 상관된 둘 가운데 어느 쪽이 목록에 오르는가.

라쏘는 상관이 큰 변수 무리에서 **하나만** 집는 버릇이 있다. 어느 하나를 집는지는
자료가 조금만 흔들려도 바뀐다. 그 흔들림을 세어 본다.

자료는 최소각회귀 논문이 쓴 당뇨 자료(환자 $442$ 명, 변수 $10$ 개)다. 혈청 변수
`s1` 과 `s2` 의 상관이 $0.897$ 로 가장 크다. 환자를 복원추출로 다시 뽑기를
$500$ 번 하고, 그때마다 같은 벌점으로 라쏘와 엘라스틱넷을 적합해 **어느 변수가
목록에 올랐는지** 센다.

라쏘는 둘 가운데 하나만 올리는 일이 잦고(그 하나가 표본마다 바뀐다), 엘라스틱넷은
둘을 함께 올린다. 이것이 그룹 효과다.

그리는 것:

  ① 변수마다 $500$ 번 가운데 몇 번 뽑혔는가 — 라쏘와 엘라스틱넷
  ② 상관 $0.897$ 인 `s1`·`s2` 짝에서 네 가지 결과(둘 다·s1 만·s2 만·둘 다 아님)의 비율

만드는 파일:

  ch18/lasso/img/which_one_got_picked.png

실행:  python3 scripts/make_ch18_which_one_got_picked.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, ElasticNet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/lasso/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

ALPHA = 2.0        # 두 모형에 같은 벌점 세기를 준다
L1R = 0.3          # 엘라스틱넷의 L1 몫
B = 500
SEED = 18


def run(model, X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    count = np.zeros(p)
    pair = np.zeros(4)          # 둘다없음 · s2만 · s1만 · 둘다
    i1, i2 = 4, 5               # s1, s2
    for _ in range(B):
        idx = rng.integers(0, n, n)
        model.fit(X[idx], y[idx])
        on = np.abs(model.coef_) > 1e-8
        count += on
        pair[int(on[i1]) * 2 + int(on[i2])] += 1
    return count / B, pair / B


def main():
    d = load_diabetes()
    X, y = d.data.copy(), d.target.copy()
    X = (X - X.mean(0)) / X.std(0)
    names = list(d.feature_names)
    r12 = np.corrcoef(X[:, 4], X[:, 5])[0, 1]

    lasso_f, lasso_p = run(Lasso(alpha=ALPHA, max_iter=50_000), X, y)
    en_f, en_p = run(ElasticNet(alpha=ALPHA, l1_ratio=L1R, max_iter=50_000), X, y)

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"s1 과 s2 의 상관 = {r12:.3f}   복원추출 {B} 번, 벌점 alpha = {ALPHA}")
    print("변수     라쏘    엘라스틱넷")
    for k in np.argsort(-lasso_f):
        print(f"{names[k]:5s} {lasso_f[k]:8.3f} {en_f[k]:11.3f}")
    lab = ["둘 다 아님", "s2 만", "s1 만", "둘 다"]
    print("\ns1·s2 짝의 결과")
    for i in range(4):
        print(f"  {lab[i]:10s} 라쏘 {lasso_p[i]:.3f}   엘라스틱넷 {en_p[i]:.3f}")
    one = lasso_p[1] + lasso_p[2]
    print(f"\n라쏘가 둘 가운데 **하나만** 올린 비율 {one:.3f} "
          f"— 그 가운데 s1 이 {lasso_p[2] / one:.1%}, s2 가 {lasso_p[1] / one:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 변수마다 뽑힌 비율 ──
    ax = axes[0]
    order = np.argsort(-lasso_f)
    ks = np.arange(len(names))
    w = 0.38
    ax.bar(ks - w / 2, lasso_f[order], width=w, color=RED, alpha=0.85,
           label="라쏘")
    ax.bar(ks + w / 2, en_f[order], width=w, color=TEAL, alpha=0.85,
           label=f"엘라스틱넷 ($\\ell_1$ 몫 ${L1R}$)")
    # 문턱 줄은 범례에 넣는다 — 그림 안에 글로 두면 막대와 겹친다
    ax.axhline(0.6, color=AMBER, lw=1.4, ls="--",
               label="안정성 선택의 흔한 문턱 $0.6$")
    ax.set_xticks(ks)
    ax.set_xticklabels([names[k] for k in order], fontsize=10.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f"{B} 번 가운데 뽑힌 비율", fontsize=11.5)
    ax.set_xlabel("변수 (라쏘가 자주 고른 차례)", fontsize=11.5)
    ax.set_title("① 같은 자료를 다시 뽑으면", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 상관된 짝에서 무슨 일이 ──
    ax = axes[1]
    labs = ["$s_1$ 만", "$s_2$ 만", "둘 다", "둘 다 아님"]
    lv = [lasso_p[2], lasso_p[1], lasso_p[3], lasso_p[0]]
    ev = [en_p[2], en_p[1], en_p[3], en_p[0]]
    ks = np.arange(4)
    ax.bar(ks - w / 2, lv, width=w, color=RED, alpha=0.85, label="라쏘")
    ax.bar(ks + w / 2, ev, width=w, color=TEAL, alpha=0.85, label="엘라스틱넷")
    for k in ks:
        ax.text(k - w / 2, lv[k] + 0.02, f"${lv[k]:.2f}$", ha="center",
                fontsize=10.6, color=RED, fontweight="bold")
        ax.text(k + w / 2, ev[k] + 0.02, f"${ev[k]:.2f}$", ha="center",
                fontsize=10.6, color=TEAL, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xticklabels(labs, fontsize=11.5)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel(f"{B} 번 가운데 비율", fontsize=11.5)
    ax.set_xlabel(f"상관 ${r12:.3f}$ 인 두 혈청 변수에 일어난 일", fontsize=11.5)
    ax.set_title("② 어느 하나가 오르는가", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("목록은 자료를 조금만 흔들어도 바뀐다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"당뇨 자료(환자 ${X.shape[0]}$ 명)를 복원추출로 ${B}$ 번 다시 뽑아 "
             f"같은 벌점(${ALPHA}$)으로 적합했다. 모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "which_one_got_picked.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
