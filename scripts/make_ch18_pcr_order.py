r"""18.5절 「그림으로 보는 통계학」 — 분산의 차례는 쓸모의 차례가 아니다.

PCR 은 성분을 **$\mathbf X$ 의 분산이 큰 차례로** 세우고 앞에서부터 $M$ 개를 쓴다.
그 차례가 $y$ 와는 아무 상관이 없다는 것을 하디와 링이 $1998$ 년에 지적했다.

여기서는 그 말을 실제 자료 하나에서 수로 확인한다. 자료는 최소각회귀 논문
(에프런·헤이스티·존스톤·팁시라니, $2004$)이 쓴 당뇨 자료로, 환자 $442$ 명의
기준 변수 $10$ 개와 한 해 뒤의 질병 진행 정도다. scikit-learn 에 들어 있다.

성분들이 직교하므로 $R^2$ 가 성분마다 **그대로 더해진다.** 그래서 「이 성분
하나가 설명하는 몫」을 따로 적을 수 있다.

  성분 $1$   분산 $40.2\%$   홀로 $R^2 = 0.308$
  성분 $2$   분산 $14.9\%$   홀로 $R^2 = 0.038$
  성분 $3$   분산 $12.1\%$   홀로 $R^2 = 0.026$
  성분 $4$   분산  $9.6\%$   홀로 $R^2 = 0.128$   <- 둘·셋을 합친 것의 두 배
  성분 $5$   분산  $6.6\%$   홀로 $R^2 = 0.000$

그리는 것:

  ① 성분마다 분산 몫과 홀로 설명하는 몫을 나란히
  ② 분산 차례로 쌓은 $R^2$ 와 쓸모 차례로 쌓은 $R^2$

만드는 파일:

  ch18/pcr_pls/img/pcr_order.png

실행:  python3 scripts/make_ch18_pcr_order.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.datasets import load_diabetes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/pcr_pls/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"


def main():
    d = load_diabetes()
    X, y = d.data.copy(), d.target.copy()
    X = (X - X.mean(0)) / X.std(0)          # 1 단계: 표준화
    U, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = U * s                                # 2 단계: 주성분 점수
    p = Z.shape[1]
    var = s ** 2 / (s ** 2).sum()
    yc = y - y.mean()
    r2 = np.array([np.corrcoef(Z[:, k], yc)[0, 1] ** 2 for k in range(p)])

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"당뇨 자료 — 환자 {X.shape[0]} 명, 변수 {p} 개")
    print("성분   분산 몫    홀로 R^2")
    for k in range(p):
        print(f"{k+1:3d}  {var[k]*100:8.2f}%  {r2[k]:9.4f}")

    cum_var = np.cumsum(r2)                       # 분산 차례 = 그대로
    order = np.argsort(-r2)
    cum_rel = np.cumsum(r2[order])                # 쓸모 차례
    m90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
    print(f"\n분산 차례로 쌓은 R^2  : {np.round(cum_var, 4)}")
    print(f"쓸모 차례로 쌓은 R^2  : {np.round(cum_rel, 4)}")
    print(f"쓸모 차례 둘 {cum_rel[1]:.4f}  >  분산 차례 셋 {cum_var[2]:.4f}")
    print(f"분산 {90}% 를 담으려면 성분 {m90} 개가 필요하다 "
          f"(그때 R^2 = {cum_var[m90-1]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 성분마다 두 가지 몫 ──
    ax = axes[0]
    ks = np.arange(1, p + 1)
    w = 0.38
    ax.bar(ks - w / 2, var * 100, width=w, color=MUTED, alpha=0.85,
           label="$\\mathbf{X}$ 의 분산 가운데 몫")
    ax.bar(ks + w / 2, r2 * 100, width=w, color=TEAL, alpha=0.9,
           label="이 성분 홀로 설명하는 $y$ 의 몫 ($R^2$)")
    ax.bar([4 + w / 2], [r2[3] * 100], width=w, color=RED, alpha=0.95)
    ax.annotate(f"성분 $4$ — 분산은 넷째인데\n$R^2$ 는 둘째다 "
                f"(${r2[3]:.3f}$)",
                xy=(4 + w / 2, r2[3] * 100), xytext=(5.2, 26),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xticks(ks)
    ax.set_xlabel("주성분 (분산이 큰 차례)", fontsize=11.5)
    ax.set_ylabel("백분율 ($\\%$)", fontsize=11.5)
    ax.set_ylim(0, 45)
    ax.set_title("① 두 차례가 다르다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 쌓아 보면 ──
    ax = axes[1]
    ax.plot(ks, cum_var, "o-", ms=8, lw=2.6, color=BLUE,
            label="PCR 이 쓰는 차례 (분산이 큰 것부터)")
    ax.plot(ks, cum_rel, "s--", ms=7, lw=2.2, color=TEAL,
            label="$y$ 와 가까운 것부터 썼다면")
    ax.axvline(m90, color=AMBER, lw=1.6, ls=":")
    ax.text(m90 - 0.12, 0.13, f"분산 $90\\%$ 를 담는 자리 — 성분 {m90} 개",
            rotation=90, va="bottom", ha="right", fontsize=10.6, color=AMBER)
    ax.annotate(f"둘로 ${cum_rel[1]:.3f}$", xy=(2, cum_rel[1]), xytext=(1.35, 0.505),
                fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate(f"셋으로 ${cum_var[2]:.3f}$", xy=(3, cum_var[2]),
                xytext=(3.5, 0.275), fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_xticks(ks)
    ax.set_xlabel("쓴 성분의 수 $M$", fontsize=11.5)
    ax.set_ylabel("설명된 $y$ 의 몫 ($R^2$)", fontsize=11.5)
    ax.set_ylim(0.0, 0.58)
    ax.set_title("② 같은 성분, 다른 차례", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("가장 큰 성분이 가장 쓸모 있는 성분은 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "최소각회귀 논문이 쓴 당뇨 자료(환자 $442$ 명, 변수 $10$ 개)다. "
             "성분이 직교하므로 $R^2$ 가 성분마다 그대로 더해진다. "
             "모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "pcr_order.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
