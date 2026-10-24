r"""11.8절 「그림으로 보는 통계학」 — 「$17\%$ 로 최선일 확률」을 믿을 수 있는가.

이 쪽의 $7$ 절은 아카이케 가중치를 확률처럼 읽는다 — **「모형 B 가 셋 가운데 최선일
근사적 확률이 $82\%$, 모형 C 가 그럴 확률이 $17\%$」.** 그 읽기가 얼마나 맞는지를
재어 본다.

후보 셋을 만든다(이 쪽의 A·B·C 와 같은 크기다).

  후보 1 : 설명변수 1 개          후보 2 : 설명변수 3 개 (참 모형)
  후보 3 : 설명변수 6 개 (참 모형에 계수 0 인 셋을 더 얹은 것)

관측 $50$ 개로 적합해 가중치를 구하고, **새로 뽑은 $20{,}000$ 개**에서 세 적합이
실제로 어느 쪽을 더 잘 맞히는지 본다. 이것을 $4000$ 번 되풀이한다.

이 스크립트가 재는 값:

  후보        받은 평균 가중치   실제로 가장 잘 맞힌 비율
  1 (1 개)        0.000                 0.000
  2 (3 개)        0.737                 0.975
  3 (6 개)        0.263                 0.025

  가중치가 큰 모형(후보 3)을 가리킨 횟수 = 627 번, 그때의 평균 가중치 0.732
    그 627 번 가운데 큰 모형이 실제로 최선이었던 횟수 = 0 번
  큰 모형이 실제로 최선이었던 99 번에 그 모형이 받은 평균 가중치 = 0.077

  그런데 이긴 모형만 보면 평균 가중치 0.809, 실제로 최선이었던 비율 0.819 다.

곧 **평균은 맞는데 조각은 거꾸로다.** 가중치가 큰 모형을 가리킬 때는 틀리고, 큰 모형이
정말 나을 때는 가중치가 낮다.

만드는 파일:

  ch11/model_selection/img/akaike_weight_calibration.png

실행:  python3 scripts/make_ch13_akaike_weight_calibration.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 50             # 적합에 쓰는 관측 수
NT = 20_000        # 새로 뽑아 맞혀 보는 관측 수
REPS = 4000
SIGMA = 1.0
BETA = np.array([2.0, 1.5, 1.0, 0.0, 0.0, 0.0])
CAND = ([0], [0, 1, 2], [0, 1, 2, 3, 4, 5])
LABELS = ("후보 $1$\n(설명변수 $1$ 개)", "후보 $2$\n(설명변수 $3$ 개)",
          "후보 $3$\n(설명변수 $6$ 개)")
SEED = 3


def one(r):
    """한 번의 자료에서 가중치와, 새 자료에서 실제로 가장 잘 맞힌 후보."""
    rr = np.random.default_rng([SEED, r])
    X = rr.normal(size=(N, 6))
    y = X @ BETA + rr.normal(0, SIGMA, N)
    Xt = rr.normal(size=(NT, 6))
    yt = Xt @ BETA + rr.normal(0, SIGMA, NT)
    aic, tmse = [], []
    for c in CAND:
        D = np.column_stack([np.ones(N)] + [X[:, j] for j in c])
        b = np.linalg.lstsq(D, y, rcond=None)[0]
        rssv = ((y - D @ b) ** 2).sum()
        k = D.shape[1] + 1                     # 분산도 모수로 센다
        aic.append(N * np.log(rssv / N) + 2 * k)
        Dt = np.column_stack([np.ones(NT)] + [Xt[:, j] for j in c])
        tmse.append(float(((yt - Dt @ b) ** 2).mean()))
    d = np.array(aic) - min(aic)
    w = np.exp(-d / 2)
    return w / w.sum(), int(np.argmin(tmse))


def main():
    W, B = [], []
    for r in range(REPS):
        w, b = one(r)
        W.append(w)
        B.append(b)
    W, B = np.array(W), np.array(B)
    win = W.argmax(axis=1)

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"되풀이 {REPS} 번 — 적합 {N} 개, 맞혀 보기 {NT} 개")
    print("  후보        받은 평균 가중치   실제로 가장 잘 맞힌 비율")
    mean_w, hit = [], []
    for i, c in enumerate(CAND):
        mean_w.append(float(W[:, i].mean()))
        hit.append(float((B == i).mean()))
        print(f"  {i + 1} ({len(c)} 개) {mean_w[i]:16.3f} {hit[i]:21.3f}")

    nC = int((win == 2).sum())
    both = int(((win == 2) & (B == 2)).sum())
    cbest = int((B == 2).sum())
    print(f"\n  가중치가 큰 모형(후보 3)을 가리킨 횟수 = {nC} 번, "
          f"그때의 평균 가중치 {W[win == 2, 2].mean():.3f}")
    print(f"    그 가운데 큰 모형이 실제로 최선이었던 횟수 = {both} 번")
    print(f"  큰 모형이 실제로 최선이었던 {cbest} 번에 그 모형이 받은 평균 가중치 "
          f"= {W[B == 2, 2].mean():.3f}")
    wwin = W[np.arange(REPS), win]
    print(f"\n  이긴 모형만 보면 — 평균 가중치 {wwin.mean():.3f}, "
          f"실제로 최선이었던 비율 {(win == B).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    # ── ① 받은 가중치 대 실제 비율 ──
    ax = axes[0]
    xs = np.arange(3)
    w = 0.34
    ax.bar(xs - w / 2, mean_w, width=w * 0.92, color=AMBER, alpha=0.9,
           label="아카이케 가중치가 준 몫 (평균)")
    ax.bar(xs + w / 2, hit, width=w * 0.92, color=TEAL, alpha=0.9,
           label="새 자료에서 실제로 가장 잘 맞힌 비율")
    for i in range(3):
        ax.text(i - w / 2, mean_w[i] + 0.022, f"${mean_w[i]:.3f}$", ha="center",
                fontsize=11, color=AMBER, fontweight="bold")
        ax.text(i + w / 2, hit[i] + 0.022, f"${hit[i]:.3f}$", ha="center",
                fontsize=11, color=TEAL, fontweight="bold")
    ax.annotate("$0.263$ 대 $0.025$ — 열 배다",
                xy=(2 + w / 2, hit[2] + 0.03), xytext=(1.36, 0.60),
                fontsize=11.4, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS, fontsize=10.8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_title("① 큰 모형이 받는 몫과 실제로 이기는 몫", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 2 x 2 — 가리킨 것과 맞은 것 ──
    ax = axes[1]
    tab = np.zeros((2, 2), int)
    for a in (1, 2):
        for b in (1, 2):
            tab[a - 1, b - 1] = int(((win == a) & (B == b)).sum())
    ax.imshow(tab, cmap="Oranges", vmin=0, vmax=tab.max() * 1.25)
    for i in range(2):
        for j in range(2):
            col = "white" if tab[i, j] > tab.max() * 0.6 else INK
            ax.text(j, i, f"${tab[i, j]}$ 번", ha="center", va="center",
                    fontsize=19, color=col, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["참 모형(후보 $2$)", "큰 모형(후보 $3$)"], fontsize=11.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["가중치가\n참 모형을\n가리켰다",
                        "가중치가\n큰 모형을\n가리켰다"], fontsize=11.4)
    ax.set_xlabel("새 자료에서 실제로 더 잘 맞힌 모형", fontsize=11.5)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.text(0.5, -0.80, f"큰 모형을 가리킨 {nC} 번 가운데 맞은 것은 ${both}$ 번이다",
            ha="center", fontsize=11.6, color=RED, fontweight="bold")
    ax.set_title("② 가리킨 것과 맞은 것", fontsize=12.8, color=INK, pad=62)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    fig.suptitle("확률처럼 읽은 가중치를 되짚어 보면", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개로 적합하고 새로 뽑은 ${NT // 1000}$ 천 개에서 맞혀 본 "
             f"것을 ${REPS}$ 번 되풀이했다. 참 계수는 앞의 셋뿐이고 나머지 셋은 "
             f"$0$ 이다. 모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "akaike_weight_calibration.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
