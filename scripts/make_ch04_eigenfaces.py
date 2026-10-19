r"""4.6절 「AI로 보는 통계학」 — 고유벡터로 얼굴을 적는다.

정리 1 은 공분산행렬의 고유벡터가 등고선 타원의 축이라고 말한다. 이 쪽
그림은 그 축을 2 차원이 아니라 2304 차원에서 구한 것이다.

남의 사진을 쓰지 않으려고 얼굴을 직접 만들어 썼다. 매개변수 여섯 개
(머리 너비·높이, 눈 사이, 눈 크기, 입 굽이, 밝기)를 흔들어 48x48 그림을
600 장 그리고, 그 2304 차원 구름의 공분산행렬을 고유분해한다.

  평균 얼굴 + 고유얼굴 넷    앞쪽 축이 무엇을 담는지 보인다
  k 를 늘려 가며 되살린 것   매개변수가 여섯이므로 여섯 축이면 거의 끝난다

만드는 파일:

  ch04/bivariate_normal/img/eigenfaces.png

실행:  python3 scripts/make_ch04_eigenfaces.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch04/bivariate_normal/img/"

INK = "#37474F"
TEAL = "#00838F"

SIDE = 48
NFACE = 600
SEED = 17
KS = (1, 2, 4, 12)


def face(w, h, eye_gap, eye_size, mouth, bright):
    """매개변수 여섯 개로 그린 48x48 얼굴 한 장."""
    g = np.linspace(-1, 1, SIDE)
    X, Y = np.meshgrid(g, -g)

    head = np.exp(-(((X / w) ** 2 + (Y / h) ** 2) ** 3))
    img = bright * head

    for sgn in (-1, 1):
        d = (X - sgn * eye_gap) ** 2 + (Y - 0.28) ** 2
        img -= 0.9 * np.exp(-d / (2 * eye_size ** 2))

    ridge = (Y + 0.42 - mouth * X ** 2) ** 2
    img -= 0.7 * np.exp(-ridge / 0.004) * np.exp(-(X ** 2) / 0.18)
    return img.ravel()


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(NFACE):
        rows.append(face(w=rng.uniform(0.52, 0.74),
                         h=rng.uniform(0.68, 0.92),
                         eye_gap=rng.uniform(0.20, 0.34),
                         eye_size=rng.uniform(0.055, 0.095),
                         mouth=rng.uniform(-0.55, 0.55),
                         bright=rng.uniform(0.65, 1.0)))
    A = np.array(rows)

    mean = A.mean(axis=0)
    C = np.cov(A - mean, rowvar=False)
    lam, vec = np.linalg.eigh(C)              # 정리 1 이 말하는 고유분해
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    share = np.cumsum(lam) / lam.sum()

    for k in KS:
        print(f"    축 {k:2} 개로 총분산의 {share[k-1]*100:5.2f}%")

    target = A[0] - mean
    fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.6), constrained_layout=True)

    panels = [("평균 얼굴", mean)]
    for i in range(4):
        panels.append((f"고유얼굴 {i+1}   {lam[i]/lam.sum()*100:.1f}%", vec[:, i]))
    for ax, (title, v) in zip(axes[0], panels):
        ax.imshow(v.reshape(SIDE, SIDE), cmap="gray")
        ax.set_title(title, fontsize=10.6, color=TEAL, pad=5)

    for ax, k in zip(axes[1], KS):
        rec = mean + vec[:, :k] @ (vec[:, :k].T @ target)
        ax.imshow(rec.reshape(SIDE, SIDE), cmap="gray")
        ax.set_title(f"축 {k} 개 — 총분산 {share[k-1]*100:.1f}%",
                     fontsize=10.6, color=INK, pad=5)
    axes[1][-1].imshow((mean + target).reshape(SIDE, SIDE), cmap="gray")
    axes[1][-1].set_title("원본 — 되살릴 대상", fontsize=10.6, color=TEAL, pad=5)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("2304 차원 구름의 고유벡터도 결국 타원의 축이다", fontsize=14.5)
    fig.text(0.5, -0.035,
             f"매개변수 여섯 개로 그린 얼굴 {NFACE} 장이다. 윗줄은 공분산행렬의 "
             f"앞선 고유벡터 넷, 아랫줄은 그 축만으로 한 장을 되살린 것이다. "
             f"만든 자유도가 여섯이니 여섯 축에서 거의 멈춘다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "eigenfaces.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
