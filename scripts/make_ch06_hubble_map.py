r"""6.4절 「그림으로 보는 통계학」 — 거울을 못 고치는 동안.

$1990$ 년 올라간 허블 우주망원경의 주경은 가장자리가 $2.2$ 마이크로미터 평평하게
깎여 있었다. 수리까지 세 해 반 동안, 흐려진 영상은 **통계적 복원**으로 쓰였다.
널리 쓰인 리처드슨–루시 반복은 포아송 가능도의 EM, 곧 **MLE** 다. 그런데 반복을
오래 돌리면 잡음이 폭발한다 — 사전분포를 더해 **MAP** 로 바꾸면 그것이 멈춘다.

① 참 모습 ② 흐려진 관측 ③ 오차 곡선 ④ ML 50 회 ⑤ ML 1,000 회 ⑥ MAP 1,000 회

만드는 파일: docs/ch06/bayesian/img/hubble_map.png
실행:  python3 scripts/make_ch06_hubble_map.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import ndimage, signal

OUT = "docs/ch06/bayesian/img/hubble_map.png"
RNG = np.random.default_rng(20261022)

M = 128
BG = 1.0                     # 배경 밝기
BETA = 0.01                  # 사전분포의 세기


def make_truth():
    """은하 하나와 별 셋 — 퍼진 구조가 대부분이다."""
    yy, xx = np.mgrid[0:M, 0:M]
    t = 120 * np.exp(-0.5 * (((yy - 64) / 16.0) ** 2 + ((xx - 60) / 9.0) ** 2))
    t += 60 * np.exp(-0.5 * (((yy - 64) / 34.0) ** 2 + ((xx - 60) / 22.0) ** 2))
    r = np.hypot(yy - 64, xx - 60)
    th = np.arctan2(yy - 64, xx - 60)
    t += 45 * np.exp(-0.5 * ((r - 22) / 4.0) ** 2) * np.clip(np.cos(2 * th - r / 6.0), 0, 1)
    for r0, c0, b in [(34, 40, 1500), (96, 92, 2500), (40, 98, 900)]:
        t[r0, c0] += b
    return t


def aberrated_psf():
    """수차가 있는 점퍼짐함수 — 좁은 심 + 넓은 무리."""
    k = 41
    yy, xx = np.mgrid[0:k, 0:k] - k // 2
    r2 = yy ** 2 + xx ** 2
    core = np.exp(-0.5 * r2 / 1.3 ** 2)
    halo = np.exp(-0.5 * r2 / 5.5 ** 2)
    psf = 0.30 * core / core.sum() + 0.70 * halo / halo.sum()
    return psf / psf.sum()


def conv(a, p):
    return signal.fftconvolve(a, p, mode="same")


truth = make_truth()
psf = aberrated_psf()
clean = conv(truth, psf) + BG
obs = RNG.poisson(clean).astype(float)
corner = (slice(0, 24), slice(100, 128))
print(f"참 영상 총광량 {truth.sum():,.0f},  관측 총광량 {obs.sum():,.0f}")
print(f"점퍼짐함수: 심 {0.30:.2f} / 무리 {0.70:.2f},  무리의 폭 {5.5:.1f} 화소")


def rl(y, p, iters, beta=0.0, keep=()):
    """리처드슨–루시(=포아송 EM). beta > 0 이면 매끄러움 사전분포를 더한 MAP."""
    x = np.full_like(y, y.mean())
    pf = p[::-1, ::-1]
    snaps, errs = {}, []
    for k in range(1, iters + 1):
        est = conv(x, p) + BG
        x = x * conv(y / np.maximum(est, 1e-10), pf)
        if beta > 0:
            # 한 걸음 늦은(OSL) 벌점: 이웃과 다른 만큼 끌어내린다
            pen = 1 + 2 * beta * (x - ndimage.uniform_filter(x, 3)) / np.maximum(x, 1e-6)
            x = x / np.maximum(pen, 0.2)
        errs.append(np.sqrt(np.mean((x - truth) ** 2)))
        if k in keep:
            snaps[k] = x.copy()
    return x, np.array(errs), snaps


ITERS = 1000
KEEP = (50, ITERS)
_, err_ml, snap_ml = rl(obs, psf, ITERS, 0.0, KEEP)
_, err_map, snap_map = rl(obs, psf, ITERS, BETA, KEEP)
print(f"\nRMSE — 관측 {np.sqrt(np.mean((obs - truth) ** 2)):8.2f}")
for k in KEEP:
    print(f"   ML  {k:>5} 회: {err_ml[k-1]:8.2f}     MAP {k:>5} 회: {err_map[k-1]:8.2f}")
best_ml = int(np.argmin(err_ml)) + 1
print(f"   빈 구석의 표준편차 — 참값 {truth[corner].std():.2f}, 관측 {obs[corner].std():.2f}, "
      f"ML {ITERS} 회 {snap_ml[ITERS][corner].std():.2f}, MAP {ITERS} 회 "
      f"{snap_map[ITERS][corner].std():.2f}")
print(f"   ML 은 {best_ml} 회에서 가장 좋고(RMSE {err_ml[best_ml-1]:.2f}) "
      f"그 뒤로는 나빠진다 — {ITERS} 회에서 {err_ml[-1]/err_ml[best_ml-1]:.1f} 배")
print(f"   MAP 은 {int(np.argmin(err_map))+1} 회에서 가장 좋고 그 뒤로 거의 평평하다 "
      f"({ITERS} 회 RMSE {err_map[-1]:.2f})")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13.6, 9.0))


def show(ax, img, title, vmax=None):
    ax.imshow(np.sqrt(np.maximum(img, 0)), cmap="bone", origin="lower",
              vmax=vmax if vmax else np.sqrt(np.percentile(img, 99.9)))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


V = np.sqrt(np.percentile(truth, 99.95))
show(axes[0, 0], truth, "① 참 모습", V)
show(axes[0, 1], obs, "② 수차가 있는 관측", V)

ax = axes[0, 2]
it = np.arange(1, ITERS + 1)
ax.plot(it, err_ml, color="#c0532b", lw=2, label="ML (리처드슨–루시)")
ax.plot(it, err_map, color="#1f6f8b", lw=2, label=f"MAP (사전분포 β = {BETA})")
ax.axvline(best_ml, color="#c0532b", ls=":", lw=1.2)
ax.text(best_ml * 1.15, err_ml.max() * 0.6, f"ML 의 최적 {best_ml} 회",
        color="#c0532b", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("반복 횟수")
ax.set_ylabel("참 영상과의 RMSE")
ax.set_title("③ 오래 돌릴수록 나빠지는 쪽", fontsize=11)
ax.legend(fontsize=9)

show(axes[1, 0], snap_ml[50], f"④ ML {50} 회", V)
show(axes[1, 1], snap_ml[ITERS], f"⑤ ML {ITERS:,} 회 — 잡음이 솟는다", V)
show(axes[1, 2], snap_map[ITERS], f"⑥ MAP {ITERS:,} 회", V)

fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"\n저장: {OUT}")
