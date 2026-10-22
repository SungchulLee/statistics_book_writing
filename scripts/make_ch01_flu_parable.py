r"""1.6절 「그림으로 보는 통계학」 — 겨울 탐지기.

$2013$ 년 $2$ 월, 검색어로 독감 유행을 좇던 시스템이 질병통제센터 추정치의 **두 배**를
가리켰다. 레이저 등($2014$)은 그 까닭을 둘로 갈랐다 — **큰 자료의 오만**과
**알고리즘 동역학**이다. 이 스크립트는 앞엣것을 직접 만든 자료로 재현한다.

후보를 많이 볼수록 훈련 적합은 좋아지지만, 뽑히는 것은 「독감을 아는 변수」가 아니라
**겨울을 아는 변수**다. 그래서 계절 밖에서 유행이 오면 통째로 놓친다.

① 후보 수를 늘리면 — 훈련은 좋아지고 비계절 유행은 못 잡는다.
② 그 해의 그림 — 검색어 모형은 여름 유행을 보지 못한다.
③ 오래된 자료가 더 낫고, 둘을 합치면 더 낫다.

만드는 파일: docs/ch01/contrast/img/flu_parable.png
실행:  python3 scripts/make_ch01_flu_parable.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/contrast/img/flu_parable.png"
RNG = np.random.default_rng(20261022)

WEEKS = 52
N_TRAIN = 1152          # 레이저 등이 적은 「자료점 1,152 개」
N_TEST = 3 * WEEKS
N_SEL = 45              # 「검색어 45 개」
KMAX = 12_000
t_all = np.arange(N_TRAIN + N_TEST)


def season(t, phase=0.0):
    """한 해를 주기로 하는 모양. phase 로 봉우리 자리를 옮긴다."""
    w = 2 * np.pi * (t / WEEKS + phase)
    return np.sin(w) + 0.35 * np.sin(2 * w)


# --- 참 ILI: 계절 유행 + 잡음, 그리고 시험 구간에 비계절 유행 ----------
amp = RNG.uniform(0.8, 1.2, (N_TRAIN + N_TEST) // WEEKS + 1).repeat(WEEKS)
noise = np.zeros(len(t_all))
for i in range(1, len(t_all)):
    noise[i] = 0.72 * noise[i - 1] + RNG.normal(0, 0.22)
ili = 3.0 + 1.3 * amp[: len(t_all)] * season(t_all, phase=-0.25) + noise

PAND = slice(N_TRAIN + 34, N_TRAIN + 54)       # 계절 골짜기에 터지는 유행
bump = np.zeros(len(t_all))
bump[PAND] = 3.4 * np.exp(-0.5 * ((np.arange(20) - 9) / 4.5) ** 2)
ili = ili + bump

# --- 후보 검색어: 모두 「겨울을 아는」 계열이다 -------------------------
phases = RNG.uniform(-0.18, 0.18, KMAX)
scale = RNG.uniform(0.5, 1.5, KMAX)
cand = (scale[:, None] * season(t_all[None, :], phases[:, None])
        + RNG.normal(0, 0.9, (KMAX, len(t_all)))).astype(np.float32)

tr = slice(0, N_TRAIN)
te = slice(N_TRAIN, N_TRAIN + N_TEST)
y_tr = ili[tr]
yc = y_tr - y_tr.mean()
cc = cand[:, tr] - cand[:, tr].mean(axis=1, keepdims=True)
corr = (cc @ yc) / (np.linalg.norm(cc, axis=1) * np.linalg.norm(yc))


def fit_terms(idx):
    """고른 검색어들로 최소제곱 적합. 훈련·시험 예측을 돌려준다."""
    Xtr = np.column_stack([np.ones(N_TRAIN), cand[idx][:, tr].T])
    b, *_ = np.linalg.lstsq(Xtr, y_tr, rcond=None)
    Xte = np.column_stack([np.ones(N_TEST), cand[idx][:, te].T])
    return Xtr @ b, Xte @ b


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- ① 후보마다 「훈련 상관」과 「유행 구간 상관」을 함께 잰다 ----------
pand_rel = slice(34, 54)                      # 시험 구간 안에서 유행 자리
season_rel = np.setdiff1d(np.arange(N_TEST), np.arange(34, 54))
yp = ili[te][pand_rel] - ili[te][pand_rel].mean()
cp = cand[:, te][:, pand_rel] - cand[:, te][:, pand_rel].mean(axis=1, keepdims=True)
corr_p = (cp @ yp) / (np.linalg.norm(cp, axis=1) * np.linalg.norm(yp))

idx = np.argsort(-np.abs(corr))[:N_SEL]
print(f"① 후보 {KMAX:,} 개 — 훈련 상관 평균 {np.abs(corr).mean():.3f}, "
      f"유행 구간 상관 평균 {np.abs(corr_p).mean():.3f}")
print(f"   뽑힌 {N_SEL} 개 — 훈련 상관 {corr[idx].mean():+.3f}, "
      f"유행 구간 상관 {corr_p[idx].mean():+.3f}")
pred_tr, pred_te = fit_terms(idx)
print(f"\n뽑힌 {N_SEL} 개의 훈련 상관: 최대 {np.abs(corr[idx]).max():.3f}, "
      f"최소 {np.abs(corr[idx]).min():.3f}")

# --- ③ 3 주 지난 자료로 만든 모형, 그리고 둘을 합친 모형 ---------------
LAG = 3
Xl_tr = np.column_stack([np.ones(N_TRAIN - LAG), ili[tr][:-LAG]])
bl, *_ = np.linalg.lstsq(Xl_tr, y_tr[LAG:], rcond=None)
lag_te = bl[0] + bl[1] * ili[N_TRAIN - LAG: N_TRAIN + N_TEST - LAG]

Xb_tr = np.column_stack([np.ones(N_TRAIN - LAG), pred_tr[LAG:], ili[tr][:-LAG]])
bb, *_ = np.linalg.lstsq(Xb_tr, y_tr[LAG:], rcond=None)
both_te = bb[0] + bb[1] * pred_te + bb[2] * ili[N_TRAIN - LAG: N_TRAIN + N_TEST - LAG]

y_te = ili[te]
models = {"검색어 45 개": pred_te, f"{LAG} 주 지난 자료": lag_te, "둘을 합친 것": both_te}
print()
for nm, p in models.items():
    e = p - y_te
    ac = float(np.corrcoef(e[:-1], e[1:])[0, 1])
    print(f"③ {nm:>12s}: 전체 RMSE {rmse(p, y_te):.3f}   "
          f"유행 구간 {rmse(p[pand_rel], y_te[pand_rel]):.3f}   "
          f"오차의 1 차 자기상관 {ac:+.2f}")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"

ax = axes[0]
sub = RNG.choice(KMAX, 4000, replace=False)
ax.scatter(corr[sub], corr_p[sub], s=6, color="0.72", alpha=0.6, label="후보 검색어")
ax.scatter(corr[idx], corr_p[idx], s=34, color=C2, zorder=3,
           label=f"뽑힌 {N_SEL} 개")
ax.axhline(0, color="0.4", lw=1)
ax.axvline(0, color="0.4", lw=1)
ax.set_xlabel("훈련 구간에서 참 유행과의 상관")
ax.set_ylabel("비계절 유행 구간에서의 상관")
ax.set_title(f"① 후보 {KMAX:,} 개 — 무엇이 뽑히는가", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.text(0.03, 0.06,
        f"뽑힌 것들의 훈련 상관 평균 {corr[idx].mean():+.2f}\n"
        f"유행 구간에서는 {corr_p[idx].mean():+.2f} — 거꾸로 간다",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[1]
w = np.arange(N_TEST)
ax.plot(w, y_te, color="0.35", lw=2.2, label="참 유행 수준")
ax.plot(w, pred_te, color=C2, lw=1.8, label="검색어 45 개")
ax.plot(w, lag_te, color=C3, lw=1.6, ls="--", label=f"{LAG} 주 지난 자료")
ax.axvspan(34, 54, color=C2, alpha=0.12)
ax.text(44, ax.get_ylim()[0] + 0.12, "계절 밖 유행", color=C2, fontsize=9,
        ha="center", va="bottom")
ax.set_xlabel("시험 구간의 주")
ax.set_ylabel("유행 수준 (%)")
ax.set_title("② 겨울만 아는 모형이 놓치는 것", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[2]
names = list(models)
vals = [rmse(models[n], y_te) for n in names]
vals_p = [rmse(models[n][pand_rel], y_te[pand_rel]) for n in names]
xx = np.arange(3)
ax.bar(xx - 0.2, vals, 0.4, color=C1, label="시험 구간 전체")
ax.bar(xx + 0.2, vals_p, 0.4, color=C2, label="유행 구간만")
for i, (a, b) in enumerate(zip(vals, vals_p)):
    ax.text(i - 0.2, a + 0.03, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 0.03, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("RMSE")
ax.set_title("③ 오래된 자료가 낫고, 합치면 더 낫다", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
