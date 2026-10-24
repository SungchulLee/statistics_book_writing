r"""두 학습 알고리즘을 견주는 네 가지 방법의 제1종 오류를 직접 잰다.

귀무가설이 **참이 되도록** 만든다. 참 모형이 두 특성 덩어리에 대해 대칭이고
알고리즘 A 는 앞 덩어리, B 는 뒤 덩어리만 쓴다. 기대 정확도가 정확히 같고
예측은 서로 다르다 — McNemar 가 쓰는 비대각선이 비지 않는다.

① 한 시험집합의 $2 \times 2$ 표.
② 네 검정의 제1종 오류율 (귀무가설이 참일 때).
③ 검정력 (한쪽이 정말 조금 나을 때).

만드는 파일: docs/ch13/test/img/mcnemar_ml.png
"""

import pathlib
import tempfile
import warnings

import numpy as np
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

RNG = np.random.default_rng(20261020)
K = 6               # 덩어리 하나의 특성 수
N_TR, N_TE = 400, 400
REPS = 1000
ALPHA = 0.05
N_RESAMPLE, N_FOLD = 30, 10


def make(n, rng, edge=0.0):
    """참 모형이 두 덩어리에 대해 대칭이다. edge > 0 이면 앞 덩어리에 힘을 싣는다."""
    X = rng.standard_normal((n, 2 * K))
    w = np.r_[np.full(K, 1.0 + edge), np.full(K, 1.0)]
    p = 1 / (1 + np.exp(-(X @ w) / np.sqrt(2 * K)))
    return X, (rng.random(n) < p).astype(int)


def fit_pred(Xtr, ytr, Xte, cols):
    m = LogisticRegression(max_iter=1000).fit(Xtr[:, cols], ytr)
    return m.predict(Xte[:, cols])


A_COLS, B_COLS = np.arange(K), np.arange(K, 2 * K)


def mcnemar_p(okA, okB):
    """어긋난 칸만 쓴다. n01 ~ Binomial(n01+n10, 1/2) 의 정확검정."""
    b = int(np.sum(okA & ~okB))          # A 만 맞음
    c = int(np.sum(~okA & okB))          # B 만 맞음
    if b + c == 0:
        return 1.0, b, c
    return stats.binomtest(b, b + c, 0.5).pvalue, b, c


def two_prop_p(okA, okB):
    """두 정확도를 **독립** 두 비율로 보고 견준다 (짝을 버린다)."""
    n = len(okA)
    p1, p2 = okA.mean(), okB.mean()
    pp = (okA.sum() + okB.sum()) / (2 * n)
    se = np.sqrt(2 * pp * (1 - pp) / n)
    if se == 0:
        return 1.0
    return 2 * stats.norm.sf(abs(p1 - p2) / se)


def run_once(rng, edge=0.0):
    Xtr, ytr = make(N_TR, rng, edge)
    Xte, yte = make(N_TE, rng, edge)
    okA = fit_pred(Xtr, ytr, Xte, A_COLS) == yte
    okB = fit_pred(Xtr, ytr, Xte, B_COLS) == yte
    p_mc, b, c = mcnemar_p(okA, okB)
    p_tp = two_prop_p(okA, okB)

    # 재표집 t — 같은 자료를 30 번 무작위로 갈라 차이를 모아 대응 t
    X = np.vstack([Xtr, Xte]); y = np.r_[ytr, yte]
    n = len(y); n_tr = int(0.7 * n)
    d = []
    for _ in range(N_RESAMPLE):
        idx = rng.permutation(n)
        tr, te = idx[:n_tr], idx[n_tr:]
        a = (fit_pred(X[tr], y[tr], X[te], A_COLS) == y[te]).mean()
        bb = (fit_pred(X[tr], y[tr], X[te], B_COLS) == y[te]).mean()
        d.append(a - bb)
    d = np.array(d)
    p_rs = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0

    # 10 겹 교차검증 t
    d2 = []
    for tr, te in KFold(N_FOLD, shuffle=True,
                        random_state=int(rng.integers(1 << 31))).split(X):
        a = (fit_pred(X[tr], y[tr], X[te], A_COLS) == y[te]).mean()
        bb = (fit_pred(X[tr], y[tr], X[te], B_COLS) == y[te]).mean()
        d2.append(a - bb)
    d2 = np.array(d2)
    p_cv = stats.ttest_1samp(d2, 0).pvalue if d2.std() > 0 else 1.0
    return dict(mc=p_mc, tp=p_tp, rs=p_rs, cv=p_cv), (b, c, okA.mean(), okB.mean())


NAMES = [("mc", "McNemar (정확)", "#1f77b4"),
         ("tp", "두 비율 검정 (짝을 버림)", "#9e9e9e"),
         ("rs", f"재표집 $t$ (${N_RESAMPLE}$ 번 다시 가름)", "#ff7f0e"),
         ("cv", f"${N_FOLD}$ 겹 교차검증 $t$", "#d62728")]

# 모의가 비싸므로 결과를 한 번 재면 캐시해 둔다. 그림만 손볼 때 다시 돌리지
# 않으려는 것이고, 캐시가 없거나 반복 수가 다르면 처음부터 다시 잰다.
CACHE = pathlib.Path(tempfile.gettempdir()) / f"mcnemar_ml_{REPS}.npz"
res = {}
if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    res = z["res"].item()
    print(f"캐시에서 읽음 ({CACHE})")
for edge, tag in ((0.0, "H0 참"), (0.25, "A 가 조금 나음")):
    if tag in res:
        continue
    acc = {k: [] for k, *_ in NAMES}
    last = None
    for i in range(REPS):
        ps, info = run_once(RNG, edge)
        for k in acc:
            acc[k].append(ps[k] < ALPHA)
        last = info
    res[tag] = ({k: np.mean(v) for k, v in acc.items()}, last)
    print(f"{tag}: " + "  ".join(f"{k} {np.mean(v) * 100:.1f}%" for k, v in acc.items())
          + f"   (마지막 판: b={last[0]}, c={last[1]}, "
            f"정확도 {last[2]:.3f} / {last[3]:.3f})")
    np.savez(CACHE, res=np.array(res, dtype=object))

# ① 에 쓸 한 판 — 어긋난 칸이 동점이면 p = 1 이라 그림이 아무 말도 못 한다.
# 씨앗을 바꿔 가며 전형적인 판(비기지 않은 판) 하나를 고른다.
for seed in range(1, 200):
    rng1 = np.random.default_rng(seed)
    Xtr, ytr = make(N_TR, rng1); Xte, yte = make(N_TE, rng1)
    oA = fit_pred(Xtr, ytr, Xte, A_COLS) == yte
    oB = fit_pred(Xtr, ytr, Xte, B_COLS) == yte
    TAB = np.array([[np.sum(oA & oB), np.sum(oA & ~oB)],
                    [np.sum(~oA & oB), np.sum(~oA & ~oB)]])
    p_tab, _, _ = mcnemar_p(oA, oB)
    if 0.1 < p_tab < 0.6:
        break
print(f"   (① 에 쓴 씨앗 {seed})")
print(f"\n① 표 {TAB.tolist()}  정확도 {oA.mean():.3f} / {oB.mean():.3f}  "
      f"McNemar p = {p_tab:.3f}")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

im = ax1.imshow(TAB, cmap="Blues", vmin=0, vmax=TAB.max())
for i in range(2):
    for j in range(2):
        off = i != j
        ax1.text(j, i, f"{TAB[i, j]}", ha="center", va="center",
                 fontsize=21 if off else 16,
                 fontweight="bold" if off else "normal",
                 color="#d62728" if off else
                 ("white" if TAB[i, j] > TAB.max() * 0.6 else "#333333"))
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["맞음", "틀림"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["맞음", "틀림"])
ax1.set_xlabel("모형 B"); ax1.set_ylabel("모형 A")
ax1.set_title(f"① 같은 시험집합 ${N_TE}$ 개", fontsize=12, pad=26)
ax1.text(0.5, 1.03, f"어긋난 칸 ${TAB[0, 1]}$ 대 ${TAB[1, 0]}$ — "
                    f"McNemar $p = {p_tab:.3f}$",
         ha="center", fontsize=10.5, color="#d62728", transform=ax1.transAxes)

for ax, tag, ttl, ref in ((ax2, "H0 참", "② 귀무가설이 참일 때 — 제1종 오류율", ALPHA * 100),
                          (ax3, "A 가 조금 나음", "③ 한쪽이 정말 나을 때 — 검정력", None)):
    vals = [res[tag][0][k] * 100 for k, *_ in NAMES]
    cols = [c for *_, c in NAMES]
    err = [100 * np.sqrt(v / 100 * (1 - v / 100) / REPS) * 1.96 for v in vals]
    ax.bar(range(4), vals, 0.6, color=cols, yerr=err, capsize=5,
           error_kw=dict(ecolor="#333333", lw=1.2))
    for i, v in enumerate(vals):
        ax.text(i, v + err[i] + max(vals) * 0.025, f"${v:.1f}\\%$", ha="center",
                fontsize=10.5)
    if ref is not None:
        ax.axhline(ref, color="black", lw=1.5, ls="--")
        ax.text(-0.45, ref * 0.28, f"명목 ${ALPHA}$", ha="left", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n for _, n, _ in NAMES], fontsize=9, rotation=14,
                       ha="right")
    ax.set_ylabel("기각한 비율 ($\\%$)")
    ax.set_title(ttl, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch13/test/img/mcnemar_ml.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

print(f"\n반복 {REPS} 회, 시험집합 {N_TE} 개, alpha = {ALPHA}")
print(f"{'검정':>28}{'제1종 오류':>12}{'검정력':>10}")
for k, n, _ in NAMES:
    print(f"{n:>28}{res['H0 참'][0][k] * 100:>11.1f}%"
          f"{res['A 가 조금 나음'][0][k] * 100:>9.1f}%")
