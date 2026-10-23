r"""엘라스틱넷이 이기는 조건 셋을 하나씩 떼어 본다.

이 쪽 절 4 는 엘라스틱넷이 「능형이나 라쏘 단독보다 나은 예측 정확도를 자주
달성」하며 특히 세 상황에서 그렇다고 적는다.

  1. 참 모형의 0 이 아닌 계수 개수가 중간 정도일 때
  2. 설명변수가 집단으로 상관되어 있을 때
  3. p 가 n 과 비슷하거나 더 클 때

셋을 요인설계로 갈라 각각이 얼마나 값을 하는지 잰다. 비교 상대는 능형과 라쏘이고,
엘라스틱넷의 alpha 는 **안쪽 값만**(0.1-0.9) 쓴다. 끝값을 넣으면 엘라스틱넷이
라쏘와 능형을 특수한 경우로 품어 버려 「이긴다」가 공허해지기 때문이다.

① 세 요인을 모두 켠 자리와 모두 끈 자리.
② 요인 하나씩만 바꿔 가며 — 무엇이 실제로 값을 하는가.
③ 이길 확률 (되풀이 가운데 엘라스틱넷이 둘 다 이긴 비율).

만드는 파일: docs/ch18/elastic_net/img/when_en_wins.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV   # noqa: E402

REP, NTE = 25, 3000
L1 = [0.1, 0.3, 0.5, 0.7, 0.9]          # 안쪽 alpha 만


def design(rng, n, p, grouped, nnz, blk=5, rho=0.9):
    if grouped:
        nb = p // blk
        g = rng.normal(size=(n, nb))
        X = np.repeat(g, blk, axis=1) * np.sqrt(rho) \
            + np.sqrt(1 - rho) * rng.normal(size=(n, nb * blk))
        X = X[:, :p]
    else:
        X = rng.normal(size=(n, p))
    X -= X.mean(0)
    X /= X.std(0) + 1e-12
    b = np.zeros(p)
    if grouped and nnz >= blk:
        b[:nnz] = 1.0                     # 집단을 통째로 켠다
    else:
        b[:nnz] = 1.0
    y = X @ b + rng.normal(size=n) * 2.0
    return X, y


def one(rng, n, p, grouped, nnz):
    X, y = design(rng, n + NTE, p, grouped, nnz)
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    ytr = ytr - ytr.mean()
    out = {}
    out["능형"] = np.mean((yte - yte.mean() - RidgeCV(
        alphas=np.logspace(-3, 5, 50)).fit(Xtr, ytr).predict(Xte)) ** 2)
    out["라쏘"] = np.mean((yte - yte.mean() - LassoCV(
        n_alphas=40, cv=5, max_iter=30_000,
        random_state=0).fit(Xtr, ytr).predict(Xte)) ** 2)
    out["엘라스틱넷"] = np.mean((yte - yte.mean() - ElasticNetCV(
        l1_ratio=L1, n_alphas=40, cv=5, max_iter=30_000,
        random_state=0).fit(Xtr, ytr).predict(Xte)) ** 2)
    return out


# 조건 셋: (중간 희소성, 집단 상관, p >= n)
CELLS = [
    ("셋 다 끔",       dict(n=200, p=20,  grouped=False, nnz=3)),
    ("희소성만",       dict(n=200, p=20,  grouped=False, nnz=8)),
    ("상관만",         dict(n=200, p=20,  grouped=True,  nnz=3)),
    ("차원만",         dict(n=100, p=200, grouped=False, nnz=3)),
    ("희소성+상관",    dict(n=200, p=20,  grouped=True,  nnz=8)),
    ("희소성+차원",    dict(n=100, p=200, grouped=False, nnz=40)),
    ("상관+차원",      dict(n=100, p=200, grouped=True,  nnz=5)),
    ("셋 다 켬",       dict(n=100, p=200, grouped=True,  nnz=40)),
]

RES = {}
for name, cfg in CELLS:
    rng = np.random.default_rng(11)
    rows = [one(rng, **cfg) for _ in range(REP)]
    r = np.array([[d["능형"], d["라쏘"], d["엘라스틱넷"]] for d in rows])
    best_other = r[:, :2].min(1)
    RES[name] = dict(
        med=r.mean(0),
        ratio=float(np.median(r[:, 2] / best_other)),
        winrate=float(np.mean(r[:, 2] < best_other)),
        cfg=cfg)

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"되풀이 {REP} 번, 시험자료 {NTE:,} 개. "
      "엘라스틱넷의 alpha 는 0.1-0.9 안쪽만 쓴다.")
print("\n① 조건 조합별 시험 MSE 평균")
print("   조합            n     p   집단   0아닌수   능형    라쏘   EN    "
      "EN/더나은쪽   EN 승률")
for name, _ in CELLS:
    d = RES[name]
    c = d["cfg"]
    print(f"   {name:<13} {c['n']:>4} {c['p']:>5}   "
          f"{'예' if c['grouped'] else '—':>3}   {c['nnz']:>5}   "
          f"{d['med'][0]:>5.2f}  {d['med'][1]:>5.2f}  {d['med'][2]:>5.2f}   "
          f"{d['ratio']:>9.3f}   {d['winrate']:>7.2f}")

off = RES["셋 다 끔"]
on = RES["셋 다 켬"]
print(f"\n② 셋을 다 끄면 EN 이 더 나은 쪽의 {off['ratio']:.3f} 배 "
      f"(승률 {off['winrate']:.2f}),")
print(f"   셋을 다 켜면 {on['ratio']:.3f} 배 (승률 {on['winrate']:.2f}) 다.")

singles = [("희소성만", "희소성"), ("상관만", "상관"), ("차원만", "차원")]
print("\n③ 요인 하나씩만 켰을 때 (기준은 「셋 다 끔」)")
print("   켠 요인     EN/더나은쪽   기준 대비 개선")
for key, lab in singles:
    d = RES[key]
    print(f"   {lab:<9}   {d['ratio']:>9.3f}   "
          f"{(off['ratio']-d['ratio'])/off['ratio']*100:>+13.1f} %")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_R, C_L, C_E = "#1f77b4", "#d62728", "#2ca02c"

names = [n for n, _ in CELLS]
xs = np.arange(len(names))
w = 0.26
for k, (lab, c) in enumerate((("능형", C_R), ("라쏘", C_L),
                              ("엘라스틱넷", C_E))):
    ax1.bar(xs + (k - 1) * w, [RES[n]["med"][k] for n in names], width=w,
            color=c, label=lab)
ax1.set_xticks(xs)
ax1.set_xticklabels(names, rotation=38, ha="right", fontsize=9)
ax1.set_ylabel("시험 MSE 평균")
ax1.set_title("① 조건 조합별 세 방법", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

rat = np.array([RES[n]["ratio"] for n in names])
cols = [C_E if v < 1 else "#999999" for v in rat]
ax2.barh(xs, rat - 1.0, left=1.0, color=cols, height=0.62)
ax2.axvline(1.0, color="#333333", lw=2.2, ls="--")
ax2.set_yticks(xs)
ax2.set_yticklabels(names, fontsize=9.5)
ax2.invert_yaxis()
ax2.set_xlabel("EN 의 MSE $\\div$ 능형·라쏘 가운데 더 나은 쪽")
ax2.set_title("② $1$ 보다 왼쪽이면 EN 이 이긴 것", fontsize=12)
ax2.grid(alpha=0.25, axis="x")

wr = np.array([RES[n]["winrate"] for n in names])
ax3.barh(xs, wr, color=[C_E if v > 0.5 else "#999999" for v in wr],
         height=0.62)
ax3.axvline(0.5, color="#333333", lw=2.2, ls="--")
ax3.set_yticks(xs)
ax3.set_yticklabels(names, fontsize=9.5)
ax3.invert_yaxis()
ax3.set_xlim(0, 1)
ax3.set_xlabel("되풀이 가운데 EN 이 둘 다 이긴 비율")
ax3.set_title("③ 얼마나 자주 이기는가", fontsize=12)
ax3.grid(alpha=0.25, axis="x")

fig.tight_layout()
out = "docs/ch18/elastic_net/img/when_en_wins.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
