r"""「어느 모형이 제일 좋은가」를 분산분석으로 물으면 답이 달라진다.

유방암 자료(sklearn, $569 \times 30$)에서 훈련 $80$ 개만 쓰고, 세 가지를 모두 바꾸어
가며 시험 정확도를 잰다 — **모형** $4$ 가지, **전처리** $4$ 가지, **분할 시드**
$10$ 가지. 전부 $160$ 번 적합한다.

① 전체 변동을 항마다 쪼갠다. 모형이 가장 크지만 혼자가 아니다.
② 모형별 분포가 서로 크게 겹친다.
③ 전처리를 바꾸면 $1$ 위가 바뀐다 — 상호작용이 순위를 옮겨 놓는다.

만드는 파일: docs/ch12/practical_applications/img/variance_components.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer                     # noqa: E402
from sklearn.ensemble import (GradientBoostingClassifier,           # noqa: E402
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.model_selection import train_test_split                # noqa: E402
from sklearn.pipeline import make_pipeline                          # noqa: E402
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,  # noqa: E402
                                   StandardScaler)
from sklearn.svm import SVC                                         # noqa: E402

NTRAIN, NSEED = 80, 10

MODELS = {
    "로지스틱": lambda: LogisticRegression(max_iter=5000),
    "SVM(RBF)": lambda: SVC(),
    "랜덤숲": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "부스팅": lambda: GradientBoostingClassifier(random_state=0),
}
PREP = {
    "없음": lambda: None,
    "표준화": lambda: StandardScaler(),
    "최소최대": lambda: MinMaxScaler(),
    "분위수변환": lambda: QuantileTransformer(
        n_quantiles=50, output_distribution="normal", random_state=0),
}

X, y = load_breast_cancer(return_X_y=True)
A, B, C = len(MODELS), len(PREP), NSEED
acc = np.zeros((A, B, C))
for s in range(C):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, train_size=NTRAIN, random_state=s, stratify=y)
    for i, mf in enumerate(MODELS.values()):
        for j, pf in enumerate(PREP.values()):
            p = pf()
            est = mf() if p is None else make_pipeline(p, mf())
            acc[i, j, s] = est.fit(Xtr, ytr).score(Xte, yte)

print(f"적합 {A * B * C} 번, 전체 평균 {acc.mean():.4f}, "
      f"범위 [{acc.min():.4f}, {acc.max():.4f}]")
print(f"\n모형 x 전처리 평균 (시드 {C} 개)")
print(f"{'':<10}" + "".join(f"{p:>12}" for p in PREP))
for i, m in enumerate(MODELS):
    print(f"{m:<10}" + "".join(f"{acc[i, j].mean():>12.4f}" for j in range(B)))

# === ① 전체 변동의 분해 ==============================================
g = acc.mean()
M, P, S = acc.mean(axis=(1, 2)), acc.mean(axis=(0, 2)), acc.mean(axis=(0, 1))
MP, MS, PS = acc.mean(axis=2), acc.mean(axis=1), acc.mean(axis=0)
ss = {
    "모형": B * C * ((M - g) ** 2).sum(),
    "모형 $\\times$ 전처리": C * ((MP - M[:, None] - P[None, :] + g) ** 2).sum(),
    "전처리": A * C * ((P - g) ** 2).sum(),
    "시드": A * B * ((S - g) ** 2).sum(),
    "모형 $\\times$ 시드": B * ((MS - M[:, None] - S[None, :] + g) ** 2).sum(),
    "전처리 $\\times$ 시드": A * ((PS - P[:, None] - S[None, :] + g) ** 2).sum(),
}
tot = ((acc - g) ** 2).sum()
ss["나머지"] = tot - sum(ss.values())
print(f"\n① 전체 제곱합 {tot:.6f} (전체 표준편차 {acc.std():.4f})")
for k, v in ss.items():
    print(f"   {k.replace(chr(92) + 'times', 'x').replace('$', ''):<22}"
          f"{100 * v / tot:>7.1f}%")
inc = sum(v for k, v in ss.items() if "전처리" in k or "시드" in k)
print(f"   -> 모형이 아닌 선택이 모두 합쳐 {100 * inc / tot:.1f}%")

# === ② 겹침 ==========================================================
print(f"\n② 모형별 (전처리·시드 {B * C} 조합)")
for i, m in enumerate(MODELS):
    f = acc[i].ravel()
    print(f"   {m:<10} 평균 {f.mean():.4f}  범위 [{f.min():.4f}, {f.max():.4f}]")

# === ③ 1 위가 바뀐다 =================================================
print(f"\n③ 전처리마다의 1 위")
win = [int(np.argmax(MP[:, j])) for j in range(B)]
for j, p in enumerate(PREP):
    print(f"   {p:<10} -> {list(MODELS)[win[j]]}  ({MP[win[j], j]:.4f})")
print(f"   서로 다른 1 위가 {len(set(win))} 가지")
print(f"\n   전처리·시드 {B * C} 조합 가운데 1 위를 차지한 횟수")
for i, m in enumerate(MODELS):
    n = int((np.argmax(acc, axis=0) == i).sum())
    print(f"   {m:<10}{n:>3}/{B * C}")
isvm = list(MODELS).index("SVM(RBF)")
ignb = list(MODELS).index("부스팅")
print(f"\n   SVM 은 전처리에 {100 * (MP[isvm].max() - MP[isvm].min()):.1f} 점 흔들리고")
print(f"   부스팅은 {100 * (MP[ignb].max() - MP[ignb].min()):.1f} 점 흔들린다")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

names = list(ss)
vals = np.array([100 * ss[k] / tot for k in names])
cols = ["#1f77b4" if n == "모형" else
        "#999999" if n == "나머지" else "#d62728" for n in names]
ax1.barh(np.arange(len(names))[::-1], vals, 0.66, color=cols)
for i, v in enumerate(vals):
    ax1.text(v + 0.7, len(names) - 1 - i, f"${v:.1f}\\%$", va="center",
             fontsize=10.5)
ax1.set_yticks(np.arange(len(names))[::-1])
ax1.set_yticklabels(names, fontsize=10)
ax1.set_xlim(0, max(vals) * 1.22)
ax1.set_xlabel("전체 변동에서 차지하는 몫")
ax1.set_title("① 모형이 전부가 아니다", fontsize=12)
ax1.grid(alpha=0.25, axis="x")

parts = [acc[i].ravel() for i in range(A)]
bp = ax2.boxplot(parts, widths=0.5, patch_artist=True, showfliers=False)
for patch, c in zip(bp["boxes"], COL):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
for med in bp["medians"]:
    med.set_color("black")
for i, f in enumerate(parts):
    ax2.scatter(i + 1 + np.linspace(-0.17, 0.17, len(f)), f,
                s=11, color=COL[i], alpha=0.7, edgecolors="none", zorder=3)
ax2.set_xticks(np.arange(1, A + 1))
ax2.set_xticklabels(list(MODELS), fontsize=10)
ax2.set_ylabel("시험 정확도")
ax2.set_title(f"② 한 모형 안에서 {B * C} 가지", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

xs = np.arange(B)
for i, m in enumerate(MODELS):
    ax3.plot(xs, MP[i], "o-", color=COL[i], lw=2.5, ms=8, label=m)
for j in range(B):
    ax3.plot(j, MP[win[j], j], "*", color="black", ms=17, zorder=5)
ax3.text(0.26, 0.33, "★ 그 전처리에서의 $1$ 위", transform=ax3.transAxes,
         fontsize=10.5, va="center")
ax3.set_xticks(xs)
ax3.set_xticklabels(list(PREP), fontsize=10)
ax3.set_xlabel("전처리 — 「본질이 아닌」 선택")
ax3.set_ylabel("평균 시험 정확도")
ax3.set_title("③ 1 위가 세 번 바뀐다", fontsize=12)
ax3.legend(loc="lower right", fontsize=9.5, ncol=2)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/practical_applications/img/variance_components.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
