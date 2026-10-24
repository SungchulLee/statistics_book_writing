r"""쿡의 거리를 신경망으로 옮기면 무엇이 깨지는가.

이 쪽의 영향 진단은 「이 점을 빼면 결과가 얼마나 바뀌는가」를 묻는다. 선형모형에서는
그 답이 닫힌 식으로 나온다. 신경망에서도 같은 일을 하려고 **영향함수**를 쓴다 —
$\mathcal I(z) = -H^{-1}\nabla_\theta \ell(z)$ 로 재학습 없이 어림하는 것이다.

① 볼록한 모형(로지스틱 회귀)에서는 그 어림이 잘 맞는다.
② 신경망에서는 **재야 할 양 자체가** 재학습의 잡음에 묻힌다.
③ 그래서 「영향이 큰 점 상위 $20$ 개」가 시드마다 달라진다.

만드는 파일: docs/ch13/diagnostics/img/influence_fragile.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer                 # noqa: E402
from sklearn.metrics import log_loss                            # noqa: E402
from sklearn.model_selection import train_test_split            # noqa: E402
from sklearn.neural_network import MLPClassifier                # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402

NTRAIN = 120
LAM = 0.05                 # 로지스틱의 L2 (헤세가 가역이어야 한다)
DEPTHS = (1, 2, 3, 4, 5)
WIDTH = 16
SEEDS = (0, 1, 2, 3, 4, 5)
TOPK = 20

X, y = load_breast_cancer(return_X_y=True)
Xtr0, Xte0, ytr, yte = train_test_split(
    X, y, train_size=NTRAIN, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr0)
Xtr0, Xte0 = sc.transform(Xtr0), sc.transform(Xte0)
N = len(ytr)
print(f"훈련 {N} 개, 시험 {len(yte)} 개, 특징 {Xtr0.shape[1]} 개")

# === ① 볼록한 모형 ===================================================
Xtr = np.c_[np.ones(N), Xtr0]
Xte = np.c_[np.ones(len(yte)), Xte0]


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logit(Xa, ya, iters=80):
    """뉴턴법 — L2 를 넣어 헤세를 가역으로 만든다."""
    th = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = sigm(Xa @ th)
        g = Xa.T @ (p - ya) / len(ya) + LAM * th
        w = p * (1 - p) + 1e-10
        H = (Xa * w[:, None]).T @ Xa / len(ya) + LAM * np.eye(Xa.shape[1])
        th -= np.linalg.solve(H, g)
    return th


def nll(Xa, ya, th):
    p = np.clip(sigm(Xa @ th), 1e-12, 1 - 1e-12)
    return float(-(ya * np.log(p) + (1 - ya) * np.log(1 - p)).mean())


th = fit_logit(Xtr, ytr)
p = sigm(Xtr @ th)
w = p * (1 - p) + 1e-10
Hinv = np.linalg.inv((Xtr * w[:, None]).T @ Xtr / N + LAM * np.eye(Xtr.shape[1]))
g_te = Xte.T @ (sigm(Xte @ th) - yte) / len(yte)
g_tr = Xtr * (sigm(Xtr @ th) - ytr)[:, None]
pred = (g_te @ Hinv @ g_tr.T) / N                     # 영향함수의 예측
L0 = nll(Xte, yte, th)
act = np.array([nll(Xte, yte, fit_logit(np.delete(Xtr, i, 0),
                                        np.delete(ytr, i))) - L0
                for i in range(N)])
r1 = float(np.corrcoef(pred, act)[0, 1])
ov1 = len(set(np.argsort(-act)[:TOPK]) & set(np.argsort(-pred)[:TOPK]))
print(f"\n① 로지스틱 회귀 — 재학습 없이 어림한 값과 실제로 {N} 번 재학습한 값")
print(f"   상관 {r1:.4f}, 상위 {TOPK} 개 가운데 {ov1} 개가 같다")
print(f"   실제 변화의 범위 [{act.min():.2e}, {act.max():.2e}]")

# === ②③ 신경망 ======================================================
print(f"\n②③ MLP (너비 {WIDTH}, 시드 {len(SEEDS)} 개, 깊이마다 "
      f"{(N + 1) * len(SEEDS)} 번 학습)")
print(f"{'깊이':>5}{'기준손실':>10}{'신호 SD':>11}{'잡음 SD':>11}"
      f"{'신호/잡음':>11}{'상위겹침':>10}")
sig_sd, noi_sd, overlap, base_l = [], [], [], []
for d in DEPTHS:
    h = (WIDTH,) * d
    delta, L_base = {}, {}
    for s in SEEDS:
        m = MLPClassifier(h, max_iter=600, random_state=s, alpha=1e-3)
        m.fit(Xtr0, ytr)
        L_base[s] = log_loss(yte, m.predict_proba(Xte0)[:, 1], labels=[0, 1])
        dd = np.empty(N)
        for i in range(N):
            mi = MLPClassifier(h, max_iter=600, random_state=s, alpha=1e-3)
            mi.fit(np.delete(Xtr0, i, 0), np.delete(ytr, i))
            dd[i] = log_loss(yte, mi.predict_proba(Xte0)[:, 1],
                             labels=[0, 1]) - L_base[s]
        delta[s] = dd
    # 신호 = 점을 바꿀 때의 흩어짐, 잡음 = 시드만 바꿀 때의 흩어짐
    sg = float(np.mean([delta[s].std() for s in SEEDS]))
    nz = float(np.std([L_base[s] for s in SEEDS]))
    ovs = [len(set(np.argsort(-delta[a])[:TOPK])
               & set(np.argsort(-delta[b])[:TOPK]))
           for ai, a in enumerate(SEEDS) for b in SEEDS[ai + 1:]]
    sig_sd.append(sg); noi_sd.append(nz)
    overlap.append(float(np.mean(ovs)))
    base_l.append(float(np.mean(list(L_base.values()))))
    print(f"{d:>5}{base_l[-1]:>10.4f}{sg:>11.5f}{nz:>11.5f}"
          f"{sg / nz:>11.2f}{overlap[-1]:>10.1f}")

sig_sd, noi_sd, overlap = map(np.array, (sig_sd, noi_sd, overlap))
chance = TOPK * TOPK / N
print(f"\n   우연히 겹칠 기대값 {chance:.1f}/{TOPK}")
print(f"   볼록한 모형에서는 재학습이 결정적이라 겹침이 {TOPK}/{TOPK} 이다")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.scatter(pred * 1e3, act * 1e3, s=26, alpha=0.75, color="#1f77b4",
            edgecolors="none")
lo = min(pred.min(), act.min()) * 1e3
hi = max(pred.max(), act.max()) * 1e3
ax1.plot([lo, hi], [lo, hi], color="#999999", lw=1.3, ls=":")
ax1.axhline(0, color="black", lw=1.0, ls="-", alpha=0.35)
ax1.axvline(0, color="black", lw=1.0, ls="-", alpha=0.35)
ax1.text(0.04, 0.95, f"상관 ${r1:.3f}$\n상위 ${TOPK}$ 개 중 ${ov1}$ 개 일치",
         transform=ax1.transAxes, va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax1.set_xlabel("영향함수의 예측 ($\\times 10^{-3}$)")
ax1.set_ylabel("실제로 빼고 다시 적합한 값 ($\\times 10^{-3}$)")
ax1.set_title("① 볼록하면 재학습이 필요 없다", fontsize=12)
ax1.grid(alpha=0.25)

xs = np.arange(len(DEPTHS))
ax2.bar(xs - 0.2, sig_sd * 1e3, 0.38, color="#1f77b4",
        label="한 점을 뺐을 때 (신호)")
ax2.bar(xs + 0.2, noi_sd * 1e3, 0.38, color="#d62728",
        label="시드만 바꿨을 때 (잡음)")
for i in range(len(DEPTHS)):
    ax2.text(i, max(sig_sd[i], noi_sd[i]) * 1e3 * 1.06,
             f"${sig_sd[i] / noi_sd[i]:.2f}$", ha="center", fontsize=10.5)
ax2.set_ylim(0, noi_sd.max() * 1e3 * 1.24)
ax2.set_xticks(xs)
ax2.set_xticklabels([f"${d}$" for d in DEPTHS])
ax2.set_xlabel("은닉층 수")
ax2.set_ylabel("시험손실의 흩어짐 ($\\times 10^{-3}$)")
ax2.set_title("② 신호가 잡음에 묻힌다 (수는 신호/잡음)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(DEPTHS, overlap, "o-", color="#1f77b4", lw=2.8, ms=9)
ax3.axhline(TOPK, color="#2ca02c", lw=2.0, ls="--",
            label=f"볼록한 모형 (${TOPK}/{TOPK}$)")
ax3.axhline(chance, color="#d62728", lw=2.0, ls=":",
            label=f"우연 수준 (${chance:.1f}/{TOPK}$)")
ax3.set_xticks(list(DEPTHS))
ax3.set_xticklabels([f"${d}$" for d in DEPTHS])
ax3.set_ylim(0, TOPK * 1.12)
ax3.set_xlabel("은닉층 수")
ax3.set_ylabel(f"시드가 다른 두 번 사이의 상위 ${TOPK}$ 개 겹침")
ax3.set_title("③ 「가장 영향이 큰 점」이 바뀐다", fontsize=12)
ax3.legend(loc="center right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/diagnostics/img/influence_fragile.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
