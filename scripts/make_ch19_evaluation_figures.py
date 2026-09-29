r"""19.3 분류 성능 평가 일곱 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch19/evaluation/img/confusion_two_ways.png    같은 표, 행으로 나누느냐 열로 나누느냐
  ch19/evaluation/img/metrics_vs_threshold.png  모든 지표는 문턱의 함수다
  ch19/evaluation/img/auc_as_ranking.png        AUC 는 순위를 맞힐 확률이다
  ch19/evaluation/img/pr_iso_f1.png             PR 곡선의 기준선과 F1 등고선
  ch19/evaluation/img/cost_threshold.png        비용이 문턱을 정한다
  ch19/evaluation/img/calibration_same_auc.png  같은 AUC, 다른 보정
  ch19/evaluation/img/imbalance_auc_brier.png   가중은 순위를 못 고치고 보정을 깬다

실행:  python3 scripts/make_ch19_evaluation_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             brier_score_loss)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch19/evaluation/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def loan_data():
    """19.3 절 여러 쪽이 함께 쓰는 연체율 19% 자료."""
    rng = np.random.default_rng(0)
    n = 4000
    score = rng.normal(0, 1, n)
    dti = rng.normal(0, 1, n)
    income = rng.normal(0, 1, n)
    X = np.column_stack([score, dti, income])
    z = -2.0 - 1.1 * score + 0.9 * dti - 0.5 * income
    y = (rng.random(n) < sigmoid(z)).astype(int)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=0, stratify=y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# =====================================================================
# 1. 같은 표, 두 방향의 나눗셈  (confusion_matrix.md)
# =====================================================================
def fig_confusion_two_ways():
    TN, FP, FN, TP = 14523, 8148, 8335, 14336
    M = np.array([[TN, FP], [FN, TP]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    specs = [
        ("행으로 나눈다: 실제 범주가 분모",
         M / M.sum(axis=1, keepdims=True),
         [["특이도", "위양성률"], ["위음성률", "재현율"]], BLUE),
        ("열로 나눈다: 예측 범주가 분모",
         M / M.sum(axis=0, keepdims=True),
         [["음성예측도", "허위경보율"], ["누락률", "정밀도"]], ORANGE),
    ]

    for ax, (title, R, names, col) in zip(axes, specs):
        for i in range(2):
            for j in range(2):
                shade = R[i, j]
                ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=col,
                                       alpha=0.13 + 0.55 * shade,
                                       edgecolor="white", lw=2.5))
                ax.text(j + 0.5, 1 - i + 0.62, "%s" % f"{int(M[i, j]):,}",
                        ha="center", va="center", color=INK, fontsize=13)
                ax.text(j + 0.5, 1 - i + 0.38, "%.4f" % R[i, j],
                        ha="center", va="center", color=col, fontsize=12,
                        fontweight="bold")
                ax.text(j + 0.5, 1 - i + 0.17, names[i][j],
                        ha="center", va="center", color=col, fontsize=9.5)

        ax.set_xlim(-0.72, 2.1)
        ax.set_ylim(-0.42, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 2.12, "상환으로 예측", ha="center", color=INK, fontsize=10)
        ax.text(1.5, 2.12, "연체로 예측", ha="center", color=INK, fontsize=10)
        ax.text(-0.06, 1.5, "실제 상환", ha="right", va="center", color=INK,
                fontsize=10)
        ax.text(-0.06, 0.5, "실제 연체", ha="right", va="center", color=INK,
                fontsize=10)
        ax.set_title(title, color=col, fontsize=11.5)

    # 나눗셈 방향 표시
    ax = axes[0]
    for yv in (1.03, 0.03):
        ax.annotate("", xy=(2.06, yv), xytext=(-0.03, yv),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5,
                                    alpha=0.5))
    ax.text(1.0, -0.26, "각 행의 합이 $1$", ha="center", color=BLUE,
            fontsize=10)
    ax = axes[1]
    for xv in (0.04, 1.04):
        ax.annotate("", xy=(xv, -0.06), xytext=(xv, 2.03),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                    alpha=0.5))
    ax.text(1.0, -0.26, "각 열의 합이 $1$", ha="center", color=ORANGE,
            fontsize=10)

    fig.suptitle("같은 혼동행렬, 나누는 방향만 다르다  (대출 45,342 건, 정확도 $0.6365$)",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, "confusion_two_ways.png")


# =====================================================================
# 2. 모든 지표는 문턱의 함수다  (metrics.md)
# =====================================================================
def fig_metrics_vs_threshold():
    X_tr, y_tr, X_va, y_va, X_te, y_te = loan_data()
    m = LogisticRegression().fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    taus = np.linspace(0.01, 0.99, 197)

    acc, prec, rec, spec, f1 = [], [], [], [], []
    for t in taus:
        yh = (p >= t).astype(int)
        tp = int(((yh == 1) & (y_te == 1)).sum())
        fp = int(((yh == 1) & (y_te == 0)).sum())
        fn = int(((yh == 0) & (y_te == 1)).sum())
        tn = int(((yh == 0) & (y_te == 0)).sum())
        acc.append((tp + tn) / len(y_te))
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    acc, prec, rec, spec, f1 = map(np.array, (acc, prec, rec, spec, f1))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    ax.plot(taus, rec, color=BLUE, lw=2.3, label="재현율 (민감도)")
    ax.plot(taus, prec, color=ORANGE, lw=2.3, label="정밀도")
    ax.plot(taus, spec, color=GREEN, lw=2.0, ls="--", label="특이도")
    ax.plot(taus, f1, color=PURPLE, lw=2.3, label="$F_1$")
    ax.plot(taus, acc, color=INK, lw=2.6, label="정확도")
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=":")
    ax.text(0.515, 0.03, r"$\tau=0.5$", color=MUTED, fontsize=9.5)
    ax.axhline(1 - y_te.mean(), color=RED, lw=1.2, ls=":")
    ax.text(0.02, 1.035,
            "빨간 점선: 전부 음성이라 답해도 정확도 $%.3f$" % (1 - y_te.mean()),
            color=RED, fontsize=9.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.32)
    ax.set_xlabel(r"문턱 $\tau$", color=INK, fontsize=10)
    ax.set_ylabel("지표값", color=INK, fontsize=10)
    ax.set_title("지표 다섯 개는 하나의 문턱 스윕에서 나온다",
                 color=INK, fontsize=11)
    ax.legend(loc="upper center", fontsize=8.8, frameon=False, ncol=3,
              bbox_to_anchor=(0.5, 1.02))
    tidy(ax)

    i5 = int(np.argmin(np.abs(taus - 0.5)))
    ibest = int(np.nanargmax(f1))
    print("  tau=0.5 : acc %.3f prec %.3f rec %.3f spec %.3f f1 %.3f"
          % (acc[i5], prec[i5], rec[i5], spec[i5], f1[i5]))
    print("  best F1 : tau %.3f f1 %.3f prec %.3f rec %.3f acc %.3f"
          % (taus[ibest], f1[ibest], prec[ibest], rec[ibest], acc[ibest]))

    ax = axes[1]
    width = 0.34
    labels = ["정확도", "정밀도", "재현율", "특이도", "$F_1$"]
    v5 = [acc[i5], prec[i5], rec[i5], spec[i5], f1[i5]]
    vb = [acc[ibest], prec[ibest], rec[ibest], spec[ibest], f1[ibest]]
    xs = np.arange(5)
    ax.bar(xs - width / 2, v5, width, color=BLUE_F, edgecolor=BLUE,
           label=r"$\tau=0.5$")
    ax.bar(xs + width / 2, vb, width, color=ORANGE_F, edgecolor=ORANGE,
           label=r"$\tau=%.2f$ ($F_1$ 최대)" % taus[ibest])
    for k in range(5):
        ax.text(xs[k] - width / 2, v5[k] + 0.02, "%.3f" % v5[k], ha="center",
                color=BLUE, fontsize=9)
        ax.text(xs[k] + width / 2, vb[k] + 0.02, "%.3f" % vb[k], ha="center",
                color=ORANGE, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("지표값", color=INK, fontsize=10)
    ax.set_title("문턱을 바꾸면 무엇이 오르고 무엇이 내리나",
                 color=INK, fontsize=11)
    ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=2)
    tidy(ax)

    fig.tight_layout()
    save(fig, "metrics_vs_threshold.png")


# =====================================================================
# 3. AUC 는 순위를 맞힐 확률이다  (roc_auc.md)
# =====================================================================
def fig_auc_as_ranking():
    rng = np.random.default_rng(19)
    n_pos = n_neg = 2000
    d = np.sqrt(2) * 0.5          # AUC = Phi(d/sqrt(2)) = 0.6915
    s_neg = rng.normal(0.0, 1.0, n_neg)
    s_pos = rng.normal(d, 1.0, n_pos)
    s = np.concatenate([s_neg, s_pos])
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    auc = roc_auc_score(y, s)
    # 모든 양성-음성 쌍을 직접 세어 본다
    wins = (s_pos[:, None] > s_neg[None, :]).sum()
    ties = (s_pos[:, None] == s_neg[None, :]).sum()
    pair = (wins + 0.5 * ties) / (n_pos * n_neg)
    fpr, tpr, thr = roc_curve(y, s)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    bins = np.linspace(-4.2, 4.6, 55)
    ax.hist(s_neg, bins=bins, color=BLUE, alpha=0.45, label="음성 (상환)")
    ax.hist(s_pos, bins=bins, color=ORANGE, alpha=0.45, label="양성 (연체)")
    cut = 1.0
    ax.axvline(cut, color=INK, lw=1.4, ls="--")
    ax.text(cut + 0.12, 168, "문턱 하나는\n이 선 하나일 뿐", color=INK,
            fontsize=9.3, va="top")
    ax.set_xlim(-4.2, 4.6)
    ax.set_ylim(0, 205)
    ax.set_xlabel("모형이 준 점수 (로그오즈)", color=INK, fontsize=10)
    ax.set_ylabel("도수", color=INK, fontsize=10)
    ax.set_title("두 분포가 겹치는 만큼 순위가 섞인다", color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    tidy(ax)

    ax = axes[1]
    ax.fill_between(fpr, 0, tpr, color=BLUE_F, alpha=0.85)
    ax.plot(fpr, tpr, color=BLUE, lw=2.4)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls=":")
    i = int(np.argmin(np.abs(thr - cut)))
    ax.plot([fpr[i]], [tpr[i]], "o", color=INK, ms=8)
    ax.annotate("왼쪽 그림의 문턱", xy=(fpr[i], tpr[i]),
                xytext=(0.40, 0.42), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.text(0.46, 0.23, "곡선 아래 면적\nAUC $= %.4f$" % auc, color=BLUE,
            fontsize=11.5, ha="left")
    ax.text(0.46, 0.10,
            "양성이 음성보다 높은\n쌍의 비율 $= %.4f$" % pair,
            color=ORANGE, fontsize=11.5, ha="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("거짓양성률 FPR", color=INK, fontsize=10)
    ax.set_ylabel("참양성률 TPR", color=INK, fontsize=10)
    ax.set_title("같은 수를 두 가지 방법으로 센 것", color=INK, fontsize=11)
    tidy(ax)

    fig.tight_layout()
    save(fig, "auc_as_ranking.png")
    print("  AUC %.6f  pairwise %.6f  (400만 쌍)" % (auc, pair))


# =====================================================================
# 4. PR 곡선의 기준선과 F1 등고선  (precision_recall.md)
# =====================================================================
def fig_pr_iso_f1():
    X_tr, y_tr, X_va, y_va, X_te, y_te = loan_data()
    m = LogisticRegression().fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_te, p)
    base = y_te.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    for f in (0.2, 0.3, 0.4, 0.5, 0.6):
        r = np.linspace(f / (2 - f) + 1e-4, 1.0, 300)
        pr = f * r / (2 * r - f)
        ok = (pr <= 1.02) & (pr > 0)
        ax.plot(r[ok], pr[ok], color=MUTED, lw=0.9, ls=":")
        ax.text(0.995, min(f * 1.0 / (2 - f), 0.98) + 0.004,
                "$F_1=%.1f$" % f, color=MUTED, fontsize=8.5, ha="right",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.plot(rec, prec, color=BLUE, lw=2.4, label="정밀도-재현율 곡선")
    ax.axhline(base, color=RED, lw=1.3, ls="--")
    ax.text(0.03, base + 0.025, "기준선 = 유병률 $%.3f$" % base, color=RED,
            fontsize=9.5)

    f1s = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    ib = int(np.argmax(f1s))
    ax.plot([rec[ib]], [prec[ib]], "o", color=PURPLE, ms=8)
    ax.annotate("$F_1$ 최대 $%.3f$\n(정밀도 $%.3f$, 재현율 $%.3f$)"
                % (f1s[ib], prec[ib], rec[ib]),
                xy=(rec[ib], prec[ib]), xytext=(0.36, 0.86),
                color=PURPLE, fontsize=9.3,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("재현율", color=INK, fontsize=10)
    ax.set_ylabel("정밀도", color=INK, fontsize=10)
    ax.set_title("PR 곡선은 $F_1$ 등고선 위를 지나간다", color=INK, fontsize=11)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    tidy(ax)

    ax = axes[1]
    pp = np.linspace(0.001, 0.999, 500)
    rr = 1.0 - pp
    arith = (pp + rr) / 2
    harm = 2 * pp * rr / (pp + rr)
    ax.plot(pp, arith, color=ORANGE, lw=2.2, ls="--", label="산술평균")
    ax.plot(pp, harm, color=BLUE, lw=2.4, label="조화평균 $F_1$")
    ax.fill_between(pp, harm, arith, color=BLUE_F, alpha=0.8)
    for pv in (0.5, 0.8, 0.95):
        rv = 1 - pv
        ax.plot([pv], [2 * pv * rv / (pv + rv)], "o", color=PURPLE, ms=6)
    ax.annotate(r"$P=0.95,\ R=0.05$ 이면 $F_1=0.095$",
                xy=(0.95, 0.095), xytext=(0.52, 0.20), color=PURPLE,
                fontsize=9.3, ha="left",
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.annotate(r"$P=R=0.5$ 이면 $F_1=0.5$", xy=(0.5, 0.5),
                xytext=(0.13, 0.61), color=PURPLE, fontsize=9.3,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("정밀도 $P$  (재현율은 $R=1-P$ 로 고정)", color=INK,
                  fontsize=10)
    ax.set_ylabel("두 평균값", color=INK, fontsize=10)
    ax.set_title("합이 같아도 치우치면 $F_1$ 은 무너진다", color=INK,
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "pr_iso_f1.png")
    print("  PR: base %.3f  bestF1 %.3f at P %.3f R %.3f (tau %.3f)"
          % (base, f1s[ib], prec[ib], rec[ib], thr[min(ib, len(thr) - 1)]))


# =====================================================================
# 5. 비용이 문턱을 정한다  (threshold_tuning.md)
# =====================================================================
def fig_cost_threshold():
    X_tr, y_tr, X_va, y_va, X_te, y_te = loan_data()
    m = LogisticRegression().fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    taus = np.linspace(0.005, 0.995, 199)

    def cost(c_fp, c_fn):
        out = []
        for t in taus:
            yh = p >= t
            fp = int((yh & (y_te == 0)).sum())
            fn = int((~yh & (y_te == 1)).sum())
            out.append(c_fp * fp + c_fn * fn)
        return np.array(out, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    specs = [(50, 1000, RED, "놓친 연체 1,000달러 · 허위경보 50달러"),
             (200, 400, BLUE, "놓친 연체 400달러 · 허위경보 200달러"),
             (500, 500, GREEN, "두 오류가 같은 비용")]
    for c_fp, c_fn, col, lab in specs:
        c = cost(c_fp, c_fn)
        ax.plot(taus, c / 1000.0, color=col, lw=2.3, label=lab)
        tstar = c_fp / (c_fp + c_fn)
        i = int(np.argmin(np.abs(taus - tstar)))
        ax.plot([tstar], [c[i] / 1000.0], "o", color=col, ms=8, zorder=5)
        i5 = int(np.argmin(np.abs(taus - 0.5)))
        print("  cfp %d cfn %d : t*=%.3f cost %.0f | at 0.5 cost %.0f"
              % (c_fp, c_fn, tstar, c[i], c[i5]))

    ax.axvline(0.5, color=MUTED, lw=1.2, ls=":")
    ax.text(0.515, 4, r"관행적인 $\tau=0.5$", color=MUTED, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 205)
    ax.set_xlabel(r"문턱 $\tau$", color=INK, fontsize=10)
    ax.set_ylabel("검정자료 1,000건의 총비용 (천 달러)", color=INK,
                  fontsize=10)
    ax.set_title("비용곡선의 바닥은 $t^*=C_{FP}/(C_{FP}+C_{FN})$",
                 color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)
    tidy(ax)

    ax = axes[1]
    ratios = np.logspace(-1.3, 1.6, 200)      # C_FN / C_FP
    tstar = 1.0 / (1.0 + ratios)
    emp = []
    for r in ratios:
        c = cost(1.0, r)
        emp.append(taus[int(np.argmin(c))])
    ax.plot(ratios, tstar, color=BLUE, lw=2.4,
            label=r"이론값 $t^*=1/(1+C_{FN}/C_{FP})$")
    ax.plot(ratios, emp, color=ORANGE, lw=1.6, alpha=0.85,
            label="검정자료에서 실제로 비용이 최소인 문턱")
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":")
    ax.plot([20], [1 / 21], "o", color=RED, ms=8)
    ax.annotate(r"$C_{FN}/C_{FP}=20$ 이면 $t^*=0.048$",
                xy=(20, 1 / 21), xytext=(0.062, 0.12), color=RED, fontsize=9.3,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlim(0.05, 40)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("비용비 $C_{FN}/C_{FP}$", color=INK, fontsize=10)
    ax.set_ylabel("최적 문턱", color=INK, fontsize=10)
    ax.set_title("비용비 하나가 문턱을 결정한다", color=INK, fontsize=11)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "cost_threshold.png")


# =====================================================================
# 6. 같은 AUC, 다른 보정  (calibration.md)
# =====================================================================
def reliability(p, y, G=10):
    q = np.quantile(p, np.linspace(0, 1, G + 1))
    q[0] -= 1e-9
    idx = np.digitize(p, q[1:-1])
    pbar = np.array([p[idx == g].mean() for g in range(G)])
    ybar = np.array([y[idx == g].mean() for g in range(G)])
    size = np.array([int((idx == g).sum()) for g in range(G)])
    return pbar, ybar, size


def fig_calibration_same_auc():
    X_tr, y_tr, X_va, y_va, X_te, y_te = loan_data()
    m = LogisticRegression().fit(X_tr, y_tr)
    p_ok = m.predict_proba(X_te)[:, 1]
    p_over = sigmoid(logit(p_ok) + 1.8)       # 순위는 그대로, 값만 부풀린다

    auc_ok = roc_auc_score(y_te, p_ok)
    auc_ov = roc_auc_score(y_te, p_over)
    bs_ok = brier_score_loss(y_te, p_ok)
    bs_ov = brier_score_loss(y_te, p_over)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=":",
            label="완벽한 보정")
    for p, col, lab in ((p_ok, BLUE, "원래 모형"),
                        (p_over, ORANGE, "로그오즈에 $1.8$ 을 더한 모형")):
        pbar, ybar, size = reliability(p, y_te)
        ax.plot(pbar, ybar, "o-", color=col, lw=2.2, ms=6, label=lab)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"구간 평균 예측확률 $\bar p_g$", color=INK, fontsize=10)
    ax.set_ylabel(r"구간 관측 비율 $\bar y_g$", color=INK, fontsize=10)
    ax.set_title("신뢰도 그림: 대각선에서 얼마나 벗어나는가",
                 color=INK, fontsize=11)
    ax.text(0.06, 0.72, "대각선 아래 = 과대예측\n(모형이 지나치게 확신함)",
            color=ORANGE, fontsize=9.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    tidy(ax)

    ax = axes[1]
    labels = ["AUC", "브라이어 점수"]
    xs = np.arange(2)
    w = 0.34
    v1 = [auc_ok, bs_ok]
    v2 = [auc_ov, bs_ov]
    ax.bar(xs - w / 2, v1, w, color=BLUE_F, edgecolor=BLUE, label="원래 모형")
    ax.bar(xs + w / 2, v2, w, color=ORANGE_F, edgecolor=ORANGE,
           label="로그오즈에 $1.8$ 을 더한 모형")
    for k in range(2):
        ax.text(xs[k] - w / 2, v1[k] + 0.015, "%.4f" % v1[k], ha="center",
                color=BLUE, fontsize=10)
        ax.text(xs[k] + w / 2, v2[k] + 0.015, "%.4f" % v2[k], ha="center",
                color=ORANGE, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("순위는 한 글자도 안 바뀌고 확률만 망가졌다",
                 color=INK, fontsize=11)
    ax.text(0.5, 0.62, "단조변환이라 AUC 는\n소수점 여섯 자리까지 같다",
            ha="center", color=INK, fontsize=9.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "calibration_same_auc.png")
    print("  calib: AUC %.6f vs %.6f | Brier %.4f vs %.4f"
          % (auc_ok, auc_ov, bs_ok, bs_ov))


# =====================================================================
# 7. 가중은 순위를 못 고치고 보정을 깬다  (imbalanced_data.md)
# =====================================================================
def fig_imbalance_auc_brier():
    X_tr, y_tr, X_va, y_va, X_te, y_te = loan_data()
    m0 = LogisticRegression().fit(X_tr, y_tr)
    w = np.where(y_tr == 1, 5.3, 1.0)
    mw = LogisticRegression(class_weight="balanced").fit(
        X_tr, y_tr, sample_weight=w)
    p0 = m0.predict_proba(X_va)[:, 1]
    pw = mw.predict_proba(X_va)[:, 1]

    auc0, aucw = roc_auc_score(y_va, p0), roc_auc_score(y_va, pw)
    bs0, bsw = brier_score_loss(y_va, p0), brier_score_loss(y_va, pw)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    ax = axes[0]
    f0, t0, _ = roc_curve(y_va, p0)
    fw, tw, _ = roc_curve(y_va, pw)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls=":")
    ax.plot(f0, t0, color=BLUE, lw=3.4, alpha=0.9,
            label="가중 없음  AUC $=%.4f$" % auc0)
    ax.plot(fw, tw, color=ORANGE, lw=1.6, ls="--",
            label="가중 적용  AUC $=%.4f$" % aucw)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("거짓양성률 FPR", color=INK, fontsize=10)
    ax.set_ylabel("참양성률 TPR", color=INK, fontsize=10)
    ax.set_title("두 곡선이 사실상 포개진다", color=INK, fontsize=11)
    ax.text(0.30, 0.30, "가중을 줘도\n순위는 달라지지 않는다", color=INK,
            fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    tidy(ax)

    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    ax.hist(p0, bins=bins, color=BLUE, alpha=0.5, label="가중 없음")
    ax.hist(pw, bins=bins, color=ORANGE, alpha=0.5, label="가중 적용")
    ax.axvline(y_va.mean(), color=RED, lw=1.6, ls="--")
    ax.text(y_va.mean() + 0.02, 190, "실제 연체율\n$%.3f$" % y_va.mean(),
            color=RED, fontsize=9.5, va="top")
    ax.plot([p0.mean()], [12], "v", color=BLUE, ms=10)
    ax.plot([pw.mean()], [12], "v", color=ORANGE, ms=10)
    ax.text(p0.mean() - 0.015, 26, "평균 $%.3f$" % p0.mean(), color=BLUE,
            fontsize=9.3, ha="right")
    ax.text(pw.mean(), 26, "평균 $%.3f$" % pw.mean(), color=ORANGE,
            fontsize=9.3, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 230)
    ax.set_xlabel("예측확률", color=INK, fontsize=10)
    ax.set_ylabel("도수", color=INK, fontsize=10)
    ax.set_title("브라이어 점수 $%.4f \\to %.4f$" % (bs0, bsw),
                 color=INK, fontsize=11)
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "imbalance_auc_brier.png")
    print("  imbalance: AUC %.4f vs %.4f | Brier %.4f vs %.4f | mean p %.3f vs %.3f | actual %.3f"
          % (auc0, aucw, bs0, bsw, p0.mean(), pw.mean(), y_va.mean()))


if __name__ == "__main__":
    fig_confusion_two_ways()
    fig_metrics_vs_threshold()
    fig_auc_as_ranking()
    fig_pr_iso_f1()
    fig_cost_threshold()
    fig_calibration_same_auc()
    fig_imbalance_auc_brier()
