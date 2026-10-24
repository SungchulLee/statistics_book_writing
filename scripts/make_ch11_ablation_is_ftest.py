r"""요즘 모형은 $p$ 값을 적지 않고 **절제 실험**(ablation)을 한다.

변수를 하나 빼고 다시 적합해 성능이 얼마나 떨어지는지를 본다. 그런데 선형회귀에서
그 「떨어진 몫」은 이 쪽의 $t$ 통계량과 **정확히 비례**한다.

$$F_j = t_j^2 = \Delta R^2_j \cdot \frac{n-p-1}{1-R^2}$$

같은 계산인데 한쪽에는 귀무분포가 붙어 있고 다른 쪽에는 없다. 없으면 어떻게 되는지를
잡음만 있는 자료로 보인다.

① 절제의 $\Delta R^2$ 와 $t^2$ 는 같은 수다.
② 잡음뿐인 자료에서도 절제 순위표는 만들어진다.
③ 그 $1$ 위를 귀무분포에 놓으면 아무 일도 아니다.

만드는 파일: docs/ch11/testing_coefficients/img/ablation_is_ftest.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N, P = 200, 30


def fit(X, y):
    D = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ b
    return b, r @ r


def table(X, y):
    """전체 적합의 t 통계량과, 변수마다 하나씩 빼서 잰 dR^2 를 함께 돌려준다."""
    n, p = X.shape
    D = np.column_stack([np.ones(n), X])
    b, rss = fit(X, y)
    s2 = rss / (n - p - 1)
    cov = s2 * np.linalg.inv(D.T @ D)
    t = b[1:] / np.sqrt(np.diag(cov)[1:])
    sst = ((y - y.mean()) ** 2).sum()
    d = np.array([(fit(np.delete(X, j, axis=1), y)[1] - rss) / sst
                  for j in range(p)])
    return t, d, 1 - rss / sst


# === ① 신호가 있는 자료 =============================================
Xs = RNG.standard_normal((N, P))
beta = np.zeros(P)
beta[:6] = [1.2, -0.9, 0.6, 0.4, -0.3, 0.2]
ys = Xs @ beta + RNG.standard_normal(N)
t_s, d_s, R2_s = table(Xs, ys)
const = (N - P - 1) / (1 - R2_s)
print(f"① 전체 R^2 = {R2_s:.3f}, 비례상수 (n-p-1)/(1-R^2) = {const:.2f}")
err = np.abs(d_s * const - t_s**2).max()
print(f"   max |dR^2 x 상수 - t^2| = {err:.3e}  (같은 수다)")

# === ②③ 잡음뿐인 자료 ===============================================
Xn = RNG.standard_normal((N, P))
yn = RNG.standard_normal(N)          # X 와 아무 관계도 없다
t_n, d_n, R2_n = table(Xn, yn)
order = np.argsort(-d_n)
p_n = 2 * stats.t.sf(np.abs(t_n), N - P - 1)
print(f"\n② 잡음뿐인 자료 — 전체 R^2 = {R2_n:.3f}")
print(f"{'순위':>4}{'변수':>6}{'dR^2':>10}{'|t|':>8}{'p':>8}")
for r, j in enumerate(order[:6], 1):
    print(f"{r:>4}{f'x{j}':>6}{d_n[j]:>10.4f}{abs(t_n[j]):>8.2f}{p_n[j]:>8.3f}")
print(f"   상위 다섯의 dR^2 합 = {d_n[order[:5]].sum():.3f}")

# 귀무분포: y 를 섞어 같은 절제를 되풀이한다
B = 2000
null_max, null_all = [], []
for _ in range(B):
    yp = RNG.permutation(yn)
    _, dp, _ = table(Xn, yp)
    null_max.append(dp.max())
    null_all.append(dp)
null_max = np.array(null_max)
null_all = np.concatenate(null_all)
obs_max = d_n[order[0]]
print(f"\n③ 관측된 1 위의 dR^2 = {obs_max:.4f}")
print(f"   섞어서 만든 1 위 분포의 평균 {null_max.mean():.4f}, "
      f"가운데 95% [{np.quantile(null_max, .025):.4f}, "
      f"{np.quantile(null_max, .975):.4f}]")
print(f"   관측값보다 큰 비율 = {np.mean(null_max >= obs_max):.3f}")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.scatter(d_s, t_s**2, s=55, color="#1f77b4", zorder=3,
            edgecolors="white", linewidths=0.8)
xs_ = np.linspace(0, d_s.max() * 1.05, 50)
ax1.plot(xs_, xs_ * const, color="#d62728", lw=2.0, ls="--",
         label=f"$t^2 = \\Delta R^2 \\times {const:.1f}$")
ax1.set_xlabel("절제의 $\\Delta R^2$ — 변수를 빼면 잃는 설명력")
ax1.set_ylabel("이 쪽의 $t^2$ (= 부분 $F$)")
ax1.set_title("① 둘은 같은 수다", fontsize=12)
ax1.legend(loc="upper left", fontsize=11)
_m, _e = f"{err:.1e}".split("e")      # 지수표기는 쪼개서 수식으로 조립한다
ax1.text(0.97, 0.06,
         f"둘의 최대 어긋남\n${_m} \\times 10^{{{int(_e)}}}$",
         transform=ax1.transAxes, ha="right", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax1.grid(alpha=0.25)

top = order[:12]
ax2.bar(range(len(top)), d_n[top], 0.65, color="#ff7f0e")
ax2.set_xticks(range(len(top)))
ax2.set_xticklabels([f"$x_{{{j}}}$" for j in top], fontsize=9)
ax2.set_xlabel("절제 순위 (위에서부터)")
ax2.set_ylabel("$\\Delta R^2$")
ax2.set_title("② 잡음뿐인 자료의 절제 순위표", fontsize=12)
ax2.text(0.5, 0.93,
         f"$y$ 는 어느 변수와도 관계가 없다\n상위 다섯의 합 "
         f"$\\Delta R^2 = {d_n[order[:5]].sum():.3f}$",
         transform=ax2.transAxes, ha="center", va="top", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f0", ec="#ff7f0e"))
ax2.set_ylim(0, d_n[order[0]] * 1.55)
ax2.grid(alpha=0.25, axis="y")

ax3.hist(null_max, bins=50, density=True, color="#9e9e9e", edgecolor="none",
         label="섞어서 만든 $1$ 위의 분포")
ax3.axvline(obs_max, color="#d62728", lw=2.4)
ax3.text(obs_max, ax3.get_ylim()[1] * 0.92,
         f"  관측된 $1$ 위\n  ${obs_max:.4f}$", fontsize=10.5, color="#d62728",
         va="top")
ax3.set_xlabel("$30$ 개 변수 가운데 가장 큰 $\\Delta R^2$")
ax3.set_ylabel("밀도")
ax3.set_title("③ 귀무분포에 놓으면 아무 일도 아니다", fontsize=12)
ax3.text(0.97, 0.55,
         f"이보다 큰 경우가\n섞은 자료의 "
         f"${np.mean(null_max >= obs_max) * 100:.0f}\\%$",
         transform=ax3.transAxes, ha="right", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax3.legend(loc="upper right", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch11/testing_coefficients/img/ablation_is_ftest.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
