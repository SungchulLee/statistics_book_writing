import matplotlib
matplotlib.use('Agg')
import io,contextlib
_buf=io.StringIO()
with contextlib.redirect_stdout(_buf):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # 세 집단의 참 평균을 10, 12, 15 로 두고 표준편차는 모두 3.5 로 맞췄다.
    # A와 B의 차이는 표준편차보다 작고 A와 C의 차이는 그보다 크다.
    # Tukey 가 어느 쌍을 갈라내고 어느 쌍을 갈라내지 못하는지 보게 된다.
    rng = np.random.default_rng(42)
    n = 15
    df = pd.DataFrame({
        "response": np.concatenate([
            rng.normal(10.0, 3.5, n),
            rng.normal(12.0, 3.5, n),
            rng.normal(15.0, 3.5, n),
        ]),
        "group": ["A"] * n + ["B"] * n + ["C"] * n,
    })
    
    # Tukey HSD 는 쌍 세 개를 한꺼번에 견주면서 전체 오류율을 0.05 로 묶는다.
    # 쌍마다 t-검정을 따로 하면 이 통제가 무너진다.
    print(pairwise_tukeyhsd(endog=df["response"], groups=df["group"], alpha=0.05))

from scipy import stats

k, sigma = 3, 3.5
nu = k * n - k                      # 42
delta = 2.0                         # A 와 B 의 참 평균차
se_true = sigma * np.sqrt(2 / n)    # 차이의 '참' 표준오차
ncp = delta / se_true
print(f"참 표준오차 = {se_true:.4f},  비중심모수 delta/SE = {ncp:.4f}")

thr = {
    "보정 없음": stats.t.ppf(0.975, nu),
    "Tukey": stats.studentized_range.ppf(0.95, k, nu) / np.sqrt(2),
    "Bonferroni": stats.t.ppf(1 - 0.05 / 6, nu),
    "Scheffe": np.sqrt((k - 1) * stats.f(k - 1, nu).ppf(0.95)),
}

# 모의실험으로 확인한다 (통계량을 직접 만들어 문턱과 견준다).
B = 200_000
sim = np.random.default_rng(7)
mu = np.array([10.0, 12.0, 15.0])
Y = sim.normal(mu[None, :, None], sigma, size=(B, k, n))
gm = Y.mean(axis=2)
MSW = ((Y - gm[:, :, None]) ** 2).sum(axis=(1, 2)) / nu
T_AB = (gm[:, 1] - gm[:, 0]) / np.sqrt(MSW * 2 / n)

print(f"\n{'방법':<12}{'문턱(SE 단위)':>14}{'검정력(이론)':>14}{'검정력(모의)':>14}")
for name, c in thr.items():
    th = stats.nct(nu, ncp).sf(c) + stats.nct(nu, ncp).cdf(-c)
    print(f"{name:<12}{c:>14.4f}{th:>14.4f}{np.mean(np.abs(T_AB) > c):>14.4f}")

# Tukey 로 A-B 를 80% 확률로 잡으려면 집단당 몇 개가 필요한가.
print(f"\n{'n':>5}{'nu':>6}{'Tukey 문턱':>12}{'검정력':>10}")
for m in (15, 30, 50, 60, 63, 70):
    nu_m = k * m - k
    c = stats.studentized_range.ppf(0.95, k, nu_m) / np.sqrt(2)
    nc = delta / (sigma * np.sqrt(2 / m))
    pw = stats.nct(nu_m, nc).sf(c) + stats.nct(nu_m, nc).cdf(-c)
    print(f"{m:>5}{nu_m:>6}{c:>12.4f}{pw:>10.4f}")

print(f"\n이 표본의 MSW = 8.088 로 참 분산 sigma^2 = {sigma ** 2:.3f} 보다 작다")
