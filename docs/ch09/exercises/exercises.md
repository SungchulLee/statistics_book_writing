# 연습문제

!!! note "Welch 자유도 공식에 관한 정정"
    원 강의노트의 여러 코드 조각이 Welch-Satterthwaite 자유도를 다음처럼 계산했다.

    ```python
    bottom = (s_1**2/n_1)**2 / n_1 + (s_2**2/n_2)**2 / n_2      # 잘못됨
    ```

    분모는 $n_i$가 아니라 $n_i - 1$이어야 한다.

    $$
    \nu = \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}
              {\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}
    $$

    표본이 크면 차이가 작지만($n_1=65$, $n_2=75$에서 $137.77$ 대 $135.84$), 작으면 무시할 수 없다. 연습문제 18($n_1=n_2=5$)에서는 $9.09$ 대 $7.27$로 **자유도가 25% 부풀려진다.** 아래 풀이는 모두 올바른 공식을 쓴다.

---

**1. 귀무가설과 대립가설 세우기**

[Writing null and alternative hypotheses (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/e/writing-null-and-alternative-hypotheses-informal)

**2. 평균에 대한 검정의 가설 세우기**

[Writing Hypotheses for a Test about a Mean (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-quantitative-means/one-sample-t-test-mean/e/writing-hypotheses-one-sample-t-test-mean)

---

**3. 자동 음료 기계**

한 음식점 주인이 새 자동 음료 기계를 설치했다. 이 기계는 중간 크기 설정에서 530 mL를 따르도록 설계되었다. 주인은 기계가 중간 크기 음료를 너무 많이 따른다고 의심하여, 평균이 530 mL보다 유의하게 큰지 확인하려고 중간 크기 음료 30잔을 표본으로 뽑기로 했다. 유의성 검정에 적절한 가설은 무엇인가?

??? success "풀이"
    $$H_0: \mu = 530 \quad \text{대} \quad H_1: \mu > 530$$

---

**4. 제1종 오류와 제2종 오류**

[Type I vs Type II Error (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/error-probabilities-power/e/type-i-error-type-ii-error-power)

---

**5. 실업률 (제1종 오류)**

어떤 지역 시장이 전국 실업률 9%가 자기 도시에도 해당하는지 알아보려 한다. 검정하는 가설은

$$H_0: p = 0.09 \quad \text{대} \quad H_1: p \neq 0.09$$

시장이 제1종 오류를 범하는 것은 어떤 경우인가?

??? success "풀이"
    도시의 실업률이 실제로 9%인데도 그렇지 않다고 결론짓는 경우이다.

---

**6. 모의실험으로 $p$값 추정하기**

[Estimating p-values from Simulations (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/e/estimating-p-values-and-making-conclusions)

**7. $p$값으로 결론 내리기**

[Using P-values to make conclusions (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/a/p-value-conclusions)

---

**8. 모의실험을 이용한 $p$값 추정**

Evie는 청소년의 6%가 채식주의자라는 글을 읽었다. 자기 학교의 비율은 더 높다고 생각한다. 학생 25명을 표본으로 뽑았더니 20%가 채식주의자였다.

$$H_0: p = 0.06 \quad \text{대} \quad H_1: p > 0.06$$

채식주의자 비율이 6%인 모집단에서 $n = 25$인 표본 40개를 모의생성했다.

??? success "풀이"
    ```python
    import numpy as np

    statistic = 0.2
    rng = np.random.default_rng(0)

    p, num_samples, n = 0.06, 40, 25
    samples = rng.binomial(1, p, size=(num_samples, n)).sum(axis=1) / n
    estimated_p_value = (samples >= statistic).mean()
    print(estimated_p_value)
    ```

    모의실험 횟수가 40회뿐이므로 추정된 $p$값의 눈금이 $1/40 = 0.025$로 거칠다. 실무에서는 훨씬 큰 값을 쓴다.

---

**9. 이중언어 사용자 (p값)**

Fay는 미국인의 26%가 두 개 이상의 언어를 구사한다는 글을 읽었다. $H_0: p = 0.26$ 대 $H_1: p > 0.26$을 검정한다. 120명 중 40명이 두 개 이상의 언어를 구사했다.

??? success "풀이"
    $\hat{p} = 40/120 = 0.3333$이므로

    $$
    z = \frac{0.3333 - 0.26}{\sqrt{0.26 \times 0.74/120}} = 1.8314, \qquad p = 0.0335
    $$

    $\alpha = 0.05$에서 기각한다.

---

**10. 전구의 평균수명 (일표본 $z$ 검정)**

제조사는 평균수명이 1,200시간이라고 주장한다. 전구 36개 표본에서 $\bar{X} = 1150$시간, $s = 200$시간을 얻었다. $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $$H_0: \mu = 1200 \quad \text{대} \quad H_1: \mu \neq 1200$$

    $$z = \frac{1150 - 1200}{200/\sqrt{36}} = \frac{-50}{33.33} = -1.5$$

    임계값 $z = \pm 1.96$. $-1.96 < -1.5 < 1.96$이므로 $H_0$을 **기각하지 못한다.**

---

**11. 평균에 대한 $t$ 검정의 조건**

[Conditions for a t Test about a Mean (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-quantitative-means/one-sample-t-test-mean/e/checking-conditions-one-sample-t-test-mean)

---

**12. 시리얼의 평균 중량 (일표본 $t$ 검정)**

표시된 평균 중량은 500 g이다. 25상자 표본에서 $\bar{X} = 490$ g, $s = 15$ g를 얻었다. $\alpha = 0.01$에서 검정하라.

??? success "풀이"
    $$H_0: \mu = 500 \quad \text{대} \quad H_1: \mu \neq 500$$

    $$t = \frac{490 - 500}{15/\sqrt{25}} = \frac{-10}{3} = -3.33$$

    임계값 $t_{0.005,\,24} = \pm 2.797$. $|-3.33| > 2.797$이므로 $H_0$을 **기각한다.** 평균 중량이 500 g와 유의하게 다르다.

---

**13. 동전 (일표본 비율 검정)**

Rahim이 $H_0: p = 0.5$ 대 $H_1: p > 0.5$를 검정한다. 100번 돌려 59번 앞면이 나왔다.

??? success "풀이"
    $$z = \frac{0.59 - 0.5}{\sqrt{0.5 \times 0.5/100}} = 1.80, \qquad p = 0.0359$$

    $0.0359 \le 0.05$이므로 $H_0$을 기각한다.

    ```python
    import numpy as np
    from scipy import stats

    p_hat, n, p_0 = 0.59, 100, 0.5
    z = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
    print(z, stats.norm.sf(z))
    ```

---

**14. 검사를 통과한 차량의 비율**

주장: 80%가 통과한다. 표본: 100대 중 74대 통과. $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $$H_0: p = 0.80 \quad \text{대} \quad H_1: p \neq 0.80$$

    $$z = \frac{0.74 - 0.80}{\sqrt{0.80 \times 0.20/100}} = \frac{-0.06}{0.04} = -1.5$$

    임계값 $\pm 1.96$. $|-1.5| < 1.96$이므로 **기각하지 못한다.**

---

**15. 평균 연봉의 차이 (이표본 $t$ 검정)**

프로그램 A: $\bar{X}_1 = 55{,}000$, $s_1 = 7{,}500$, $n_1 = 14$. 프로그램 B: $\bar{X}_2 = 60{,}000$, $s_2 = 8{,}000$, $n_2 = 16$. 등분산을 가정한다. $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $$S_p^2 = \frac{13 \times 7500^2 + 15 \times 8000^2}{28} = 60{,}401{,}786$$

    $$t = \frac{55{,}000 - 60{,}000}{\sqrt{60{,}401{,}786\left(\frac{1}{14}+\frac{1}{16}\right)}} = -1.758$$

    $df = 28$, $t_{0.025,\,28} = \pm 2.048$. $|-1.758| < 2.048$이므로 **기각하지 못한다** ($p = 0.0897$).

---

**16. 평균 연간소득: 노르웨이 대 미국**

|  | 노르웨이 | 미국 |
|:---:|:---:|:---:|
| 평균 | 64.3 | 53.4 |
| 표준편차 | 18.2 | 23.9 |
| $n$ | 65 | 75 |

$\alpha = 0.05$에서 $H_0: \mu_A = \mu_B$ 대 $H_1: \mu_A \neq \mu_B$를 검정하라.

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    m1, m2, s1, s2, n1, n2 = 64.3, 53.4, 18.2, 23.9, 65, 75
    t = (m1 - m2) / np.sqrt(s1**2/n1 + s2**2/n2)
    df = ((s1**2/n1 + s2**2/n2)**2
          / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)))
    p = 2 * stats.t.cdf(-abs(t), df)
    ```

    | 양 | 값 |
    |:---|---:|
    | $t$ | 3.0572 |
    | $\nu$ | 135.84 |
    | $p$ | **0.0027** |

    $H_0$을 **기각한다.** 두 나라의 평균 소득이 유의하게 다르다.

---

**17. 초혼 평균연령: 미국 대 캐나다**

|  | 미국 | 캐나다 |
|:---:|:---:|:---:|
| 평균 | 25.5 | 26.3 |
| 표준편차 | 3.8 | 3.2 |
| $n$ | 108 | 102 |

$\alpha = 0.05$에서 검정하라(합동분산).

??? success "풀이"
    $t = -1.6454$, $df = 208$, $p = 0.1014$. **기각하지 못한다.**

---

**18. 전기차 모델 A와 B의 평균 주행거리**

|  | 모델 A | 모델 B |
|:---:|:---:|:---:|
| 평균 | 168 km | 172 km |
| 표준편차 | 5.4 km | 7.5 km |
| $n$ | 5 | 5 |

$\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $t = -0.9678$, Welch 자유도 $\nu = 7.27$, $p = 0.3642$. **기각하지 못한다.**

    이 문제가 위 정정 상자의 사례다. 잘못된 공식은 $\nu = 9.09$를 주어 $p = 0.3582$가 된다. 결론은 같지만 자유도가 25% 부풀려진다. 각 집단이 5개뿐인 자료에서 자유도를 과대평가하면 **검정이 실제보다 관대해진다.**

---

**19. 평균 교육연수: 이탈리아 대 프랑스**

|  | 이탈리아 | 프랑스 |
|:---:|:---:|:---:|
| 평균 | 10.7 | 10.4 |
| 표준편차 | 2.3 | 2.5 |
| $n$ | 46 | 58 |

합동분산을 써서 $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $t = 0.6295$, $df = 102$, $p = 0.5304$. **기각하지 못한다.**

---

**20. 두 부서의 평균 연봉 (Welch $t$ 검정)**

부서 A: $\bar{X}_1 = 60{,}000$, $s_1 = 8{,}000$, $n_1 = 12$. 부서 B: $\bar{X}_2 = 65{,}000$, $s_2 = 10{,}000$, $n_2 = 15$. 이분산. $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $$t = \frac{60{,}000 - 65{,}000}{\sqrt{\frac{8000^2}{12} + \frac{10000^2}{15}}} = \frac{-5{,}000}{3{,}464.1} = -1.4434$$

    $\nu = 25.0$, $t_{0.025,\,25} = \pm 2.060$, $p = 0.1613$. **기각하지 못한다.**

---

**21. 금연 성공 비율의 차이**

처치 A: 200명 중 60명 금연. 처치 B: 180명 중 54명 금연. $\alpha = 0.05$에서 검정하라.

??? success "풀이"
    $$\hat{p}_1 = 0.30, \quad \hat{p}_2 = 0.30, \quad \hat{p}_{\text{pool}} = 114/380 = 0.30$$

    $$z = \frac{0.30 - 0.30}{\sqrt{0.30 \times 0.70 \times (1/200 + 1/180)}} = 0$$

    $|0| < 1.96$이므로 **기각하지 못한다.** 두 처치에 유의한 차이가 없다.

    두 표본비율이 정확히 같으므로 표본크기와 무관하게 $z = 0$이다.

---

## 대응표본 연습문제

**22. 운동 프로그램의 효과**

| 참가자 | 사전 (%) | 사후 (%) |
|---|---|---|
| 1 | 25 | 23 |
| 2 | 28 | 26 |
| 3 | 30 | 28 |
| 4 | 27 | 26 |
| 5 | 32 | 30 |
| 6 | 29 | 27 |
| 7 | 26 | 25 |
| 8 | 31 | 29 |
| 9 | 24 | 23 |
| 10 | 30 | 28 |

**(1)** 자료의 유형은? **(2)** 표집분포는? **(3)** 검정을 수행하라.

??? success "풀이"
    **(1)** **대응표본**이다. 같은 참가자를 전후로 측정했다.

    **(2)** $t_9$. 대응차이 $n = 10$이므로 $df = 9$.

    **(3)** 차이(사전 $-$ 사후)는 $2, 2, 2, 1, 2, 2, 1, 2, 1, 2$이다.

    $$\bar{d} = 1.7, \quad s_d = 0.4830, \quad t = \frac{1.7}{0.4830/\sqrt{10}} = 11.13$$

    임계값 $t_{0.05,\,9} = 1.833$(단측). $11.13 > 1.833$이므로 $H_0$을 **기각한다** ($p \approx 10^{-6}$). 운동 프로그램이 체지방률을 유의하게 낮춘다.

    차이가 모두 $1$ 또는 $2$로 매우 일관되어 $s_d$가 작다. 이것이 $t$ 통계량을 크게 만든다. **대응설계가 개체 간 변동을 제거한 결과다.**

---

**23. 식이요법과 콜레스테롤 수치**

참가자 12명을 6주 프로그램 전후로 측정했다. $\bar{d} = 7.5$, $s_d = 2.61$이다.

??? success "풀이"
    $$t = \frac{7.5}{2.61/\sqrt{12}} = \frac{7.5}{0.7534} = 9.954$$

    $df = 11$, $t_{0.05,\,11} = 1.796$(단측). $9.954 > 1.796$이므로 **기각한다.** 식이요법이 콜레스테롤을 유의하게 낮춘다.
