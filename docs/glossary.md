# 용어집

이 책이 쓰는 약어와 전문 용어를 영어와 맞대어 적었다. 본문에서 약어가 처음
나오는 자리는 모두 이 표로 링크되어 있으므로, 낯선 약어를 만나면 눌러서
돌아오면 된다. 영어를 함께 적는 까닭은 이 책 밖의 문헌이 거의 다 영어이기
때문이다 — 「가족단위 오류율」로 검색해서 나오지 않는 것이 `familywise error
rate` 로는 나온다.

## 1. 약어

| 약어 | 우리말 | 영어 | 설명하는 쪽 |
|---|---|---|---|
| <span id="ad"></span>**AD** ※ | 앤더슨–달링 | Anderson-Darling | [앤더슨–달링 검정](ch14/formal_tests/anderson_darling.md) |
| <span id="aic"></span>**AIC** | 아카이케 정보기준 | Akaike information criterion | [Akaike 정보기준](ch12/model_selection/aic.md) |
| <span id="anova"></span>**ANOVA** | 분산분석 | analysis of variance | [모형과 가정](ch10/anova_one_way/one_way_model.md) |
| <span id="ar"></span>**AR** | 자기회귀 | autoregressive | — |
| <span id="are"></span>**ARE** | 점근상대효율 | asymptotic relative efficiency | [표본평균의 효율성](ch05/mean/efficiency.md) |
| <span id="ate"></span>**ATE** | 평균처치효과 | average treatment effect | [인과 추론의 기준](ch11/causation/causal_criteria.md) |
| <span id="auc"></span>**AUC** | 곡선아래면적 | area under the curve | [ROC 곡선과 AUC](ch19/evaluation/roc_auc.md) |
| <span id="bh"></span>**BH** | 벤자미니–호크버그 | Benjamini-Hochberg | [거짓발견율 (Benjamini-Hochberg)](ch09/multiple_testing/fdr.md) |
| <span id="bic"></span>**BIC** | 베이즈 정보기준 | Bayesian information criterion | [베이즈 정보기준](ch12/model_selection/bic.md) |
| <span id="by"></span>**BY** ※ | 벤자미니–예쿠티엘리 (그림 출처의 「CC BY」와 겹친다) | Benjamini-Yekutieli | [거짓발견율 (Benjamini-Hochberg)](ch09/multiple_testing/fdr.md) |
| <span id="cdf"></span>**CDF** | 누적분포함수 | cumulative distribution function | [확률질량함수, 확률밀도함수, 누적분포함수](ch03/rv/pmf_pdf_cdf.md) |
| <span id="ci"></span>**CI** | 신뢰구간 | confidence interval | — |
| <span id="clt"></span>**CLT** | 중심극한정리 | central limit theorem | [중심극한정리](ch03/limits/clt.md) |
| <span id="crlb"></span>**CRLB** | 크라메르–라오 하한 | Cramer-Rao lower bound | [효율성과 Cramér-Rao 하한](ch07/theory/efficiency_crlb.md) |
| <span id="cv"></span>**CV** ※ | 변동계수 / 교차검증 | coefficient of variation / cross-validation | — |
| <span id="dag"></span>**DAG** | 유향비순환그래프 | directed acyclic graph | [인과 추론의 기준](ch11/causation/causal_criteria.md) |
| <span id="dffits"></span>**DFFITS** | 적합값 변화량 | difference in fits | [영향점과 지렛대 (쿡 거리, DFFITS)](ch12/diagnostics/influence.md) |
| <span id="ecdf"></span>**ECDF** | 경험누적분포함수 | empirical cumulative distribution function | [히스토그램과 밀도 그림](ch02/exploration/histograms.md) |
| <span id="em"></span>**EM** | 기댓값–최대화 알고리즘 | expectation-maximization algorithm | — |
| <span id="f1"></span>**F1** | F₁ 점수 | F1 score | [혼동행렬·ROC·분류 지표](ch19/evaluation/metrics.md) |
| <span id="fdr"></span>**FDR** | 거짓발견율 | false discovery rate | [거짓발견율 (Benjamini-Hochberg)](ch09/multiple_testing/fdr.md) |
| <span id="fpc"></span>**FPC** | 유한모집단 수정 | finite population correction | [모집단과 표본](ch01/classical/population_sample.md) |
| <span id="fpr"></span>**FPR** | 거짓양성률 | false positive rate | [혼동행렬·ROC·분류 지표](ch19/evaluation/metrics.md) |
| <span id="fwer"></span>**FWER** | 가족단위 오류율 | familywise error rate | [가족단위 오류율](ch09/multiple_testing/fwer.md) |
| <span id="gam"></span>**GAM** | 일반화 가법모형 | generalized additive model | — |
| <span id="garch"></span>**GARCH** | 일반화 자기회귀 조건부 이분산 | generalized autoregressive conditional heteroskedasticity | — |
| <span id="gcv"></span>**GCV** | 일반화 교차검증 | generalized cross-validation | [교차검증 다항 모형선택](ch12/model_selection/cv_polynomial_model_selection.md) |
| <span id="glm"></span>**GLM** | 일반화선형모형 | generalized linear model | — |
| <span id="gmm"></span>**GMM** | 일반화 적률법 | generalized method of moments | [적률법의 기초](ch06/estimation_methods/moments_foundations.md) |
| <span id="hac"></span>**HAC** | 이분산·자기상관 일치 표준오차 | heteroskedasticity and autocorrelation consistent standard error | — |
| <span id="hc3"></span>**HC3** | 이분산 일치 표준오차 | heteroskedasticity-consistent covariance estimator (type 3) | [이원배치 웰치 분산분석 (로버스트 HC3)](ch10/anova_welch/welch_twoway_robust.md) |
| <span id="hsd"></span>**HSD** | 정직유의차 | honestly significant difference | [투키 정직유의차](ch10/post_hoc/tukey.md) |
| <span id="icc"></span>**ICC** | 급내상관계수 | intraclass correlation coefficient | — |
| <span id="iqr"></span>**IQR** | 사분위범위 | interquartile range | [사분위범위와 로버스트 측도](ch02/spread/robust.md) |
| <span id="iv"></span>**IV** ※ | 도구변수 (책의 「IV 부」는 로마숫자다) | instrumental variable | [도구변수 (입문)](ch11/causation/instrumental_variables.md) |
| <span id="jb"></span>**JB** | 자크–베라 | Jarque-Bera | — |
| <span id="kde"></span>**KDE** | 커널밀도추정 | kernel density estimation | [히스토그램과 밀도 그림](ch02/exploration/histograms.md) |
| <span id="kkt"></span>**KKT** | 카루시–쿤–터커 조건 | Karush-Kuhn-Tucker conditions | [능형 정식화와 닫힌 형태의 해](ch18/ridge/formulation.md) |
| <span id="ks"></span>**KS** | 콜모고로프–스미르노프 | Kolmogorov-Smirnov | [콜모고로프–스미르노프 검정](ch14/formal_tests/ks_test.md) |
| <span id="lda"></span>**LDA** | 선형판별분석 | linear discriminant analysis | [선형판별분석과 이차판별분석](ch19/generative/lda_qda_classification.md) |
| <span id="loocv"></span>**LOOCV** | 하나뺀 교차검증 | leave-one-out cross-validation | [모형선택을 위한 교차검증](ch12/model_selection/cross_validation.md) |
| <span id="lrt"></span>**LRT** | 가능도비 검정 | likelihood ratio test | [분산에 대한 가능도비 검정](ch15/advanced_methods/likelihood_ratio.md) |
| <span id="lsd"></span>**LSD** | 최소유의차 | least significant difference | [사후비교 방법의 비교와 검정력](ch10/post_hoc/post_hoc_comparisons.md) |
| <span id="mad"></span>**MAD** | 중앙값절대편차 | median absolute deviation | [사분위범위와 로버스트 측도](ch02/spread/robust.md) |
| <span id="mae"></span>**MAE** | 평균절대오차 | mean absolute error | [평균절대오차, 평균제곱오차, 제곱근평균제곱오차](ch12/performance/mae_mse_rmse.md) |
| <span id="map"></span>**MAP** | 최대사후확률 추정 | maximum a posteriori | [베이즈 정리](ch03/probability/bayes.md) |
| <span id="mape"></span>**MAPE** | 평균절대백분율오차 | mean absolute percentage error | — |
| <span id="mcmc"></span>**MCMC** | 마르코프 연쇄 몬테카를로 | Markov chain Monte Carlo | — |
| <span id="mgf"></span>**MGF** | 적률생성함수 | moment generating function | [적률생성함수](ch03/rv/mgf.md) |
| <span id="mle"></span>**MLE** | 최대가능도추정 | maximum likelihood estimation | [가능도와 최대가능도추정](ch06/mle/likelihood.md) |
| <span id="mse"></span>**MSE** | 평균제곱오차 | mean squared error | [편향–분산 분해와 평균제곱오차](ch05/estimators/bias_variance.md) |
| <span id="ols"></span>**OLS** | 보통최소제곱 | ordinary least squares | [최소제곱 추정](ch12/estimation_inference/least_squares.md) |
| <span id="pca"></span>**PCA** | 주성분분석 | principal component analysis | [주성분회귀](ch18/pcr_pls/principal_components_regression.md) |
| <span id="pcr"></span>**PCR** | 주성분회귀 | principal component regression | [주성분회귀](ch18/pcr_pls/principal_components_regression.md) |
| <span id="pdf"></span>**PDF** ※ | 확률밀도함수 | probability density function | [확률질량함수, 확률밀도함수, 누적분포함수](ch03/rv/pmf_pdf_cdf.md) |
| <span id="pls"></span>**PLS** | 부분최소제곱 | partial least squares | [주성분회귀](ch18/pcr_pls/principal_components_regression.md) |
| <span id="pmf"></span>**PMF** | 확률질량함수 | probability mass function | [확률질량함수, 확률밀도함수, 누적분포함수](ch03/rv/pmf_pdf_cdf.md) |
| <span id="qda"></span>**QDA** | 이차판별분석 | quadratic discriminant analysis | [선형판별분석과 이차판별분석](ch19/generative/lda_qda_classification.md) |
| <span id="rmse"></span>**RMSE** | 제곱근평균제곱오차 | root mean squared error | [평균절대오차, 평균제곱오차, 제곱근평균제곱오차](ch12/performance/mae_mse_rmse.md) |
| <span id="roc"></span>**ROC** | 수신자조작특성 | receiver operating characteristic | [ROC 곡선과 AUC](ch19/evaluation/roc_auc.md) |
| <span id="rss"></span>**RSS** | 잔차제곱합 | residual sum of squares | [단순선형회귀](ch12/linear_regression/simple.md) |
| <span id="sd"></span>**SD** | 표준편차 | standard deviation | [분산과 표준편차](ch02/spread/variance_std.md) |
| <span id="se"></span>**SE** | 표준오차 | standard error | [표준오차](ch05/applications/standard_error.md) |
| <span id="sse"></span>**SSE** | 오차제곱합 | error sum of squares | [F-검정 절차](ch10/anova_one_way/f_test.md) |
| <span id="sst"></span>**SST** | 총제곱합 | total sum of squares | [F-검정 절차](ch10/anova_one_way/f_test.md) |
| <span id="svd"></span>**SVD** | 특이값분해 | singular value decomposition | — |
| <span id="umvue"></span>**UMVUE** | 균일최소분산 불편추정량 | uniformly minimum variance unbiased estimator | — |
| <span id="vc"></span>**VC** ※ | VC 차원 (밥닉–체르보넨키스) | Vapnik-Chervonenkis dimension | — |
| <span id="vif"></span>**VIF** | 분산팽창인자 | variance inflation factor | [다중공선성과 분산팽창인자](ch12/diagnostics/multicollinearity.md) |
| <span id="wls"></span>**WLS** | 가중최소제곱 | weighted least squares | [가중최소제곱](ch12/assumptions/weighted_regression.md) |

※ 표시는 이 책 안에서 **뜻이 둘 이상인** 약어다. CV 는 변동계수이기도 하고
교차검증이기도 하며, VC 는 VC 차원이기도 하고 ToothGrowth 자료의 비타민 C
투여 방법이기도 하다. 그래서 본문에서 자동으로 링크를 걸지 않는다.

## 2. 우리말–영어 대조표

본문은 이 표의 용어가 **쪽마다 처음 나오는 자리**에 영어를 괄호로 병기한다.

| 우리말 | 영어 |
|---|---|
| 가능도비 | likelihood ratio |
| 가족단위 오류율 | familywise error rate |
| 거짓발견율 | false discovery rate |
| 검정통계량 | test statistic |
| 결정계수 | coefficient of determination |
| 결합분포 | joint distribution |
| 경험분포함수 | empirical distribution function |
| 과적합 | overfitting |
| 교락 | confounding |
| 교차검증 | cross-validation |
| 교호작용 | interaction |
| 구간추정 | interval estimation |
| 귀무가설 | null hypothesis |
| 균일분포 | uniform distribution |
| 기각역 | rejection region |
| 누적분포함수 | cumulative distribution function |
| 능형회귀 | ridge regression |
| 다중공선성 | multicollinearity |
| 다중비교 | multiple comparisons |
| 다중회귀 | multiple regression |
| 단계적 선택 | stepwise selection |
| 단순회귀 | simple linear regression |
| 대립가설 | alternative hypothesis |
| 대수의 법칙 | law of large numbers |
| 대응표본 | paired samples |
| 독립성 | independence |
| 독립표본 | independent samples |
| 등분산성 | homoscedasticity |
| 레버리지 | leverage |
| 로지스틱 회귀 | logistic regression |
| 막대그래프 | bar chart |
| 모비율 | population proportion |
| 몬테카를로 | Monte Carlo |
| 무작위화 | randomization |
| 반복측정 | repeated measures |
| 베르누이 시행 | Bernoulli trial |
| 베이즈 정리 | Bayes' theorem |
| 변동계수 | coefficient of variation |
| 부트스트랩 | bootstrap |
| 분산분석 | analysis of variance |
| 분산팽창인자 | variance inflation factor |
| 분위수 | quantile |
| 불편추정량 | unbiased estimator |
| 블록화 | blocking |
| 비모수 | nonparametric |
| 사분위범위 | interquartile range |
| 사분위수 | quartile |
| 사전분포 | prior distribution |
| 사후검정 | post hoc test |
| 사후분포 | posterior distribution |
| 사후비교 | post hoc comparison |
| 산점도 | scatter plot |
| 상자그림 | box plot |
| 생존함수 | survival function |
| 선형대비 | linear contrast |
| 선형회귀 | linear regression |
| 설명변수 | explanatory variable |
| 순열검정 | permutation test |
| 스튜던트화 범위 | studentized range |
| 신뢰수준 | confidence level |
| 실험계획 | experimental design |
| 양측검정 | two-sided test |
| 연속성 수정 | continuity correction |
| 예측구간 | prediction interval |
| 오차항 | error term |
| 왜도 | skewness |
| 위험비 | hazard ratio |
| 유의확률 | p-value |
| 이분산성 | heteroscedasticity |
| 이상치 | outlier |
| 이원배치 분산분석 | two-way analysis of variance |
| 이항분포 | binomial distribution |
| 일반화선형모형 | generalized linear model |
| 일원배치 분산분석 | one-way analysis of variance |
| 일치추정량 | consistent estimator |
| 적합도 | goodness of fit |
| 전진선택 | forward selection |
| 절편 | intercept |
| 점추정 | point estimation |
| 정규방정식 | normal equations |
| 정규성 | normality |
| 제1종 오류 | type I error |
| 제2종 오류 | type II error |
| 제곱합 | sum of squares |
| 조건부확률 | conditional probability |
| 조화평균 | harmonic mean |
| 종속변수 | dependent variable |
| 주성분분석 | principal component analysis |
| 중도절단 | censoring |
| 중심극한정리 | central limit theorem |
| 지렛값 | leverage |
| 지수분포 | exponential distribution |
| 질적 변수 | categorical variable |
| 첨도 | kurtosis |
| 최대가능도추정 | maximum likelihood estimation |
| 최빈값 | mode |
| 최소제곱법 | least squares |
| 축소추정 | shrinkage estimation |
| 층화 | stratification |
| 층화추출 | stratified sampling |
| 카이제곱분포 | chi-squared distribution |
| 쿡 거리 | Cook's distance |
| 크론바흐 알파 | Cronbach's alpha |
| 트레이드오프 | trade-off |
| 평균제곱오차 | mean squared error |
| 포아송분포 | Poisson distribution |
| 표본공간 | sample space |
| 표본분산 | sample variance |
| 표본평균 | sample mean |
| 표집분포 | sampling distribution |
| 합동분산 | pooled variance |
| 혼동행렬 | confusion matrix |
| 확률과정 | stochastic process |
| 확률밀도함수 | probability density function |
| 확률질량함수 | probability mass function |
| 회귀계수 | regression coefficient |
| 회귀진단 | regression diagnostics |
| 효과크기 | effect size |

## 3. 기초어

아래는 **병기하지 않는** 말이다. 둘 가운데 하나다 — 책 거의 모든 쪽에 나와
쪽마다 괄호를 달면 글이 괄호에 잠기거나(「분산」은 $474$ 쪽에 있다),
일상어와 겹쳐 기계가 뜻을 가릴 수 없거나(「대비」는 contrast 이기도 하고
「~에 대비하여」이기도 하다). 영어가 궁금하면 여기서 찾는다.

| 우리말 | 영어 |
|---|---|
| 검정력 | power |
| 계수 | coefficient |
| 공분산 | covariance |
| 기댓값 | expected value |
| 대비 | contrast |
| 모수 | parameter |
| 모집단 | population |
| 모평균 | population mean |
| 벌점 | penalty |
| 분산 | variance |
| 비율 | proportion |
| 상관계수 | correlation coefficient |
| 수렴 | convergence |
| 신뢰구간 | confidence interval |
| 우도 | likelihood |
| 유의수준 | significance level |
| 자유도 | degrees of freedom |
| 잔차 | residual |
| 정규분포 | normal distribution |
| 중앙값 | median |
| 추정량 | estimator |
| 통계량 | statistic |
| 편의 | bias (또는 편의표집의 convenience) |
| 편차 | deviation |
| 편향 | bias |
| 표본크기 | sample size |
| 표준오차 | standard error |
| 표준편차 | standard deviation |
| 확률변수 | random variable |
