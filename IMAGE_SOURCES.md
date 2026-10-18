# 「그림으로 보는 통계학」·「AI로 보는 통계학」 — 옮긴 상자와 이미지 출처

`high_school_math` 10장(확률과 통계 = 그 책의 `ch07`·`ch09`)에 있는 상자 **114 개**를
이 책으로 옮기는 작업의 대장이다. `docs/` 밖에 두어 빌드에 끼지 않는다.

**이 파일을 두는 까닭** — 옮긴 이미지는 대부분 우표·지폐·박물관 사진이고 CC0·PD·CC BY
가 섞여 있다. 나중에 이 책을 공개하거나 출판할 일이 생기면 **이 파일만 훑으면** 저작권
점검이 끝나도록 한다. 원본 책의 `STAMP_SOURCES.md` 와 같은 구실이다.

⚠️ **라이선스는 틀 이름으로 판단한다.** 「Commons 에 있고 PD 라고 적혀 있다」로는 모자라다.
자세한 기준은 `high_school_math/STAMP_SOURCES.md` 를 따른다.

## 옮길 때 지키는 것

1. 연재 이름 **수학 → 통계학**
2. 그림 경로 `figures/` → `./img/` (이 책의 관례), 그림 파일도 함께 옮긴다
3. **그림 설명의 출처·라이선스 문구는 글자 그대로 옮긴다**
4. 원본 쪽의 정리/정의 번호를 부르는 문장은 **옮긴 쪽의 번호로 다시 맞춘다**
5. 이 책에 없는 쪽으로 가는 링크는 **빼거나** 사실만 남긴다
6. 자리는 그 상자가 이야기하는 정의/정리 **바로 뒤**(정리면 증명 뒤), 보기 앞

## 진행

| # | 원본 | 상자 | 제목 | 옮긴 곳 | 이미지 | 라이선스 | 상태 |
|---|---|---|---|---|---|---|---|
| 1 | `ch07/basics/birthday_problem.md` | 그림 | 믿기지 않아서 발표하지 않았다 | `—` | `pb__vonmises.png` | 프랑스 국립도서관 소장, Wikimedia Commons, PD-US-expired | 대기 |
| 2 | `ch07/basics/case_division_probability.md` | 그림 | 포탄이 어디에 떨어지는지 모를 때 쓰는 표 | `—` | `cs__fm640_1945_p19.png` | — | 대기 |
| 3 | `ch07/basics/combination_probability.md` | 그림 | 열셋인지 열넷인지 아직 다투는 한 벌 | `—` | `pb__mamluk_cards.png` | 낱장 사진을 Countakeshi 가 이어 붙인 것, Wikimedia Commons, CC BY-SA 4. | 대기 |
| 4 | `ch07/basics/complement_probability.md` | 그림 | 원숭이 백만 마리를 불러낸 까닭 | `ch03/probability/axioms.md` | `pb__borel1929.png` | Wikimedia Commons · 프랑스 국립도서관, 저작권 만료 | 대기 |
| 5 | `ch07/basics/dice_pair_condition.md` | 그림 | 반듯하지 않은 주사위 | `—` | `prob__roman_dice.png` | Wikimedia Commons, CC BY-SA 3.0 | 대기 |
| 6 | `ch07/basics/function_probability.md` | 그림 | 공을 칸에 던지면 함수가 하나 생긴다 | `—` | `fp__todhunter1889_p461.png` | — | 대기 |
| 7 | `ch07/basics/integer_solution_probability.md` | 그림 | 「특별한 재주」라 부르던 증명 | `—` | `is__netto1901_p21.png` | — | 대기 |
| 8 | `ch07/basics/permutation_probability.md` | 그림 | 열네 글자로 만든 자물쇠 | `—` | `pp__hooke1676_p31.png` | — | 대기 |
| 9 | `ch07/basics/probability.md` | 그림 | 세 줄로 적은 확률 | `ch03/probability/axioms.md` | `prob__kolmogorov.png, prob__grundbegriffe1933.png` | Wikimedia Commons, 촬영 Konrad Jacobs, 오버볼파흐 수학연구소, CC BY-SA 2 | 대기 |
| 10 | `ch07/basics/probability_addition.md` | 그림 | 인쇄 오류를 평생 미워한 사람의 이름이 붙은 부등식 | `ch03/probability/axioms.md` | `pb__bonferroni.png` | Wikimedia Commons, PD-Italy | 대기 |
| 11 | `ch07/basics/sample_space.md` | 그림 | 네 면으로만 서는 주사위 | `ch03/probability/sample_space.md` | `pb__ancient_dice.png` | Wikimedia Commons, 촬영 Xocolatl, PD-self / 메트로폴리탄 미술관 10.130. | **옮김** |
| 12 | `ch07/conditional/bayes.md` | 그림 | 되짚기가 활자로 놓인 자리 | `ch03/probability/bayes.md` | `cd__by_bayes1763_prop5.png` | Wikimedia Commons, PD-old-100 | 대기 |
| 13 | `ch07/conditional/bayes.md` | 그림 | 곱하기를 더하기로 바꾸어 암호를 풀다 | `ch03/probability/bayes.md` | `cd__turing1936.png` | Wikimedia Commons, PD-US-not renewed | 대기 |
| 14 | `ch07/conditional/classification_table.md` | 그림 | 네 칸짜리 표에 이름이 붙던 날 | `—` | `cd__pearson1904_p1.png` | — | 대기 |
| 15 | `ch07/conditional/classification_table.md` | AI | 「정확도 $99$ 퍼센트」가 쓸모없을 수 있는 까닭 | `—` | `—` | — | 대기 |
| 16 | `ch07/conditional/conditional_applications.md` | 그림 | 「적어도 하나」를 비밀 문서에 적어 둔 까닭 | `—` | `cd__morse_kimball1946_p89.png` | — | 대기 |
| 17 | `ch07/conditional/conditional_applications.md` | AI | 장치를 늘릴수록 더 자주 멈추는 까닭 | `—` | `—` | — | 대기 |
| 18 | `ch07/conditional/conditional_independence.md` | 그림 | 「서로 다른 기록을 섞을 때 생기는 오류」 | `ch03/probability/conditional_independence.md` | `cd__yule1903_p132.png` | — | 대기 |
| 19 | `ch07/conditional/conditional_probability.md` | 그림 | 곱셈정리가 활자로 놓인 자리 | `ch03/probability/conditional.md` | `cd__demoivre_doctrine.png` | Wikimedia Commons, PD-old-100-expired | 대기 |
| 20 | `ch07/conditional/conditional_probability.md` | AI | 문장 하나의 확률을 낱말 하나씩 쪼개는 법 | `ch03/probability/conditional.md` | `cd__ai_chain_scale.png` | — | 대기 |
| 21 | `ch07/conditional/conditional_state.md` | 그림 | 네 칸은 세어서 얻은 것이 아니라 $1$ 을 전개한 것 | `—` | `cd__boole1854_p76.png` | — | 대기 |
| 22 | `ch07/conditional/conditional_two_stage.md` | 그림 | 사전확률을 적지 못한 채 돌린 계산 | `—` | `cd__bordereau1894.png` | 프랑스 국방사료원 소장, Wikimedia Commons, PD-old-70 | 대기 |
| 23 | `ch07/conditional/game_probability.md` | 그림 | 모든 것을 모아 적은 책에, 풀지 못한 물음 하나 | `—` | `cd__pacioli1523_summa.png` | Stockholms Universitetsbibliotek 촬영, CC BY 2.0 | 대기 |
| 24 | `ch07/conditional/independent_binomial.md` | 그림 | 신의 존재를 증명하려고 쓴 첫 가설검정 | `—` | `cd__arbuthnot1710_p186.png` | — | 대기 |
| 25 | `ch07/conditional/independent_events.md` | 그림 | 완두 오백쉰여섯 알로 잰 독립 | `ch03/probability/independence.md` | `cd__mendel1866_p19.png` | — | 대기 |
| 26 | `ch07/conditional/monty_hall.md` | 그림 | 이름만 빌려준 사람 | `ch03/probability/bayes.md` | `cd__monty_hall.png` | Wikimedia Commons, PD-Pre1978 | 대기 |
| 27 | `ch07/conditional/monty_hall.md` | 그림 | 편지 만 통이 쏟아진 칼럼 | `ch03/probability/bayes.md` | `cd__erdos.png` | Topsy Kretts 촬영, Wikimedia Commons, CC BY 3.0 | 대기 |
| 28 | `ch07/conditional/monty_hall.md` | 그림 | 몬티보다 백 년 먼저 나온 문제 | `ch03/probability/bayes.md` | `cd__bertrand.png` | Wikimedia Commons, PD-old | 대기 |
| 29 | `ch07/conditional/stratified_reversal.md` | 그림 | 심슨은 만화가 아니다 | `—` | `cd__bletchley.png` | Wikimedia Commons, 촬영 DeFacto, CC BY-SA 4.0 | 대기 |
| 30 | `ch07/conditional/stratified_reversal.md` | 그림 | 소송이 될 뻔한 입학 통계 | `—` | `cd__berkeley.png` | Wikimedia Commons, 촬영 Daderot, CC0 | 대기 |
| 31 | `ch07/conditional/total_probability.md` | 그림 | 바다를 칸으로 나누고 하나씩 지워 나간 기록 | `ch03/probability/conditional.md` | `cd__palomares1975_fig333.png` | 미국 공군 자료, PD-USGov | 대기 |
| 32 | `ch07/descriptive/central_tendency.md` | 그림 | 눈금이 없는 그래프 | `ch02/center/mean_median_mode.md` | `dsc__stamp_indonesia1969.png` | Wikimedia Commons, PD | 대기 |
| 33 | `ch07/descriptive/covariance_correlation.md` | 그림 | 「회귀」라는 이름이 나온 그림 | `ch03/rv/correlation.md` | `dc__galton1886_plate10.png` | Wikimedia Commons, PD-Art | 대기 |
| 34 | `ch07/descriptive/dispersion.md` | 그림 | 백분위수를 종이에 그린 것 | `ch02/spread/variance_std.md` | `dsc__growth_chart_cdc2000.png` | Wikimedia Commons, PD-USGov-HHS-CDC | 대기 |
| 35 | `ch07/descriptive/frequency_distribution.md` | 그림 | 통계학이라는 말보다 먼저 있던 도수분포표 | `ch02/exploration/histograms.md` | `fd__bills_of_mortality_1665.png` | Wikimedia Commons, Wellcome Collection, CC BY 4.0 | 대기 |
| 36 | `ch07/descriptive/frequency_distribution.md` | 그림 | 히스토그램을 둥글게 말아 의회에 내민 그림 | `ch02/exploration/histograms.md` | `fd__nightingale1858.png` | Wikimedia Commons, PD-old | 대기 |
| 37 | `ch07/descriptive/frequency_distribution.md` | 그림 | 날 수로 보면 가장 많은 쪽이 가장 적은 쪽이었다 | `ch02/exploration/histograms.md` | `fd__snow1855_p86.png` | — | 대기 |
| 38 | `ch07/distributions/binomial_distribution.md` | 그림 | 이름이 된 책 | `ch04/discrete_distributions/binomial.md` | `bin__ars1713.png` | Wikimedia Commons, PD-old-100 | 대기 |
| 39 | `ch07/distributions/binomial_to_poisson.md` | 그림 | 말에 차여 죽은 사람을 센 표 | `ch04/discrete_distributions/poisson.md` | `binomial_to_poisson__bortkiewicz1898_p25.png` | Wikimedia Commons, PD-US-expired | 대기 |
| 40 | `ch07/distributions/exponential_distribution.md` | 그림 | 전화선을 몇 개 깔 것인가 | `ch04/continuous_distributions/exponential.md` | `ds__molina1927_p463.png` | — | 대기 |
| 41 | `ch07/distributions/geometric_distribution.md` | AI | 기다린 보람이 없으므로 간격을 늘린다 | `ch04/discrete_distributions/geometric.md` | `—` | — | 대기 |
| 42 | `ch07/distributions/geometric_distribution.md` | 그림 | 담뱃갑에 든 쉰 장, 다 모으려면 | `ch04/discrete_distributions/geometric.md` | `ds__cope_warriors_card.png` | Wikimedia Commons, 저작권 만료 | 대기 |
| 43 | `ch07/distributions/hypergeometric_distribution.md` | 그림 | 검사를 얼마나 할 것인가 | `ch04/discrete_distributions/hypergeometric.md` | `ds__dodge1929_p629.png` | — | 대기 |
| 44 | `ch07/distributions/poisson.md` | 그림 | 이름이 붙기 전의 포아송분포 | `ch04/discrete_distributions/poisson.md` | `poisson__poisson1837_p206.png` | Wikimedia Commons, PD-old-100 | 대기 |
| 45 | `ch07/distributions/poisson.md` | 그림 | 에펠탑에 새겨진 이름 | `ch04/discrete_distributions/poisson.md` | `poisson__eiffel.png` | Wikimedia Commons, 촬영 Victor R. Ruiz, CC BY-SA 2.0 | 대기 |
| 46 | `ch07/expectation_applications/conditional_expectation.md` | 그림 | 「검사를 얼마나 할 것인가」에 수로 답한 쪽 | `—` | `ce__dodge_romig1929_p621_628_629.png` | — | 대기 |
| 47 | `ch07/expectation_applications/conditional_expectation.md` | AI | 기계가 찾아내는 것은 결국 $E[Y\mid X]$ 하나 | `—` | `—` | — | 대기 |
| 48 | `ch07/expectation_applications/expectation_counting.md` | 그림 | 가장자리는 싸고 안쪽은 비싸다 | `—` | `ec__hollerith1890.png` | Wikimedia Commons · 저작권 없음 | 대기 |
| 49 | `ch07/expectation_applications/expectation_counting.md` | AI | 자료를 섞어 쓰는 일이 괜찮은 까닭 | `—` | `—` | — | 대기 |
| 50 | `ch07/expectation_applications/expectation_equation.md` | 그림 | 차수를 백 배로 올려도 실근은 셋쯤 늘 뿐이다 | `—` | `ea__ee_kac.png` | — | 대기 |
| 51 | `ch07/expectation_applications/expectation_geometry.md` | 그림 | 제곱을 떼는 순간 손이 닿지 않는다 | `—` | `—` | — | 대기 |
| 52 | `ch07/expectation_applications/expectation_process.md` | 그림 | 「최대 $n$ 번까지만」을 떼어 내면 벌어지는 일 | `—` | `ea__dbernoulli.png` | ETH 취리히 도서관 소장, Wikimedia Commons, PD-old | 대기 |
| 53 | `ch07/expectation_applications/expected_profit.md` | 그림 | 처음 이 셈을 한 곳은 가게가 아니라 은행 금고였다 | `—` | `ea__edgeworth1888_p113.png` | — | 대기 |
| 54 | `ch07/expectation_applications/order_statistics.md` | 그림 | 전차를 세지 않고 전차 수를 센 사람들 | `—` | `ea__panther1944.png` | Archives Normandie, Wikimedia Commons, PD | 대기 |
| 55 | `ch07/expectation_applications/variance_optimization.md` | 그림 | 「서로 독립이면」이라는 단서에 상을 준 까닭 | `—` | `ea__vo_markowitz.png` | — | 대기 |
| 56 | `ch07/expectation_variance/distribution_table_variance.md` | 그림 | 손에게 좋은 공식이 기계에게는 나쁜 공식이다 | `—` | `ev__welford.png` | — | 대기 |
| 57 | `ch07/expectation_variance/expectation.md` | 그림 | 확률보다 기댓값이 먼저 인쇄되었다 | `ch03/rv/expectation.md` | `ev__huygens1657.png` | Wikimedia Commons, PD-old-100 | 대기 |
| 58 | `ch07/expectation_variance/expectation_coin_and_dice.md` | 그림 | 「공정하다」를 이만 육천 번 던져 재어 본 사람 | `—` | `ev__weldon1906.png` | Wikimedia Commons, PD-US | 대기 |
| 59 | `ch07/expectation_variance/indicator_linearity.md` | AI | 점수는 더해서 나오지만 믿음은 거기서 나오지 않는다 | `—` | `ev__ai_indicator_dep.png` | — | 대기 |
| 60 | `ch07/expectation_variance/indicator_linearity.md` | 그림 | 섞었다고 믿은 날 | `—` | `ev__lottery1969.png` | Wikimedia Commons, PD-USGov | 대기 |
| 61 | `ch07/expectation_variance/indicator_variance.md` | AI | 답을 여러 개 모아도 줄지 않는 몫 | `—` | `ev__ai_ensemble_floor.png` | — | 대기 |
| 62 | `ch07/expectation_variance/indicator_variance.md` | 그림 | 오리를 세지 않고 오리 수를 세는 법 | `—` | `ev__lincoln1930.png` | — | 대기 |
| 63 | `ch07/expectation_variance/variance.md` | 그림 | 표준편차가 아니라 분산에 이름을 붙인 까닭 | `ch03/rv/variance_covariance.md` | `ev__fisher1918_variance.png` | — | 대기 |
| 64 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 금 같은 정리 | `ch03/limits/lln.md` | `lln__arsconjectandi1713.png` | Wikimedia Commons, PD-old-100 | 대기 |
| 65 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 부등식 하나로 밀어붙인 사람 | `ch03/limits/lln.md` | `lln__stamp_russia2021.png` | Wikimedia Commons, PD | 대기 |
| 66 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 바늘로 잰 원주율 | `ch03/limits/lln.md` | `lln__buffon_statue.png` | Wikimedia Commons, 촬영 Celette, $2012$, CC BY-SA 4.0 | 대기 |
| 67 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 카드 점을 치다 나온 계산법 | `ch03/limits/lln.md` | `lln__fermiac.png` | 로스앨러모스 국립연구소, Wikimedia Commons, PD-USGov | 대기 |
| 68 | `ch07/normal_distribution/binomial_to_normal.md` | 그림 | 반 칸에 사람 이름이 붙은 자리 | `—` | `nd__yates_gain.png` | — | 대기 |
| 69 | `ch07/normal_distribution/central_limit_theorem.md` | 그림 | 종 모양이 활자로 놓인 쪽 | `ch03/limits/clt.md` | `clt__doctrine1756_p243.png` | Internet Archive 원문 스캔, PD-old-100 | 대기 |
| 70 | `ch07/normal_distribution/normal_applications.md` | 그림 | 같은 높이로 튀었는데 한쪽만 이상이다 | `—` | `nd__shewhart1927_p726.png` | — | 대기 |
| 71 | `ch07/normal_distribution/normal_distribution.md` | 그림 | 지폐에 실린 확률밀도함수 | `ch04/continuous_distributions/normal.md` | `nd__note.png` | — | 대기 |
| 72 | `ch07/normal_distribution/normal_distribution.md` | 그림 | 구슬이 쌓아 올린 종 모양 | `ch04/continuous_distributions/normal.md` | `nd__galton.png` | Wikimedia Commons, PD | 대기 |
| 73 | `ch07/normal_distribution/normal_sum_percentile.md` | 그림 | 분산이 같은데 모양이 다르다 | `—` | `nd__lindeberg.png` | — | 대기 |
| 74 | `ch07/normal_distribution/normal_sum_percentile.md` | AI | 층을 쌓을 때 $\sqrt n$ 으로 나누는 까닭 | `—` | `—` | — | 대기 |
| 75 | `ch07/normal_distribution/normal_symmetry.md` | 그림 | 대칭을 눈금으로 새겨 둔 종이 | `—` | `nd__hazen.png` | — | 대기 |
| 76 | `ch07/normal_distribution/normal_symmetry.md` | AI | 눈금을 맞추지 않으면 큰 눈금만 보게 된다 | `—` | `—` | — | 대기 |
| 77 | `ch07/normal_distribution/repeated_trial_normal.md` | 그림 | 그 표를 만든 사람들은 음수를 다룰 줄 몰랐다 | `—` | `nd__tables.png` | — | 대기 |
| 78 | `ch07/random_variable/continuous_random_variable.md` | AI | 상수를 구하지 못한 채 쓰는 밀도 | `ch03/rv/continuous.md` | `rv__crv_ai_ratio.png` | — | 대기 |
| 79 | `ch07/random_variable/continuous_random_variable.md` | 그림 | 대칭인데 평균이 없는 밀도 | `ch03/rv/continuous.md` | `rv__crv_cauchy.png` | — | 대기 |
| 80 | `ch07/random_variable/cumulative_distribution.md` | 그림 | 잡화상이 만든 첫 계단 | `ch03/rv/pmf_pdf_cdf.md` | `rv__graunt1662_p62.png` | — | 대기 |
| 81 | `ch07/random_variable/geometric_probability.md` | 그림 | 세 답을 나란히 적고 물음이 틀렸다고 한 쪽 | `—` | `rv__bertrand1907_p5.png` | Wikimedia Commons, PD-US-expired | 대기 |
| 82 | `ch07/random_variable/probability_density_applications.md` | 그림 | 넓이를 지키려고 위도를 눌러 놓은 지도 | `—` | `rv__pda_equalarea1921.png` | — | 대기 |
| 83 | `ch07/random_variable/probability_density_applications.md` | AI | 공 위에 점을 고르게 뿌리려면 위도를 버려야 한다 | `—` | `rv__pda_ai_sphere.png` | — | 대기 |
| 84 | `ch07/random_variable/probability_density_function.md` | AI | 어느 자 위에서 균등하게 뽑을 것인가 | `—` | `rv__pdf_ai_logscale.png` | — | 대기 |
| 85 | `ch07/random_variable/probability_density_function.md` | 그림 | 측량 오차를 따지다 나온 분포 | `—` | `rv__pdf_helmert.png` | — | 대기 |
| 86 | `ch07/random_variable/probability_density_graph.md` | 그림 | 눈금을 바꾸면 보이는 직선 | `—` | `rv__pdg_pareto.png` | — | 대기 |
| 87 | `ch07/random_variable/probability_mass_function.md` | 그림 | 질량이 밀도로 건너가는 자리 | `ch03/rv/pmf_pdf_cdf.md` | `rv__simpson1757_p72.png` | Internet Archive 스캔, PD-old | 대기 |
| 88 | `ch07/random_variable/probability_mass_function.md` | AI | 꼬리를 자르면 확률이 아니게 된다 | `ch03/rv/pmf_pdf_cdf.md` | `rv__pmf_ai_renorm.png` | — | 대기 |
| 89 | `ch07/random_variable/probability_mass_function.md` | AI | 합이 $1$ 이라고 해서 확률인 것은 아니다 | `ch03/rv/pmf_pdf_cdf.md` | `—` | — | 대기 |
| 90 | `ch07/random_variable/random_variable.md` | 그림 | 주사위 셋의 합을 칠백 해 전에 적어 둔 표 | `ch03/rv/discrete.md` | `rv__devetula1534.png` | — | 대기 |
| 91 | `ch07/random_variable/random_variable_geometry.md` | 그림 | 「그때에야 비로소 셀 수 있다」 | `—` | `rv__crofton1868_p181.png` | Internet Archive 스캔, PD-old | 대기 |
| 92 | `ch07/random_variable/random_variable_process.md` | 그림 | 시 두 만 자를 손으로 세어 얻은 두 수 | `—` | `rv__markov1913_p153.png` | Internet Archive 스캔, PD-old | 대기 |
| 93 | `ch07/random_variable/uniform_distribution.md` | 그림 | 고르게 뽑으려고 돌을 깎았다 | `ch04/continuous_distributions/uniform.md` | `rv__kleroterion.png` | 촬영 Zde, Wikimedia Commons, CC BY-SA 4.0 | 대기 |
| 94 | `ch07/random_variable/uniform_distribution.md` | AI | 난수 하나로 다음 낱말을 고르는 법 | `ch04/continuous_distributions/uniform.md` | `rv__uni_ai_inverse.png` | — | 대기 |
| 95 | `ch07/random_walk/conditional_sequential.md` | 그림 | 가장 센 패턴이 없는 놀이 | `—` | `rw__penney.png` | — | 대기 |
| 96 | `ch07/random_walk/first_step_analysis.md` | 그림 | 「밑천이 무한하다면」이라고 적어 놓고 넘어간 쪽 | `—` | `fs__eb1911_p771.png` | Wikimedia Commons · 저작권 없음 | 대기 |
| 97 | `ch07/random_walk/first_step_analysis.md` | AI | 기계는 이 점화식을 풀지 않고 외워 넣는다 | `—` | `—` | — | 대기 |
| 98 | `ch07/random_walk/graph_walk.md` | 그림 | 지나온 길은 묻지 않는다고 적어 둔 놀이판 | `—` | `gw__gyanchaupar.png` | 사진 Nomu420 · CC BY-SA 3.0 · Wikimedia Commons | 대기 |
| 99 | `ch07/random_walk/graph_walk.md` | AI | 링크를 따라 걷는 사람이 매기는 순위 | `—` | `—` | — | 대기 |
| 100 | `ch07/random_walk/lattice_walk.md` | 그림 | 모눈종이 위에 연필로 그린 임의 보행 | `—` | `rw__perrin1916_p115.png` | Internet Archive, PD-US-expired | 대기 |
| 101 | `ch07/random_walk/line_walk.md` | 그림 | 취한 사람은 집에 오고 취한 새는 못 온다 | `—` | `rw__polya.png` | Wikimedia Commons, 촬영 Thane Plambeck, CC BY 2.0 | 대기 |
| 102 | `ch07/random_walk/trial_state.md` | 그림 | 홀짝이 매번 뒤집히는 항아리로 물리학의 다툼을 가라앉히 | `—` | `rw__ehrenfest.png` | — | 대기 |
| 103 | `ch07/statistical_estimation/population_sample.md` | 그림 | 돌아온 비행기만 세었다 | `ch01/classical/population_sample.md` | `se__wald1943.png` | — | 대기 |
| 104 | `ch07/statistical_estimation/population_sample.md` | 그림 | 눈을 가리면 고르게 섞이는가 | `ch01/classical/population_sample.md` | `se__draft_lottery.png` | — | 대기 |
| 105 | `ch07/statistical_estimation/population_sample.md` | 그림 | 사람을 똑같은 그림으로 세면 | `ch01/classical/population_sample.md` | `se__stamp_india1977.png` | Wikimedia Commons, GODL-India | 대기 |
| 106 | `ch07/statistical_estimation/proportion_difference.md` | 그림 | 이백삼십팔만 명에게 물어보고 틀렸다 | `ch05/applications/diff_proportions.md` | `se__truman1948.png` | Wikimedia Commons, PD-US-no notice | 대기 |
| 107 | `ch07/statistical_estimation/proportion_difference.md` | 그림 | 두 병동의 수가 갈라져 있었다 | `ch05/applications/diff_proportions.md` | `se__semmelweis1861_p3.png` | — | 대기 |
| 108 | `ch07/statistical_estimation/sampling_estimation.md` | 그림 | 「확률」이라 부르지 않으려고 | `—` | `se__neyman.png` | Wikimedia Commons, 촬영 Konrad Jacobs, 오버볼파흐 수학연구소, CC BY-SA 2 | 대기 |
| 109 | `ch07/statistical_estimation/standard_error.md` | 그림 | 맥주 회사가 이름을 쓰지 못하게 했다 | `ch05/applications/standard_error.md` | `se__student1908_p1.png` | — | 대기 |
| 110 | `ch09/game_theory/constant_sum_game.md` | 그림 | 신분증 사진 한 장과 미니맥스 정리 | `—` | `gt__vonneumann.png` | Wikimedia Commons, PD-LosAlamos | 대기 |
| 111 | `ch09/game_theory/dominant_strategy.md` | 그림 | 스물여섯 쪽짜리 학위논문 | `—` | `gt__nash.png` | Wikimedia Commons, 촬영 Peter Badge / Typos1, CC BY-SA 3.0 | 대기 |
| 112 | `ch09/game_theory/game_theory.md` | 그림 | 두 곡선이 만나는 점, 내시보다 백열두 해 먼저 | `—` | `gt__cournot1838_fig2.png` | — | 대기 |
| 113 | `ch09/game_theory/mixed_strategy.md` | 그림 | 「얼마나 더 자주인지는 계산이 알려 주지 않는다」 | `—` | `gt__montmort1713_p409.png` | — | 대기 |
| 114 | `ch09/game_theory/payoff_utility.md` | 그림 | 곱하지 않은 목록 | `—` | `gt__bentham_ch4.png` | — | 대기 |

합계 **114** 개 — 옮김 **1**, 대기 **113**
