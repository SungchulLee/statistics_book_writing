# 「그림으로 보는 통계학」·「AI로 보는 통계학」 — 옮긴 상자와 이미지 출처

`high_school_math` 전체에 있는 상자 **521 개**를 이 책으로 옮기는 작업의 대장이다.
`docs/` 밖에 두어 빌드에 끼지 않는다.

처음에는 그 책 10장(확률과 통계 = `ch07`·`ch09`)의 **114 개**만 옮길 생각이었으나,
나머지 장에도 이 책에 들어갈 상자가 많다. 「AI로 보는 수학」은 거의 모두 기계학습
이야기이고 기계학습은 통계학이므로, 회귀(13장)·정칙화(18장)·로지스틱(19장)·
다중분류(20장)·생존분석(21장)에 제자리가 있다. 그래서 **521 개 전부**를 하나씩
따져 보기로 하고 대장을 늘렸다.

**이 파일을 두는 까닭** — 옮긴 이미지는 대부분 우표·지폐·박물관 사진이고 CC0·PD·CC BY
가 섞여 있다. 나중에 이 책을 공개하거나 출판할 일이 생기면 **이 파일만 훑으면** 저작권
점검이 끝나도록 한다. 원본 책의 `STAMP_SOURCES.md` 와 같은 구실이다.

⚠️ **라이선스는 틀 이름으로 판단한다.** 「Commons 에 있고 PD 라고 적혀 있다」로는 모자라다.
자세한 기준은 `high_school_math/STAMP_SOURCES.md` 를 따른다.
아직 옮기지 않은 줄의 라이선스 칸은 원본 그림 설명의 `*(…)*` 를 기계로 긁어 온 것이므로,
**옮길 때 원문을 다시 확인한다.**

## 옮길 때 지키는 것

1. 연재 이름 **수학 → 통계학**
2. 그림 경로 `figures/` → `./img/` (이 책의 관례), 그림 파일도 함께 옮긴다
3. **그림 설명의 출처·라이선스 문구는 글자 그대로 옮긴다**
4. 원본 쪽의 정리/정의 번호를 부르는 문장은 **옮긴 쪽의 번호로 다시 맞춘다**.
   같은 것이 이 책에 없으면 그 문장을 홀로 서게 고쳐 쓴다
5. 이 책에 없는 쪽으로 가는 링크는 **빼거나** 사실만 남긴다
6. 자리는 그 상자가 이야기하는 정의/정리 **바로 뒤**(정리면 증명 뒤), 보기 앞.
   번호는 옮긴 쪽마다 $1$ 부터 다시 센다
7. 제자리가 없는 상자는 **억지로 넣지 않고** `보류` 로 적고 까닭을 한 줄 남긴다

## 진행

| # | 원본 | 상자 | 제목 | 옮긴 곳 | 이미지 | 라이선스 | 상태 |
|---|---|---|---|---|---|---|---|
| 1 | `ch00/equation_setup/arithmetic_geometric_mean.md` | 그림 | 파피루스에 그림으로 남은 같은 사실 | `ch02/spread/descriptive_measures_advanced.md` | `agm__papyrus.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 2 | `ch00/equation_setup/cauchy_schwarz.md` | 그림 | 항등식 한 줄 뒤에 부등식이 따라 나온 쪽 | `ch03/rv/correlation.md` | `cs__cauchy1821_p456.png` | Wikimedia Commons, PD-old-100-expired | **옮김** |
| 3 | `ch00/equation_setup/cauchy_schwarz.md` | AI | 닮은 정도를 $-1$ 과 $1$ 사이에 가두는 것 | `ch12/correlation/pearson.md` | `cs__ai_cosine.png` | — | **옮김** |
| 4 | `ch00/equation_setup/circle_equation.md` | 그림 | 계수가 같을 것, 그리고 각이 직각일 것 | `—` | `ce__fermat1679_p5.png` | 페르마의 해석기하와 원의 방정식 — 붙을 정의·정리가 없다 | 보류 |
| 5 | `ch00/equation_setup/coordinate_setup.md` | 그림 | 돌에 새겨 벽에 걸어 둔 좌표평면 | `—` | `cd__orange_cadastre_wall.png`, `cd__orange_cadastre_detail.png` | 로마 지적도의 좌표 체계 — 통계 내용이 없다 | 보류 |
| 6 | `ch00/equation_setup/coordinate_setup.md` | 그림 | 선분에 이름을 붙인 쪽 | `—` | `cd__descartes.png` | 데카르트 『기하학』의 문자 도입 — 대응하는 통계 내용 없음 | 보류 |
| 7 | `ch00/equation_setup/coordinate_setup.md` | 그림 | 좌표는 「너는 어떻게 생겼느냐」를 묻지 않는다 | `—` | `cd__varignon.png` | 바리뇽 정리 — 순수 평면기하 | 보류 |
| 8 | `ch00/equation_setup/distance_division.md` | 그림 | 가운데가 아닌 곳으로 나누어도 되는 중선정리 | `—` | `dd__stewart.png` | 스튜어트 정리 — 삼각형 기하 | 보류 |
| 9 | `ch00/equation_setup/distance_division.md` | AI | 기계가 배우는 한 걸음은 내분점 하나다 | `—` | `dd__ai_ema.png` | 지수이동평균 — 본문에서 다루는 쪽이 없다 | 보류 |
| 10 | `ch00/equation_setup/inequality_properties.md` | 그림 | 부등호가 처음 인쇄된 쪽 | `—` | `ineq__harriot1631_p10.png` | 부등호 기호의 역사 — 대응하는 통계 내용 없음 | 보류 |
| 11 | `ch00/equation_setup/inequality_setup.md` | 그림 | 푸는 법이 없어 보인다고 적어 두고 답을 냈다 | `—` | `iq__stigler.png` | 선형계획법·단순법 — 그 주제를 다루는 쪽이 없다 | 보류 |
| 12 | `ch00/equation_setup/line_equation.md` | 그림 | 평평함을 $0$ 이 아니라 $\infty$ 로 적은 표지 | `—` | `line__gradient_post.png` | 철도 구배 표기 — 회귀의 기울기 해석과 다른 이야기 | 보류 |
| 13 | `ch00/equation_setup/simultaneous.md` | AI | 가장 잘 맞는 직선은 연립방정식 하나로 끝난다 | `ch13/estimation_inference/least_squares.md` | `—` | — | **옮김** |
| 14 | `ch00/equation_setup/simultaneous.md` | 그림 | 산가지를 늘어놓고 지워 나간 사람들 | `—` | `sim__jiuzhang_fangcheng.png` | 가우스 소거법 — 소거 절차를 다루는 쪽이 없다 | 보류 |
| 15 | `ch00/equation_setup/translation_reflection.md` | 그림 | 바뀌지 않는 것만 기하라고 부르기로 하다 | `—` | `tr__klein1872_p7.png` | 클라인 에를랑겐 강령 — 변환군 기하 | 보류 |
| 16 | `ch00/equation_setup/undetermined.md` | 그림 | 세 시간마다 하나씩만 적어 둔 표 | `—` | `uc__nautical_almanac1767.png` | 항해력의 일차보간 — 통계적 추정이 아니다 | 보류 |
| 17 | `ch00/equation_setup/word_to_equation.md` | 그림 | 삼천오백 년 전의 문제집 | `—` | `wte__rhind.png` | 린드 파피루스의 일차방정식 — 대응하는 통계 내용 없음 | 보류 |
| 18 | `ch00/equation_setup/word_to_equation.md` | 그림 | 두 가지보다 더 같은 것은 없으므로 | `—` | `wte__recorde1557_equals.png` | 등호 기호의 역사 — 대응하는 통계 내용 없음 | 보류 |
| 19 | `ch00/equation_setup/word_to_equation.md` | 그림 | 식을 세우지 않고 푸는 법 | `—` | `wte__jiuzhang_yingbuzu.png` | 영부족 셈법 — 대응하는 통계 내용 없음 | 보류 |
| 20 | `ch01/complex/complex.md` | 그림 | 답이 없어서가 아니라, 길이 그리로만 나 있어서 | `—` | `cx__bombelli1572_title.png` | 복소수의 도입 — 대응하는 통계 내용 없음 | 보류 |
| 21 | `ch01/complex/complex.md` | 그림 | 다리 위에서 칼로 새긴 식 | `—` | `cx__hamilton_plaque.png` | 사원수와 교환법칙 — 대응하는 통계 내용 없음 | 보류 |
| 22 | `ch01/complex/complex_plane.md` | 그림 | 사분면마다 복소수를 하나씩 | `—` | `cxp__stamp_germany1977.png` | 복소평면 — 대응하는 통계 내용 없음 | 보류 |
| 23 | `ch01/complex/complex_powers.md` | 그림 | 금빛 별에 남은 열일곱 등분 | `—` | `cx__gauss_star.png` | 1 의 거듭제곱근·정십칠각형 — 대응하는 통계 내용 없음 | 보류 |
| 24 | `ch01/exp_log/common_log.md` | 그림 | 가수만 새겨 놓고 지표는 사람에게 맡긴 자 | `—` | `el__slide_rule.png` | 계산자의 눈금 — 통계 내용 없음 | 보류 |
| 25 | `ch01/exp_log/common_log.md` | 그림 | 로그표의 앞쪽이 먼저 닳았다 | `ch10/test/gof.md` | `el__newcomb1881_p39.png` | JSTOR Early Journal Content, 공개 자료 | **옮김** |
| 26 | `ch01/exp_log/exp.md` | 그림 | 줄을 왼쪽으로 늘리자 $2^{0}$ 과 $2^{-1}$ 이 정해졌다 | `—` | `el__stifel1544_p237.png` | 지수 약속의 확장 — 대응하는 통계 내용 없음 | 보류 |
| 27 | `ch01/exp_log/exp.md` | 그림 | 지수를 함수로 만든 사람 | `—` | `el__stamp_ussr1957.png` | 오일러와 지수함수 — 대응하는 통계 내용 없음 | 보류 |
| 28 | `ch01/exp_log/exp.md` | 그림 | 거듭제곱이 가장 먼저 쓰인 곳은 빚 문서였다 | `—` | `el__enmetena_cone.png` | 수메르의 복리 — 지수법칙 이야기 | 보류 |
| 29 | `ch01/exp_log/exp_log.md` | 그림 | 로그표를 두 벌 만들지 않아도 되었던 까닭 | `—` | `el__hutton1811_p104.png` | 밑변환 상수 — 붙을 자리가 없다 | 보류 |
| 30 | `ch01/exp_log/exp_log.md` | AI | 같은 성적을 두 가지 단위로 적는다 | `ch20/optimization/cross_entropy.md` | `—` | — | **옮김** |
| 31 | `ch01/exp_log/exponential_ratio.md` | 그림 | 이 식의 분모에 이름을 붙인 쪽 | `ch20/softmax_regression/softmax.md` | `er__gibbs1902_p33.png` | Internet Archive 스캔, PD-old | **옮김** |
| 32 | `ch01/exp_log/exponential_ratio.md` | AI | 점수를 매길 수 없는 것을 둘씩 견주어 줄 세우기 | `ch19/logistic_regression/logit.md` | `—` | — | **옮김** |
| 33 | `ch01/exp_log/inverse_symmetry.md` | 그림 | 로그가 떨어지는 수를 밑으로 골랐다 | `—` | `el__pogson1856_p14.png` | 별 등급 척도의 밑 — 자료의 로그변환과 다른 이야기 | 보류 |
| 34 | `ch01/exp_log/log.md` | 그림 | 곱셈을 덧셈으로 바꾸려고 만든 책 | `—` | `el__napier1618.png` | 네이피어의 로그표 — 통계 내용 없음 | 보류 |
| 35 | `ch01/exp_log/log.md` | 그림 | 표에서 함수로 | `—` | `el__introductio1748.png` | 오일러가 로그를 역함수로 세운 자리 — 대응하는 통계 내용 없음 | 보류 |
| 36 | `ch01/exp_log/log.md` | 그림 | 쓰는 법을 빼고 표만 찍은 책 | `—` | `el__burgi1620_title.png` | 뷔르기의 표 — 대응하는 통계 내용 없음 | 보류 |
| 37 | `ch01/exp_log/log.md` | AI | 곱을 합으로 바꾸는 일이 지금도 하는 일 | `ch06/mle/log_likelihood.md` | `el__ai_logprob.png` | — | **옮김** |
| 38 | `ch01/functions/composite.md` | 그림 | 흩트리는 손놀림인데, 되풀이하면 제자리로 돌아온다 | `—` | `cmp__rubik.png` | 치환군과 루빅 큐브 — 대응하는 통계 내용 없음 | 보류 |
| 39 | `ch01/functions/function.md` | 그림 | 어긋남을 결함이 아니라 정의로 삼은 사람 | `ch00/math/logic_sets.md` | `fn__cantor.png` | Wikimedia Commons, PD-old | **옮김** |
| 40 | `ch01/functions/inverse.md` | 그림 | $f^{-1}$ 을 쓰면서 그 자리에 달아 둔 각주 | `—` | `inv__herschel1813_p10.png` | 역함수 표기의 역사 — 대응하는 통계 내용 없음 | 보류 |
| 41 | `ch01/functions/inverse.md` | AI | 되돌아오게 만들어 두어야 하는 길 | `—` | `—` | 토큰화 — 그것을 다루는 쪽이 없다 | 보류 |
| 42 | `ch01/functions/irrational.md` | AI | 요즘 언어모델이 꼭 $\sqrt{d}$ 로 나누는 까닭 | `ch03/rv/variance_covariance.md` | `—` | — | **옮김** |
| 43 | `ch01/functions/irrational.md` | 그림 | 제곱과 제곱근을 한 쪽에 나란히 찍어 둔 책 | `—` | `irr__barlow1914_p30.png` | 수표(바로 표) — 역함수 표기 이야기 | 보류 |
| 44 | `ch01/functions/proportion.md` | 그림 | 비례상수를 재지 않고 정해 버렸다 | `—` | `prp__vitruvius_hodometer.png` | 비트루비우스의 거리 재는 수레 — 통계 내용 없음 | 보류 |
| 45 | `ch01/functions/rational_function.md` | 그림 | 아무도 재지 않은 수를 적어 넣은 열 | `—` | `rf__boyle1662_p60.png` | 보일의 법칙과 반비례함수 — 알맹이가 통계가 아니다 | 보류 |
| 46 | `ch01/functions/set.md` | 그림 | 무엇이든 모으면 집합인가 | `—` | `fn__russell1907.png` | 러셀 역설 — logic_sets.md 가 그 물음을 세우지 않는다 | 보류 |
| 47 | `ch01/functions/set.md` | 그림 | 식당 창문에 끼운 세 개의 원 | `ch03/probability/axioms.md` | `fn__venn_window.png` | Wikimedia Commons, 촬영 Schutz, CC BY-SA 2.5 | **옮김** |
| 48 | `ch01/matrix/linear_transformation.md` | 그림 | 격자를 비틀었더니 다른 종이 되었다 | `—` | `lt__thompson1917_p748.png` | 다시 톰프슨의 변환 이론 — 생물 형태학 | 보류 |
| 49 | `ch01/matrix/linear_transformation.md` | AI | 층 하나가 곧 일차변환이다 | `ch20/optimization/optimization.md` | `—` | — | **옮김** |
| 50 | `ch01/matrix/matrix.md` | 그림 | 정리 1 이 활자로 놓인 자리 | `—` | `mx__cayley1858_p23.png` | 케일리–해밀턴 정리 — 이 책에서 쓰지 않는다 | 보류 |
| 51 | `ch01/matrix/matrix_operations.md` | 그림 | 위첨자는 거듭제곱이 아니라고 적어 두어야 했던 쪽 | `—` | `mo__cramer1750_p657.png` | 크라메르 법칙과 첨자 표기 — 대응하는 통계 내용 없음 | 보류 |
| 52 | `ch01/polynomial/expand_factor.md` | 그림 | 넓이로 적은 완전제곱 | `—` | `ef__khwarizmi.png` | 알고리트미의 완전제곱 — 대수사 | 보류 |
| 53 | `ch01/polynomial/higher_equations.md` | 그림 | 비밀로 지키던 풀이법 | `—` | `he__arsmagna.png` | 삼차방정식 해법 — 대응하는 통계 내용 없음 | 보류 |
| 54 | `ch01/polynomial/higher_equations.md` | 그림 | 공식이 없다는 것을 증명한 사람 | `—` | `he__stamp_norway1929.png` | 아벨–루피니 정리 — 대응하는 통계 내용 없음 | 보류 |
| 55 | `ch01/polynomial/higher_equations.md` | 그림 | 시간이 없다 | `—` | `he__galois_letter.png` | 갈루아 이론 — 대응하는 통계 내용 없음 | 보류 |
| 56 | `ch01/polynomial/polynomial.md` | 그림 | 산가지를 늘어놓은 자리가 곧 계수였다 | `—` | `pl__qin1247_kaifang.png` | 조립제법 — 대응하는 통계 내용 없음 | 보류 |
| 57 | `ch01/polynomial/quadratic.md` | 그림 | 여섯 개였던 공식 | `—` | `qd__stamp_ussr1983.png` | 이차방정식의 여섯 꼴 — 대응하는 통계 내용 없음 | 보류 |
| 58 | `ch01/polynomial/quadratic_inequality.md` | 그림 | 운전면허 교본에 실린 이차부등식 | `—` | `qd__highwaycode.png`, `qd__stopping.png` | 정지거리의 이차식 — 통계 내용 없음 | 보류 |
| 59 | `ch01/polynomial/simultaneous_quadratic.md` | 그림 | 몇 개나 나올지는 풀기 전에 알 수 있다 | `—` | `sq__bezout.png` | 베주 정리 — 대응하는 통계 내용 없음 | 보류 |
| 60 | `ch01/polynomial/simultaneous_quadratic.md` | 그림 | 기호가 없어서 곡선으로 풀었다 | `—` | `he__khayyam1851_pxviij.png` | 하이얌의 원뿔곡선 풀이 — 대응하는 통계 내용 없음 | 보류 |
| 61 | `ch01/polynomial/simultaneous_quadratic.md` | AI | 거리를 재지 않고 가까운 쪽을 고르는 법 | `ch19/evaluation/lda_qda_classification.md` | `he__centroid.png` | — | **옮김** |
| 62 | `ch01/trig_functions/sine_cosine_laws.md` | 그림 | 사인표의 조상은 현의 표였다 | `—` | `tr__ptolemy_chords.png` | 프톨레마이오스의 현의 표 — 대응하는 통계 내용 없음 | 보류 |
| 63 | `ch01/trig_functions/sine_cosine_laws.md` | 그림 | 각을 하나도 재지 않고 넓이를 구한다 | `—` | `tr__aeolipile.png` | 헤론의 공식 — 대응하는 통계 내용 없음 | 보류 |
| 64 | `ch01/trig_functions/trig_addition.md` | 그림 | 로그가 나오기 전, 가장 빠른 곱셈 | `—` | `tr__tycho.png` | 프로스타파이레시스 — 대응하는 통계 내용 없음 | 보류 |
| 65 | `ch01/trig_functions/trig_addition.md` | 그림 | $\pi$ 라는 글자가 지금 뜻으로 인쇄된 쪽 | `—` | `trig__jones1706_p263.png` | 머신 공식과 π 기호 — 대응하는 통계 내용 없음 | 보류 |
| 66 | `ch01/trig_functions/trig_equations_inequalities.md` | 그림 | 시각마다 한 번씩 삼각방정식을 푼 자리 | `—` | `trig_equations_inequalities__sundial.png` | 해시계 시각선 — 대응하는 통계 내용 없음 | 보류 |
| 67 | `ch01/trig_functions/trig_max_min.md` | 그림 | 손잡이가 셋뿐인 기계 | `—` | `tg__harmonograph1909_p8.png` | 하모노그래프 — 대응하는 통계 내용 없음 | 보류 |
| 68 | `ch01/trig_functions/trig_max_min.md` | AI | 「몇 시에 가장 붐비는가」를 기계가 배우는 법 | `ch13/splines_gams/generalized_additive_models.md` | `—` | — | **옮김** |
| 69 | `ch01/trig_functions/trig_sums_sequences.md` | 그림 | 틈을 늘릴수록 뾰족해지는 봉우리 | `—` | `tg__grating.png` | 회절격자와 등차 위상합 — 대응하는 통계 내용 없음 | 보류 |
| 70 | `ch01/trig_functions/trig_values_double_angle.md` | 그림 | 세 변에 이름을 붙여 놓은 표 | `—` | `tg__rheticus1596.png` | 레티쿠스의 삼각비 표 — 대응하는 통계 내용 없음 | 보류 |
| 71 | `ch01/trig_functions/trig_values_double_angle.md` | AI | 낱말의 자리를 각도로 적는 법 | `—` | `—` | 회전 위치부호(RoPE) — 위치부호를 다루는 쪽이 없다 | 보류 |
| 72 | `ch02/sequence_limits/descent_recurrence.md` | 그림 | 방정식을 풀려다 내리막을 찾아낸 쪽 | `ch20/optimization/optimization.md` | `sl__cauchy1847_p536.png` | Internet Archive 스캔, PD-old | **옮김** |
| 73 | `ch02/sequence_limits/geometric_limit.md` | 그림 | 반을 가도 또 반이 남는다 | `—` | `sl__zeno.png` | 제논의 역설과 등비급수 — 대응하는 통계 내용 없음 | 보류 |
| 74 | `ch02/sequence_limits/limit_general.md` | 그림 | lim 이라는 세 글자가 놓이던 쪽 | `ch00/math/limits.md` | `sl__lhuilier_p24.png` | Gallica/BnF, 공개 자료 | **옮김** |
| 75 | `ch02/sequence_limits/monotone_convergence.md` | 그림 | 이 정리를 증명하지 못해 실수를 다시 세운 사람 | `—` | `sl__dedekind1872_p29.png` | 단조수렴정리 — 그 정리를 세우는 쪽이 없다 | 보류 |
| 76 | `ch02/sequence_limits/recurrence.md` | 그림 | 변과 대각을 번갈아 키워 $\sqrt2$ 에 다가가기 | `—` | `sl__theon1878_p44.png` | 테온의 점화식 — 대응하는 통계 내용 없음 | 보류 |
| 77 | `ch02/sequence_limits/squeeze.md` | 그림 | 구하지 못하는 값을 열여덟과 스물 사이에 가두다 | `—` | `sl__aristarchus_p377.png` | 조임정리 — 한 줄과 연습문제로만 다룬다 | 보류 |
| 78 | `ch02/sequences/arithmetic_sequence.md` | 그림 | 백 번 더하지 않은 아이 | `—` | `sq__gauss_statue.png` | 등차수열 합의 일화 — 대응하는 통계 내용 없음 | 보류 |
| 79 | `ch02/sequences/compound_interest.md` | 그림 | 이자를 잘게 쪼개다 만난 수 | `ch00/math/limits.md` | `sq__bernoulli1690_p222.png` | Wikimedia Commons, PD-old-100-expired | **옮김** |
| 80 | `ch02/sequences/compound_interest.md` | 그림 | 오십 해마다 셈을 지우라고 새긴 종 | `—` | `ci__liberty_bell.png` | 희년과 연금 현재가치 — 통계 내용 없음 | 보류 |
| 81 | `ch02/sequences/compound_interest.md` | 그림 | 한 살짜리의 연금이 열 살짜리보다 쌌다 | `ch21/introduction/censoring_types.md` | `sq__halley1693_p603.png` | JSTOR Early Journal Content, 공개 자료 | **옮김** |
| 82 | `ch02/sequences/consecutive_integer_sums.md` | 그림 | 수를 모양으로 보던 시절의 삼각수 | `—` | `ci__nicomachus_p241.png` | 삼각수 — 수론 | 보류 |
| 83 | `ch02/sequences/counting_discrete.md` | 그림 | 가우스 기호를 $E$ 로 적던 쪽 | `—` | `cd__legendre1830_p387.png` | 바닥함수 표기 — 대응하는 통계 내용 없음 | 보류 |
| 84 | `ch02/sequences/divisors.md` | 그림 | 삼백 년을 버틴 열한 개짜리 목록 — 메르센, $1644$ | `—` | `sq__mersenne1644_praef19.png` | 메르센 소수·완전수 — 수론 | 보류 |
| 85 | `ch02/sequences/geometric_sequence.md` | 그림 | 묘비에 새겨진 엉뚱한 나선 | `—` | `gs__bernoulli_tomb.png` | 로그나선과 등비수열 — 대응하는 통계 내용 없음 | 보류 |
| 86 | `ch02/sequences/geometric_sequence.md` | 그림 | 한 옥타브에 등비중항 열한 개를 끼워 넣기 | `—` | `sq__zhu1584_pipes.png` | 십이평균율 — 대응하는 통계 내용 없음 | 보류 |
| 87 | `ch02/sequences/induction.md` | 그림 | 「그리고 이렇게 끝없이, 언제나 제13명제를 되풀이하여」 | `—` | `sq__maurolico1575_p7.png` | 수학적 귀납법의 역사 — 대응하는 통계 내용 없음 | 보류 |
| 88 | `ch02/sequences/periodic_sequence.md` | 그림 | 주기 $235$ 를 청동에 새겨 둔 사람들 | `—` | `sq__antikythera.png` | 안티키테라 기계 — 대응하는 통계 내용 없음 | 보류 |
| 89 | `ch02/sequences/point_sequence.md` | 그림 | 배의 자리를 앞 자리에서 셈해 찍어 가던 그림 | `—` | `ps__bowditch1802_p93.png` | 추측항법의 점화식 — 확률적 누적오차가 아니다 | 보류 |
| 90 | `ch02/sequences/procedure_count.md` | 그림 | 여백에 달마다 하나씩 적어 내려간 표 | `—` | `pc__liberabaci_f124r.png` | 피보나치와 재귀 호출 — 대응하는 통계 내용 없음 | 보류 |
| 91 | `ch02/sequences/recurrence.md` | 그림 | 우표에 인쇄된 점화식 | `—` | `seq__stamp_serbia2023.png` | 피보나치·황금비 — 대응하는 통계 내용 없음 | 보류 |
| 92 | `ch02/sequences/recurrence.md` | 그림 | 핀을 만들듯 로그를 만들다 | `—` | `sq__babbage1846_p195.png` | 드 프로니의 분업 — 대응하는 통계 내용 없음 | 보류 |
| 93 | `ch02/sequences/sequence_extremum.md` | 그림 | 한 사람은 하룻밤, 한 사람은 일 분 | `—` | `sq__kadane.png` | 최대부분합 알고리즘 — 대응하는 통계 내용 없음 | 보류 |
| 94 | `ch02/sequences/similar_figure_sequence.md` | 그림 | 닮음을 한 번에 하나씩 그려 주는 기계 | `—` | `sf__scheiner1631_p29.png` | 판토그래프와 닮음비 — 대응하는 통계 내용 없음 | 보류 |
| 95 | `ch02/series/double_sums.md` | 그림 | 시그마가 델타의 짝으로 태어난 쪽 | `—` | `ds__euler1755_p27.png` | Σ 기호의 역사 — 대응하는 통계 내용 없음 | 보류 |
| 96 | `ch02/series/partial_fractions.md` | 그림 | 늘 갈라진다는 것을 그는 믿지 않았다 | `—` | `pf__leibniz1702_p218.png` | 부분분수 분해 — 대응하는 통계 내용 없음 | 보류 |
| 97 | `ch02/series/partial_sum_general_term.md` | 그림 | 빼기를 거꾸로 돌려 표를 찍으려던 기계 | `—` | `ps__babbage1864_frontis.png` | 배비지의 차분기관 — 대응하는 통계 내용 없음 | 보류 |
| 98 | `ch02/series/series_telescoping.md` | 그림 | 합을 세로선으로 세우고 항을 그 사이의 차로 읽은 그림 | `—` | `tel__stirling1730_p19.png` | 망원합의 그림 — 대응하는 통계 내용 없음 | 보류 |
| 99 | `ch02/series/shift_subtract.md` | 그림 | 「그러므로, 빼면」 한 줄로 닫은 이천이백 해 전의 합 | `—` | `ss__archimedes1897_p250.png` | 아르키메데스의 등비합 — 대응하는 통계 내용 없음 | 보류 |
| 100 | `ch02/series/sum_formulas.md` | 그림 | 대학에 가 본 적 없는 계산장인 | `—` | `sm__faulhaber.png` | 파울하버 공식·베르누이 수 — 대응하는 통계 내용 없음 | 보류 |
| 101 | `ch02/series/sum_formulas.md` | 그림 | 같은 수가 지구 반대편에서 한 해 먼저 찍혀 나왔다 | `—` | `sm__seki1712_bernoulli.png` | 세키의 베르누이 수 표 — 대응하는 통계 내용 없음 | 보류 |
| 102 | `ch02/series_limits/geometric_series.md` | 그림 | 합은 적혀 있고 조건은 빠져 있던 쪽 | `ch00/math/limits.md` | `srl__euler1748_p47.png` | Wikimedia Commons, PD-old-100-expired | **옮김** |
| 103 | `ch02/series_limits/geometric_series.md` | 그림 | 끝이 없는 빚에 값이 매겨지는 까닭 | `—` | `srl__southsea1784.png` | 영구채의 현재가치 — 통계 정리에 붙지 않는다 | 보류 |
| 104 | `ch02/series_limits/series_convergence.md` | 그림 | 야자잎에 적힌 원주율 | `—` | `srl__yuktibhasa_whish.png` | 마다바 급수 — 대응하는 통계 내용 없음 | 보류 |
| 105 | `ch02/series_limits/series_convergence.md` | 그림 | 제목이 「분수의 덧셈」인 책 | `—` | `srl__mengoli1650.png` | 조화급수의 발산 — 대응하는 통계 내용 없음 | 보류 |
| 106 | `ch02/series_limits/series_integral_squeeze.md` | 그림 | 「신비로 가득하다」는 비난에 칠백육십삼 쪽으로 답한 책 | `—` | `srl__maclaurin_integral_test.png` | 적분판정법·매클로린 — 대응하는 통계 내용 없음 | 보류 |
| 107 | `ch02/series_limits/similar_figures_series.md` | 그림 | 넓이를 구하려던 그림이 아니었다 | `—` | `srl__sierpinski1915_p303.png` | 시에르핀스키 삼각형 — 대응하는 통계 내용 없음 | 보류 |
| 108 | `ch03/differentiation/angle_in_figures.md` | 그림 | 길이는 한 번만 재고, 나머지는 모두 각으로 | `—` | `angle__snellius.png` | 삼각측량의 각 오차 전파 — 오차전파를 다루는 쪽이 없다 | 보류 |
| 109 | `ch03/differentiation/angle_in_figures.md` | AI | 두 구멍이 있는 공식 | `—` | `—` | 기울기 공식의 정의역 구멍 — 통계 내용 없음 | 보류 |
| 110 | `ch03/differentiation/composite.md` | AI | 역전파는 연쇄법칙을 거꾸로 읽은 것이다 | `ch20/optimization/optimization.md` | `chain__vanishing.png` | — | **옮김** |
| 111 | `ch03/differentiation/composite.md` | 그림 | 같은 수로 싸웠는데 한쪽만 남는다 | `—` | `chain__lanchester1916_p43.png` | 란체스터 N제곱 법칙 — 전투 미분방정식 | 보류 |
| 112 | `ch03/differentiation/continuity_and_composition.md` | 그림 | 「계산의 수고 없이」 접선을 긋는 세 줄짜리 규칙 | `—` | `dif__sluse1673_p5143.png` | 슬뤼즈의 접선 규칙 — 미적분 전사 | 보류 |
| 113 | `ch03/differentiation/deriv_def.md` | 그림 | 밑과 지수가 함께 움직이는 것을 처음 미분한 쪽 | `—` | `dif__bernoulli1697_p125_129.png` | 베르누이의 지수량 미분 — 대응하는 통계 내용 없음 | 보류 |
| 114 | `ch03/differentiation/derivative.md` | 그림 | $dx$ 가 처음 인쇄된 쪽 | `—` | `der__leibniz1684.png` | 라이프니츠 기호사 — 대응하는 통계 내용 없음 | 보류 |
| 115 | `ch03/differentiation/inverse.md` | 그림 | 이십 년 넘게 들키지 않은 두 자리 | `—` | `dv__shanks1873_p45.png` | 샹크스의 π 자릿수 정정 — 대응하는 통계 내용 없음 | 보류 |
| 116 | `ch03/differentiation/param_rate.md` | AI | 푸는 과정을 따라가지 않고 미분하기 | `—` | `—` | 음함수 미분으로 내부 해 미분 — 그 주제의 쪽이 없다 | 보류 |
| 117 | `ch03/differentiation/param_rate.md` | 그림 | 크랭크는 고르게 도는데 피스톤은 그렇지 않다 | `—` | `pr__perry1897_p12_191_192.png` | 기관의 매개변수 미분 — 대응하는 통계 내용 없음 | 보류 |
| 118 | `ch03/differentiation/trig_derivative.md` | 그림 | 도함수라는 말 없이 도함수를 적어 둔 쪽 | `—` | `dv__bhaskara_sine.png` | 바스카라의 사인 증분 — 대응하는 통계 내용 없음 | 보류 |
| 119 | `ch03/graph_extremum/cubic.md` | 그림 | 표지의 이름과 내용을 쓴 사람이 달랐던 책 | `—` | `ge__lhopital1716_p41.png` | 로피탈 교과서사 — 대응하는 통계 내용 없음 | 보류 |
| 120 | `ch03/graph_extremum/cubic.md` | AI | 세 점만 재고도 정확한 넓이 | `—` | `—` | 심프슨 공식(수치적분) — 통계 쪽 자리 없음 | 보류 |
| 121 | `ch03/graph_extremum/cubic_determine.md` | 그림 | 연립방정식을 세우지 않고 답을 바로 적는 법 | `—` | `ge__waring1779_p59.png`, `cubic__waring.png` | 워링·라그랑주 보간 우선권 — 대응하는 통계 내용 없음 | 보류 |
| 122 | `ch03/graph_extremum/cubic_determine.md` | AI | 화면의 곡선이 하필 삼차인 까닭 | `ch13/splines_gams/splines_patsy_examples.md` | `—` | — | **옮김** |
| 123 | `ch03/graph_extremum/cubic_roots_tangent.md` | AI | 풀기 전에 몇 개인지부터 센다 | `—` | `—` | 삼차방정식 실근 개수 판정 — 대응하는 통계 내용 없음 | 보류 |
| 124 | `ch03/graph_extremum/cubic_roots_tangent.md` | 그림 | 부호만 적어 내려간 네 줄 | `—` | `cubic__sturm.png` | 스튀름 사슬 — 대응하는 통계 내용 없음 | 보류 |
| 125 | `ch03/graph_extremum/inflection.md` | 그림 | 기차가 휘청이지 않게 하려면 | `—` | `ge__talbot1901_p1.png` | 철도 완화곡선과 곡률 — 대응하는 통계 내용 없음 | 보류 |
| 126 | `ch03/graph_extremum/intersection.md` | 그림 | 원 하나가 근의 개수를 세어 주던 때 | `—` | `ge__runge1912_p11_12.png` | 룽게의 작도식 근 찾기 — 대응하는 통계 내용 없음 | 보류 |
| 127 | `ch03/graph_extremum/logistic_curve.md` | 그림 | 이 곡선에 이름이 붙던 도판 | `ch19/logistic_regression/logit.md` | `logistic__verhulst1845.png` | Wikimedia Commons, PD-old-100-1923 | **옮김** |
| 128 | `ch03/graph_extremum/optimization.md` | 그림 | 벌은 거의 옳다 | `—` | `optimization__honeycomb.png`, `optimization__cell_angle.png` | 벌집 각의 최적화 — 대응하는 통계 내용 없음 | 보류 |
| 129 | `ch03/graph_extremum/piecewise_abs.md` | AI | 꺾인 점 하나로 버티는 함수 | `—` | `—` | ReLU — 본문에서 다루는 쪽이 없다(ch20 은닉층은 시그모이드다) | 보류 |
| 130 | `ch03/graph_extremum/piecewise_abs.md` | 그림 | 꺾인 점을 세 가지로 갈라 적어 둔 기계책의 각주 | `—` | `piecewise__willis1870_p26.png` | 윌리스의 첨점 분류 — 대응하는 통계 내용 없음 | 보류 |
| 131 | `ch03/graph_extremum/quartic.md` | 그림 | 그림을 붙일 수 없었던 책이 스물여덟을 세는 법 | `—` | `quartic__plucker1839_p230_p253.png`, `quartic__bitangent28.png` | 플뤼커의 이중접선 28 — 대응하는 통계 내용 없음 | 보류 |
| 132 | `ch03/graph_extremum/quartic.md` | AI | 골짜기를 직선으로 덮어 버리기 | `—` | `—` | 사차함수의 볼록 포락선 — 알맹이가 전역최적화다 | 보류 |
| 133 | `ch03/graph_extremum/rational_standard_form.md` | 그림 | 쌍곡선을 S 자로 그린 쪽 | `ch19/logistic_regression/logit.md` | `ge__michaelis1913_p344.png` | Internet Archive · 저작권 없음 | **옮김** |
| 134 | `ch03/graph_extremum/transcendental.md` | 그림 | 「초월」이라는 말이 결판난 쪽 | `—` | `trans__lindemann1882_p223.png` | 린데만의 π 초월성 — 대응하는 통계 내용 없음 | 보류 |
| 135 | `ch03/graph_extremum/transcendental.md` | AI | 떼어 낸 양수 인수가 계산도 살린다 | `ch06/mle/log_likelihood.md` | `—` | — | **옮김** |
| 136 | `ch03/graph_extremum/transcendental_extremum.md` | 그림 | 지수가 $5$ 일 수밖에 없었던 까닭 | `—` | `ge__wien1896_p667.png` | 빈의 변위법칙 — 대응하는 통계 내용 없음 | 보류 |
| 137 | `ch03/graph_sketch/cubic.md` | 그림 | 봉우리와 골짜기가 사라지던 온도 | `—` | `cubic__poynting1920_p194.png` | 반데르발스 임계점 — 대응하는 통계 내용 없음 | 보류 |
| 138 | `ch03/graph_sketch/line.md` | 그림 | 스물넉 점에 그은 두 직선 | `ch13/linear_regression/simple.md` | `line__hubble1929_p172.png` | Internet Archive · 저작권 없음 | **옮김** |
| 139 | `ch03/graph_sketch/log_exp.md` | 그림 | $0$ 에 눕는 곡선을 실제로 재어 그린 쪽 | `ch04/continuous_distributions/exponential.md` | `gs__rutherford1905_p366.png` | Internet Archive 스캔, $1905$ 년 발행 | **옮김** |
| 140 | `ch03/graph_sketch/order.md` | 그림 | 좌표가 생기기 삼백 년 전에 그린 그래프 | `—` | `gs__oresme1482.png` | 오렘의 라티투도 — 자료 그림과 다른 이야기 | 보류 |
| 141 | `ch03/graph_sketch/parabola.md` | 그림 | 철도가 퇴짜 놓은 곡선 | `—` | `parabola__allen1903_p68.png` | 철도 측량의 포물선 — 대응하는 통계 내용 없음 | 보류 |
| 142 | `ch03/graph_sketch/polynomial.md` | AI | 자료를 꼭 맞추면 자료 밖이 달아난다 | `ch18/motivation/overfitting.md` | `—` | — | **옮김** |
| 143 | `ch03/graph_sketch/polynomial.md` | 그림 | 굽히는 자가 곡선을 정하던 때 | `ch13/splines_gams/splines_patsy_examples.md` | `polynomial__ribbands1918_p39.png` | Internet Archive · 저작권 없음 | **옮김** |
| 144 | `ch03/graph_sketch/quartic.md` | 그림 | 들보가 내려앉는 모양은 사차곡선이다 | `—` | `quartic__merriman1909_p138_139.png` | 보의 처짐 곡선 — 대응하는 통계 내용 없음 | 보류 |
| 145 | `ch03/graph_sketch/quartic.md` | AI | 중심을 옮기지 않으면 계산이 무너진다 | `ch13/interaction_polynomial/polynomial_regression.md` | `—` | — | **옮김** |
| 146 | `ch03/graph_sketch/rational.md` | 그림 | 오역으로 마녀가 된 곡선 | `ch07/non_normal/cauchy_lln_failure.md` | `rational__agnesi_versiera_witch.png` | Internet Archive 스캔, PD-old | **옮김** |
| 147 | `ch03/graph_sketch/trig.md` | 그림 | 사인곡선 넷을 줄 하나로 더하던 놋쇠 기계 | `—` | `trig__kelvin1913_tide.png` | 조석 예측기·조화합성 — 대응하는 통계 내용 없음 | 보류 |
| 148 | `ch03/inequality/absolute_inequality_extremum.md` | AI | 기울기가 $0$ 이라고 바닥은 아니다 | `ch00/linalg_regression/square_matrices/positive_definite.md` | `—` | — | **옮김** |
| 149 | `ch03/inequality/absolute_inequality_extremum.md` | 그림 | 같은 쪽으로 늘어서 있기만 하면 | `—` | `ineq__chebyshev.png` | 체비쇼프 합 부등식 — 이름만 같은 다른 부등식 | 보류 |
| 150 | `ch03/inequality/am_gm_cauchy.md` | 그림 | 이름이 셋인 부등식, 풀에 덮인 가운데 사람 | `ch12/correlation/pearson.md` | `ineq__bunyakovsky_grave.png` | Blindpew100 촬영, Wikimedia Commons, CC BY-SA 4.0 | **옮김** |
| 151 | `ch03/inequality/am_gm_cauchy.md` | AI | 판별식이 내놓는 것은 고윳값이었다 | `ch00/linalg_regression/square_matrices/symmetric.md` | `—` | — | **옮김** |
| 152 | `ch03/inequality/convex_tangent.md` | 그림 | 전화국 기술부장이 퇴근하고 쓴 부등식 | `ch07/variance/bessels_correction.md` | `ineq__jensen.png` | Theodor Vilhelm Rieger 촬영, Wikimedia Commons, PD-Denmark50 | **옮김** |
| 153 | `ch03/inequality/convex_tangent.md` | AI | 한 점의 기울기만 보고 전체의 최솟값을 믿는 근거 | `ch19/estimation_inference/algorithms.md` | `convex_tangent__gd.png` | — | **옮김** |
| 154 | `ch03/inequality/deriv_ineq.md` | 그림 | 「예는 들지 않겠다」고 적고 넘어간 자리 | `—` | `ineq__lipschitz.png` | 립시츠 조건과 해의 유일성 — 대응하는 통계 내용 없음 | 보류 |
| 155 | `ch03/inequality/deriv_ineq.md` | AI | 보폭을 얼마로 할 것인가 — 기계가 매일 묻는 물음 | `ch19/estimation_inference/algorithms.md` | `—` | — | **옮김** |
| 156 | `ch03/inequality/impurity_function.md` | 그림 | 묘비에 새긴 로그 하나 | `ch20/optimization/cross_entropy.md` | `impurity__boltzmann_tomb.png` | — | **옮김** |
| 157 | `ch03/inequality/integral_ineq.md` | 그림 | 첫 문장이 「로저스 씨가 보였듯이」로 시작하는 논문 | `—` | `ineq__hoelder1889_p38_39.png` | 횔더 부등식의 이름 — 이 책이 횔더를 쓰지 않는다 | 보류 |
| 158 | `ch03/inequality/integral_ineq.md` | AI | 「평균을 대표하는 하나」는 어디까지 있는가 | `—` | `—` | 적분의 평균값 정리 — 이 책에 없다 | 보류 |
| 159 | `ch03/inequality/mvt_rolle.md` | 그림 | 미적분을 공격한 사람의 이름 | `—` | `iq__rolle.png` | 롤의 정리 내력 — 대응하는 통계 내용 없음 | 보류 |
| 160 | `ch03/limits_continuity/continuity_diff.md` | 그림 | 역이 거짓임을 끝까지 밀어붙인 쪽 | `—` | `lc__weierstrass1872.png` | 바이어슈트라스 함수 — 대응하는 통계 내용 없음 | 보류 |
| 161 | `ch03/limits_continuity/continuity_poly.md` | 그림 | 사잇값은 연속의 증표가 아니다 | `—` | `lc__darboux1875_p62_110.png` | 다르부의 반례 — 대응하는 통계 내용 없음 | 보류 |
| 162 | `ch03/limits_continuity/exp_log.md` | 그림 | $e$ 라는 글자가 책에 처음 찍힌 쪽 — 수학책이 아니었다 | `—` | `el__euler1736_p68.png` | 오일러의 e 기호사 — 대응하는 통계 내용 없음 | 보류 |
| 163 | `ch03/limits_continuity/exp_log.md` | AI | 로그합지수를 미분하면 확률이 나온다 | `ch20/softmax_regression/softmax.md` | `—` | — | **옮김** |
| 164 | `ch03/limits_continuity/function_limit_continuity.md` | 그림 | 사전의 한 칸에 이미 들어 있던 것 | `—` | `lc__encyclopedie1765_limite.png` | 백과전서 LIMITE — 대응하는 통계 내용 없음 | 보류 |
| 165 | `ch03/limits_continuity/function_limit_continuity.md` | 그림 | 그림을 믿지 말자고 제목에 적은 책 | `—` | `lc__bolzano1817_s15.png` | 볼차노의 해석적 증명 — 대응하는 통계 내용 없음 | 보류 |
| 166 | `ch03/limits_continuity/limit_defined.md` | 그림 | 그래프의 극한과 극한의 그래프 | `—` | `lc__gibbs1899_p606.png` | 깁스 현상 — 대응하는 통계 내용 없음 | 보류 |
| 167 | `ch03/limits_continuity/poly_rational.md` | 그림 | 자기 것이라 하지 않은 이름 | `—` | `lc__lhopital1716_p145.png` | 로피탈 규칙의 내력 — 대응하는 통계 내용 없음 | 보류 |
| 168 | `ch03/limits_continuity/trig_area_ratio.md` | 그림 | 종이쪽에 적혀 있던 $57.2957795130$ | `—` | `lc__cotes1722_p95.png` | 라디안의 기원 — 대응하는 통계 내용 없음 | 보류 |
| 169 | `ch03/limits_continuity/trig_length_angle.md` | 그림 | 차수에 글자를 붙인 사람들 | `—` | `trig_length_angle__order.png` | 란다우 기호의 기원 — 대응하는 통계 내용 없음 | 보류 |
| 170 | `ch03/limits_continuity/trig_length_angle.md` | 그림 | 유리에 새겨 둔 작은 각 | `—` | `lc__stadia1920_p233.png` | 스타디아 측량 — 대응하는 통계 내용 없음 | 보류 |
| 171 | `ch03/limits_continuity/trig_length_angle.md` | AI | 걸음을 작게 하라는 말이 어디서 나오는가 | `ch20/optimization/optimization.md` | `—` | — | **옮김** |
| 172 | `ch03/tangent_normal/derivative_applications.md` | 그림 | 도함수가 없던 시절의 최댓값 | `—` | `tn__fermat1679.png` | 페르마의 아데콸리타스 — 대응하는 통계 내용 없음 | 보류 |
| 173 | `ch03/tangent_normal/functional_equation_tangent.md` | 그림 | 네 꼴을 한자리에 모아 번호를 붙인 쪽 | `—` | `fe__cauchy1821_p103.png` | 코시 함수방정식 — 대응하는 통계 내용 없음 | 보류 |
| 174 | `ch03/tangent_normal/functional_equation_tangent.md` | AI | 꺾인 자리가 쓸모가 될 때 | `ch18/lasso/coordinate_descent.md` | `—` | — | **옮김** |
| 175 | `ch03/tangent_normal/normal_locus.md` | 그림 | 「아무 등차수열이나 곱하라」 — 도함수를 모르고 쓴 도함수 | `—` | `tn__hudde1659_p507.png` | 후데의 규칙 — 대응하는 통계 내용 없음 | 보류 |
| 176 | `ch03/tangent_normal/parabola_tangent_perpendicular.md` | 그림 | 직선만 그렸는데 곡선이 보인다 | `—` | `tn__clairaut.png` | 클레로 방정식과 포락선 — 대응하는 통계 내용 없음 | 보류 |
| 177 | `ch03/tangent_normal/poly_tangent.md` | 그림 | 절편을 다시 집어넣은 사람 | `ch19/estimation_inference/algorithms.md` | `tn__raphson.png` | 이 책에서 그렸다 | **옮김** |
| 178 | `ch03/tangent_normal/tangent_area.md` | AI | 걸음을 열 배 줄이면 오차는 백 배 준다 | `—` | `—` | 접선 근사의 2차 오차 — 통계 쪽 자리 없음 | 보류 |
| 179 | `ch03/tangent_normal/tangent_area.md` | 그림 | 점이 먼저 이름을 얻고, 직선이 뒤따라 얻었다 | `—` | `ta__chasles1837_p371.png` | 극과 극선 — 대응하는 통계 내용 없음 | 보류 |
| 180 | `ch03/tangent_normal/tangent_count.md` | AI | 윤곽선은 접점을 모은 것이다 | `—` | `—` | 렌더링 윤곽선 — 대응하는 통계 내용 없음 | 보류 |
| 181 | `ch03/tangent_normal/tangent_count.md` | 그림 | 공통접선의 길이가 공장 편람에 실린 까닭 | `—` | `tn__rankine1887_p183.png` | 벨트 길이 — 대응하는 통계 내용 없음 | 보류 |
| 182 | `ch03/tangent_normal/trans_tangent.md` | 그림 | 탁자 위를 끌려간 회중시계 | `—` | `tt__leibniz1693_tab5.png` | 추적선 — 대응하는 통계 내용 없음 | 보류 |
| 183 | `ch03/tangent_normal/trans_tangent.md` | AI | 접선은 언제나 「곧 끝난다」고 말한다 | `ch21/parametric/exponential.md` | `—` | — | **옮김** |
| 184 | `ch04/arc_length/arc_length_limit.md` | 그림 | 굽은 선을 곧게 펼 수 있는가 | `—` | `al__wren.png` | 곡선의 구장 역사 — 대응하는 통계 내용 없음 | 보류 |
| 185 | `ch04/arc_length/arc_length_limit.md` | 그림 | 호가 현보다 얼마나 더 긴지를 한 항으로 적었다 | `—` | `al__shenkuo_arc.png` | 심괄의 회원술 — 대응하는 통계 내용 없음 | 보류 |
| 186 | `ch04/arc_length/parametric_arc_length.md` | 그림 | 끝없이 감으면서도 길이는 유한한 길 | `—` | `al__loxodrome.png` | 항정선·로그나선 — 대응하는 통계 내용 없음 | 보류 |
| 187 | `ch04/arc_length/piecewise_curve_analysis.md` | 그림 | 「나는 처음이라고 주장하지 않는다」 — 열여섯 해 뒤의 편지 세 통 | `—` | `al__neile1673_p6146_6150.png` | 구장 우선권 다툼 — 대응하는 통계 내용 없음 | 보류 |
| 188 | `ch04/area/area.md` | 그림 | 알고도 쉰아홉 해를 묻어 둔 넓이 | `—` | `ar__roberval_cycloid.png` | 사이클로이드의 넓이 — 대응하는 통계 내용 없음 | 보류 |
| 189 | `ch04/area/area_bisector.md` | 그림 | 밭을 가르던 사람들의 방법 | `—` | `ar__gillespie1875_p360.png` | 측량의 등적변형 — 대응하는 통계 내용 없음 | 보류 |
| 190 | `ch04/area/area_extremum_ratio.md` | 그림 | 황소 가죽 하나로 두를 수 있는 만큼 | `—` | `ar__vergil_aeneid_i367.png` | 등주문제 — 대응하는 통계 내용 없음 | 보류 |
| 191 | `ch04/area/area_extremum_ratio.md` | AI | 더해서 재는 평균과 곱해서 재는 평균 | `ch07/mean/sample_mean.md` | `—` | — | **옮김** |
| 192 | `ch04/area/area_formulas.md` | 그림 | 긁어 지운 자리에 남은 넓이 | `—` | `ar__palimpsest.png` | 아르키메데스 팔림프세스트 — 대응하는 통계 내용 없음 | 보류 |
| 193 | `ch04/area/area_ratio.md` | 그림 | 적분을 가르치지 않는 책에 실린 $\dfrac23$ | `—` | `ar__mensuration1841_p68.png` | 측정론 교과서의 규칙 — 대응하는 통계 내용 없음 | 보류 |
| 194 | `ch04/area/area_ratio.md` | AI | 비로 말하면 단위가 지워진다 | `ch13/performance/relative_metrics.md` | `—` | — | **옮김** |
| 195 | `ch04/area/calculus_condition.md` | 그림 | 포탄의 길이 포물선이 아니라는 것을 처음 재어 본 사람 | `—` | `ca__robins1742_p132_133.png` | 탄도 저항 측정 — 대응하는 통계 내용 없음 | 보류 |
| 196 | `ch04/area/tangent_and_area.md` | AI | 「왜 그렇게 답했는가」를 접선으로 대신 말하기 | `ch01/modern/models_vs_algorithms.md` | `—` | — | **옮김** |
| 197 | `ch04/area/tangent_and_area.md` | 그림 | 접선에서 몇 치 내려오는가 | `—` | `ar__allen1903_p68.png` | 철도 오프셋 측량 — 대응하는 통계 내용 없음 | 보류 |
| 198 | `ch04/area/two_curve_area.md` | 그림 | 활이 휠수록 집중이 크다 | `ch02/visualization/pareto.md` | `ar__lorenz1905_p219.png` | Internet Archive · JSTOR Early Journal Content | **옮김** |
| 199 | `ch04/ftc/antiderivative.md` | 그림 | 편지에 붙어 다닌 적분 | `—` | `anti__stamp_ussr1976.png` | 소련 우표의 적분 기호 — 대응하는 통계 내용 없음 | 보류 |
| 200 | `ch04/ftc/area_motion.md` | 그림 | 썰지 않고 넓이를 읽어 내던 바퀴 | `—` | `ftc__houghtaling1906_p198.png` | 면적계 — 대응하는 통계 내용 없음 | 보류 |
| 201 | `ch04/ftc/ftc.md` | 그림 | 접선으로 적힌 기본정리 | `—` | `ftc__barrow_lect10.png` | 배로의 명제 — 대응하는 통계 내용 없음 | 보류 |
| 202 | `ch04/ftc/ftc_diff.md` | 그림 | 넓이를 수로 주지 않고 곡선으로 그려 주던 기계 | `—` | `ftc__abakanowicz1886_p2.png` | 적분기 — 대응하는 통계 내용 없음 | 보류 |
| 203 | `ch04/ftc/ftc_diff.md` | AI | 적분값만 받아 들고 함수를 되살린다 — 몸을 자르지 않고 보는 법 | `—` | `ftc_diff__ct_recon.png` | 라돈 변환·CT 재구성 — 대응하는 통계 내용 없음 | 보류 |
| 204 | `ch04/ftc/functional_equation_integral.md` | 그림 | 푸는 법이 먼저 있었고, 부르는 말이 예순 해 뒤에 왔다 | `—` | `ftc__bocher1909_p1.png` | 적분방정식이라는 이름의 기원 — 대응하는 통계 내용 없음 | 보류 |
| 205 | `ch04/ftc/functional_equation_integral.md` | AI | 되돌아오는 고리를 푸는 두 가지 길 | `ch01/paradigms/reinforcement.md` | `—` | — | **옮김** |
| 206 | `ch04/ftc/functional_equation_integral.md` | 그림 | 「적분을 반 번 하면 어떻게 됩니까」 | `—` | `ftc__fractional.png` | 분수계 미적분 — 대응하는 통계 내용 없음 | 보류 |
| 207 | `ch04/ftc/integral_defined_function.md` | 그림 | 쌓아 올린 곡선의 기울기를 읽던 사람들 | `—` | `ftc__meyer1917_p466.png` | 수문학의 질량곡선 — 대응하는 통계 내용 없음 | 보류 |
| 208 | `ch04/ftc/moving_limits.md` | 그림 | 파인만이 평생 써먹은 한 가지 연장을 배운 쪽 | `—` | `ftc__woods1926_p141.png` | 적분 기호 아래의 미분 — 대응하는 통계 내용 없음 | 보류 |
| 209 | `ch04/ftc/riemann_integral.md` | 그림 | 합 기호와 적분 기호를 나란히 | `—` | `ftc__stamp_ussr1966.png` | ICM 우표 — 대응하는 통계 내용 없음 | 보류 |
| 210 | `ch04/ftc/signed_area.md` | 그림 | 넓이에 이름을 붙여 둔 그림 | `ch02/visualization/line_plots.md` | `ftc__playfair1786.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 211 | `ch04/ftc/signed_area.md` | AI | 두 분포가 얼마나 다른가를 넓이로 잰다 | `ch16/two_sample_nonparametric/ks_two_sample.md` | `ftc__cdf_gap.png` | — | **옮김** |
| 212 | `ch04/integral_equation/convolution_equation.md` | 그림 | 미끄러지는 시간을 주고 비탈을 되찾는 문제 | `—` | `ce__abel1881_p27.png` | 아벨의 등시곡선 문제 — 대응하는 통계 내용 없음 | 보류 |
| 213 | `ch04/integral_equation/convolution_equation.md` | AI | 뒤집기를 생략해도 되는 까닭 | `—` | `—` | 합성곱 신경망의 커널 — 그것을 다루는 쪽이 없다 | 보류 |
| 214 | `ch04/integral_equation/integral_equation.md` | 그림 | 이 쪽이 가른 두 꼴에는 각각 이름이 있다 | `—` | `ie__volterra_fredholm.png` | 프레드홀름·볼테라 — 대응하는 통계 내용 없음 | 보류 |
| 215 | `ch04/integral_equation/integral_zero.md` | 그림 | 아무 함수나 곱해도 $0$ 이라면 | `—` | `ie__bolza1904_p21.png` | 변분법의 기본보조정리 — 대응하는 통계 내용 없음 | 보류 |
| 216 | `ch04/integral_equation/product_grouping.md` | 그림 | 곱의 미분법을 만든 사람이 처음에 틀리게 적은 쪽 | `—` | `pg__leibniz1675_p100.png` | 라이프니츠의 오기 — 대응하는 통계 내용 없음 | 보류 |
| 217 | `ch04/integral_equation/product_grouping.md` | 그림 | 두 단계까지는 묶음 인수로, 셋째부터는 다른 길로 | `—` | `ie__bateman1910_p423.png` | 베이트먼의 붕괴 사슬 — 대응하는 통계 내용 없음 | 보류 |
| 218 | `ch04/integral_equation/product_grouping.md` | AI | 지난 것을 잊어 가며 평균 내기 | `ch07/mean/bias_consistency.md` | `—` | — | **옮김** |
| 219 | `ch04/integration_tricks/abs_piecewise_integral.md` | 그림 | 쉰세 해 뒤에야 활자가 된 두 막대 | `—` | `ap__weierstrass1894_p67.png` | 절댓값 기호의 기원 — 대응하는 통계 내용 없음 | 보류 |
| 220 | `ch04/integration_tricks/abs_piecewise_integral.md` | AI | 상쇄되는 만큼 자료가 더 든다 | `ch09/errors_and_power/power_analysis.md` | `—` | — | **옮김** |
| 221 | `ch04/integration_tricks/definite_integral_inequality.md` | 그림 | 몇 점까지 바꿔도 되는가 — 「측도 $0$」이라는 자 | `ch03/rv/continuous.md` | `it__lebesgue1904_p29.png` | Internet Archive, PD-old | **옮김** |
| 222 | `ch04/integration_tricks/double_angle_integrals.md` | 그림 | 원을 재려다 나온 끝없는 곱 — 월리스, $1656$ | `—` | `it__wallis1656_p180.png` | 월리스 곱 — 대응하는 통계 내용 없음 | 보류 |
| 223 | `ch04/integration_tricks/even_odd_integral.md` | 그림 | 짝과 홀이라는 이름이 붙던 쪽 | `—` | `eo__euler1727_p98.png` | 오일러의 우·기함수 명명 — 대응하는 통계 내용 없음 | 보류 |
| 224 | `ch04/integration_tricks/even_odd_integral.md` | AI | 대칭을 미리 알려 주면 배울 것이 줄어든다 | `ch06/estimator_quality/bias_variance.md` | `—` | — | **옮김** |
| 225 | `ch04/integration_tricks/integration_applications.md` | 그림 | 뒤로 갈 때 바늘도 뒤로 돌게 만든 사람 | `—` | `it__outhier_odometre1742.png` | 거리계 — 대응하는 통계 내용 없음 | 보류 |
| 226 | `ch04/integration_tricks/integration_applications.md` | 그림 | 적분이 막히자 고도를 잘라 정리 2 로 간 사람 | `—` | `it__goddard1919_p9_p10.png` | 고다드의 로켓 방정식 — 대응하는 통계 내용 없음 | 보류 |
| 227 | `ch04/integration_tricks/integration_applications.md` | AI | 가속도만으로 길을 찾을 수 있는가 — 적분이 쌓는 것 | `—` | `it__ins_drift.png` | 관성항법의 치우침이 t² 로 자라는 이야기 — 오차 전파를 다루는 쪽이 없다 | 보류 |
| 228 | `ch04/integration_tricks/integration_by_parts.md` | 그림 | 네모가 적분 기호이던 시절의 부분적분 | `—` | `it__taylor1715_p38.png` | 테일러의 표기 — 대응하는 통계 내용 없음 | 보류 |
| 229 | `ch04/integration_tricks/integration_by_substitution.md` | 그림 | 점을 지우고 $d$ 를 들여온 번역 | `—` | `it__lacroix1816.png` | 해석학회의 기호 전환 — 대응하는 통계 내용 없음 | 보류 |
| 230 | `ch04/integration_tricks/integration_by_substitution.md` | AI | 난수를 원하는 모양으로 바꾸는 일은 치환적분이다 | `ch03/rv/pmf_pdf_cdf.md` | `—` | — | **옮김** |
| 231 | `ch04/integration_tricks/inverse_derivative.md` | 그림 | 로그를 먼저 찾고, 지수를 나중에 만든 사람들 | `—` | `it__sarasa.png` | 사라사와 쌍곡선 넓이 — 대응하는 통계 내용 없음 | 보류 |
| 232 | `ch04/integration_tricks/inverse_derivative.md` | AI | 되돌릴 때 커지는 것 — 왜 확률이 아니라 점수를 들고 다니는가 | `ch20/softmax_regression/numerical_stability.md` | `—` | — | **옮김** |
| 233 | `ch04/integration_tricks/inverse_function_integral.md` | 그림 | 그림 없이 적힌 그림 | `—` | `it__young1912_p226.png` | 영 부등식 — 대응하는 통계 내용 없음 | 보류 |
| 234 | `ch04/integration_tricks/log_integration.md` | 그림 | 나눗셈을 끝까지 해 본 사람 | `—` | `it__mercator1668_p30.png` | 메르카토르 급수 — 대응하는 통계 내용 없음 | 보류 |
| 235 | `ch04/integration_tricks/partial_fraction_and_parts.md` | 그림 | 없다는 것까지 답해 주는 절차 | `—` | `it__risch.png` | 리시 알고리즘 — 대응하는 통계 내용 없음 | 보류 |
| 236 | `ch04/integration_tricks/periodic_integrals.md` | 그림 | 한 주기만 재어 계수를 하나씩 뽑아낸 쪽 | `—` | `it__fourier1822_p240.png` | 푸리에 계수의 직교성 — 대응하는 통계 내용 없음 | 보류 |
| 237 | `ch04/integration_tricks/repeated_parts.md` | 그림 | 멈추지 않고 끝까지 되풀이하면 | `—` | `it__todhunter1857_p58.png` | 베르누이 급수 — 대응하는 통계 내용 없음 | 보류 |
| 238 | `ch04/integration_tricks/rolle_inverse_existence.md` | 그림 | 곡선이 축을 가로지르는 그림을 증명에서 몰아낸 쪽 | `—` | `it__bolzano1817_p4.png` | 볼차노의 방법론 — 대응하는 통계 내용 없음 | 보류 |
| 239 | `ch04/integration_tricks/sum_integral_bound.md` | 그림 | 가두고 남은 틈에 이름이 붙었다 | `—` | `sib__euler1740_p157.png` | 오일러–마스케로니 상수 — 대응하는 통계 내용 없음 | 보류 |
| 240 | `ch04/integration_tricks/sum_integral_bound.md` | AI | 낱말을 몇 개나 외워 두어야 하는가 | `—` | `—` | 지프 법칙과 사전 크기 — heavy_tails 는 멱법칙을 목록에만 적는다 | 보류 |
| 241 | `ch04/integration_tricks/symmetric_substitution.md` | 그림 | 솜씨가 모자라서가 아니라 아예 없는 것이다 | `ch04/continuous_distributions/normal.md` | `it__liouville.png` | Wikimedia Commons, PD-old | **옮김** |
| 242 | `ch04/integration_tricks/trig_integrals.md` | 그림 | 미적분보다 일흔 해 먼저 만들어진 적분표 | `—` | `it__wright1610_p17.png` | 메르카토르 도법의 sec 적분 — 대응하는 통계 내용 없음 | 보류 |
| 243 | `ch04/motion/velocity_distance.md` | 그림 | 속도 그래프 아래를 잰 쪽 | `—` | `vd__galilei1638.png` | 갈릴레오의 속도 그림 — 대응하는 통계 내용 없음 | 보류 |
| 244 | `ch04/volume/cross_section_volume.md` | 그림 | 두께 없는 조각을 쌓은 사람 | `—` | `vl__cavalieri.png` | 카발리에리 원리 — 대응하는 통계 내용 없음 | 보류 |
| 245 | `ch04/volume/cross_section_volume.md` | 그림 | 천백 년 앞서 같은 원리를 쓴 사람 | `—` | `vol__jiuzhang_zugeng.png` | 조긍지와 모합방개 — 대응하는 통계 내용 없음 | 보류 |
| 246 | `ch04/volume/pappus_guldinus.md` | 그림 | 빈손으로 물러서지 않으려고 적어 둔 세 문장 | `—` | `pgu__pappus1588.png` | 파푸스-굴딘 — 대응하는 통계 내용 없음 | 보류 |
| 247 | `ch04/volume/solid_of_revolution.md` | 그림 | 술통 값을 따지다 쓴 회전체 책 | `—` | `sor__kepler1615.png` | 케플러의 술통 — 대응하는 통계 내용 없음 | 보류 |
| 248 | `ch04/volume/surface_of_revolution.md` | 그림 | 끝없이 긴데 부피는 유한하다 — 토리첼리, $1644$ | `—` | `vol__torricelli1644_p115.png` | 가브리엘의 나팔 — 대응하는 통계 내용 없음 | 보류 |
| 249 | `ch04/volume/volume.md` | 그림 | 배 한 척을 두 방향으로 썰어 본 쪽 | `—` | `vol__attwood1899_p19_20.png` | 조선학의 단면 적분 — 대응하는 통계 내용 없음 | 보류 |
| 250 | `ch04/volume/volume_condition.md` | 그림 | 반쪽과 삼분의 일이 합동일 수 있다 | `—` | `vol__hausdorff1914_p469.png` | 하우스도르프·바나흐–타르스키 — 대응하는 통계 내용 없음 | 보류 |
| 251 | `ch05/conics/conic_dandelin.md` | 그림 | 이름 하나가 덮지 못한 반쪽 | `—` | `cn__dandelin.png` | 당들랭 구 작도 — 원뿔곡선을 다루는 쪽이 없다 | 보류 |
| 252 | `ch05/conics/ellipse.md` | 그림 | $8$ 분을 넘기지 않은 사람 | `ch13/diagnostics/residuals.md` | `cn__stamp_romania1971.png` | Wikimedia Commons, PD | **옮김** |
| 253 | `ch05/conics/ellipse_area.md` | 그림 | 원기둥과 원뿔을 하나의 같은 타원으로 자르다 | `—` | `cn__serenus1696_p18.png` | 타원 넓이와 정사영 — 대응하는 통계 내용 없음 | 보류 |
| 254 | `ch05/conics/ellipse_parametric.md` | 그림 | 「길을 보여 주는 사람은 내게 위대한 아폴로니오스가 될 것이다」 | `—` | `cn__kepler1609_p300.png` | 이심각·케플러 방정식 — 상자의 알맹이가 통계가 아니다 | 보류 |
| 255 | `ch05/conics/envelope.md` | 그림 | 빛이 제 손으로 그려 놓는 포락선 | `—` | `cn__caustic_pan.png` | 포락선·초선 — 대응하는 통계 내용 없음 | 보류 |
| 256 | `ch05/conics/hyperbola.md` | 그림 | 시간차가 그리는 바다 위의 쌍곡선 | `—` | `cn__loran_hyperbolic_fix.png` | LORAN 쌍곡선 항법 — 대응하는 통계 내용 없음 | 보류 |
| 257 | `ch05/conics/locus_definition.md` | 그림 | 그 책이 우리에게 닿은 길 | `—` | `cn__apollonius_ms.png` | 『원뿔곡선론』의 전승 — 대응하는 통계 내용 없음 | 보류 |
| 258 | `ch05/conics/mixed_conics.md` | 그림 | 「모자람」·「꼭 맞음」·「넘침」 | `—` | `cn__apollonius1710_p31.png` | 원뿔곡선 이름의 유래 — 대응하는 통계 내용 없음 | 보류 |
| 259 | `ch05/conics/parabola.md` | 그림 | 산비탈에 세운 포물선 | `—` | `cn__odeillo_solar_furnace.png` | 포물면 반사 성질 — 대응하는 통계 내용 없음 | 보류 |
| 260 | `ch05/conics/parabola_properties.md` | 그림 | 극을 초점에 놓자 증명이 한 줄로 줄었다 | `—` | `cn__salmon1869_p193.png` | 좌표계 선택과 포물선 — 대응하는 통계 내용 없음 | 보류 |
| 261 | `ch05/plane_geometry/area_length.md` | 그림 | 세 거리를 마음대로 나누어 가져도 합은 그대로다 | `—` | `area_length__viviani.png` | 비비아니 정리 — 대응하는 통계 내용 없음 | 보류 |
| 262 | `ch05/plane_geometry/circle_tangent.md` | 그림 | 정리는 이미 있었고, 그가 한 일은 이름을 붙인 것이다 | `—` | `pg__steiner1826_p164.png` | 방멱 명명 — 대응하는 통계 내용 없음 | 보류 |
| 263 | `ch05/plane_geometry/circumscribed_circle.md` | 그림 | 스물두 살에 쓴 한 줄 — $LS=\tfrac12R-r$ | `—` | `pg__feuerbach1822_p38.png` | 포이어바흐 정리 — 대응하는 통계 내용 없음 | 보류 |
| 264 | `ch05/plane_geometry/circumscribed_circle.md` | AI | 점을 면으로 잇는 일은 외접원이 정한다 | `—` | `—` | 들로네 삼각분할 — 메시 생성을 다루는 쪽이 없다 | 보류 |
| 265 | `ch05/plane_geometry/coord_locus.md` | 그림 | 거리 조건을 쇠막대로 걸어 자취를 손에 쥐다 | `—` | `pg__kempe1877_p12.png` | 포셀리에 연동장치 — 대응하는 통계 내용 없음 | 보류 |
| 266 | `ch05/plane_geometry/motion_optim.md` | 그림 | 거울이 굽으면 겨냥할 자리를 찾을 수 없다 | `—` | `pg__alhazen.png` | 알하젠 문제 — 대응하는 통계 내용 없음 | 보류 |
| 267 | `ch05/plane_geometry/regular_polygon_circle.md` | AI | 낱말의 자리를 각으로 적는 법 | `—` | `—` | 회전 위치부호(RoPE) — 트랜스포머를 다루는 쪽이 없다 | 보류 |
| 268 | `ch05/plane_geometry/regular_polygon_circle.md` | 그림 | 자르고 또 잘라 더 자를 수 없는 데 이르면 | `—` | `pg__jiuzhang_liuhui.png` | 유휘 할원술 — 대응하는 통계 내용 없음 | 보류 |
| 269 | `ch05/plane_geometry/tangent_circles_between_lines.md` | 그림 | 둘로는 되는데 셋으로는 안 되는 까닭 | `—` | `pg__wantzel1837_p369.png` | 방첼의 작도불가능성 — 대응하는 통계 내용 없음 | 보류 |
| 270 | `ch05/plane_geometry/trapezoid_means.md` | 그림 | 여섯·여덟·아홉·열둘 | `—` | `pg__boethius1488_f47.png` | 조화평균 이름의 유래 — 알맹이가 음악 음정이다 | 보류 |
| 271 | `ch05/plane_geometry/triangle_similar.md` | 그림 | 선이 아니라 저울로 쓴 기하학 책 | `—` | `pg__ceva_statics.png` | 체바 정리의 정역학적 증명 — 대응하는 통계 내용 없음 | 보류 |
| 272 | `ch05/plane_geometry/trig_law.md` | 그림 | 삼각법만 다룬 첫 책의 제2권 첫 명제 | `—` | `trig_law__regiomontanus1533.png` | 사인법칙의 초판 — 대응하는 통계 내용 없음 | 보류 |
| 273 | `ch05/plane_geometry/trig_law.md` | AI | 로봇 팔이 관절 각도를 정하는 식 | `—` | `—` | 역기구학과 코사인법칙 — 대응하는 통계 내용 없음 | 보류 |
| 274 | `ch05/plane_vectors/coord_component.md` | 그림 | 「방향을 가진 선」이라는 말이 붙은 쪽 | `—` | `pv__argand1806_p11.png` | 아르강의 유향선분 — ch00 은 벡터를 열벡터로 바로 세운다 | 보류 |
| 275 | `ch05/plane_vectors/coord_component.md` | 그림 | 방위가 변하지 않으면 부딪친다 | `—` | `pv__knight1901_p234.png` | 충돌 회피와 외적 — 대응하는 통계 내용 없음 | 보류 |
| 276 | `ch05/plane_vectors/coordinate_vector_theory.md` | 그림 | 식에 점을 넣었더니 거리가 나왔다 | `—` | `pv__hesse1865_p16.png` | 헤세 표준형 — 부호 있는 거리를 쓰는 쪽이 없다 | 보류 |
| 277 | `ch05/plane_vectors/inner_product.md` | 그림 | 내적이 처음 인쇄된 쪽에는 가운뎃점이 없다 | `ch00/math/linear_algebra.md` | `pv__gibbs_vector_p5.png` | Internet Archive 스캔, $1881$–$84$ 간행이라 PD | **옮김** |
| 278 | `ch05/plane_vectors/locus_region.md` | 그림 | 점선 하나가 가른 구역 | `ch01/classical/observational.md` | `pv__snow1855_map.png` | Wikimedia Commons, PD | **옮김** |
| 279 | `ch05/plane_vectors/locus_region.md` | 그림 | 각 하나로 암초를 피하는 법 | `—` | `pv__knight1901_p268_269.png` | 위험각 항법 — 대응하는 통계 내용 없음 | 보류 |
| 280 | `ch05/plane_vectors/locus_region.md` | AI | 로봇을 점으로 만드는 법 | `—` | `—` | 민코프스키 합과 경로계획 — 대응하는 통계 내용 없음 | 보류 |
| 281 | `ch05/plane_vectors/max_min.md` | 그림 | 「당나귀도 아는 것」 | `—` | `pv__heath1908_p287.png` | 삼각부등식 — 그것을 세우는 쪽이 없다 | 보류 |
| 282 | `ch05/plane_vectors/parametric_velocity_inner_product.md` | 그림 | 답이 두 칸으로 나뉜 연습문제 | `—` | `pv__gunnery1917_p53.png` | 진공 포물선 대 공기저항 사표 — 교훈만 닮고 내용이 멀다 | 보류 |
| 283 | `ch05/plane_vectors/parametric_velocity_inner_product.md` | AI | 크기를 묶어 두면 걸음은 접선 방향뿐이다 | `—` | `—` | 구면 위 제약 최적화 — 그것을 다루는 쪽이 없다 | 보류 |
| 284 | `ch05/plane_vectors/position_ratio.md` | 그림 | 좌표를 원점에서 재지 않고 꼭짓점에 얹은 사람 | `—` | `pv__moebius1827_p17.png` | 뫼비우스의 무게중심좌표 — 자료의 가중평균과 다르다 | 보류 |
| 285 | `ch05/plane_vectors/vector_basics.md` | 그림 | 「같은 시간에」 한 줄로 적은 평행사변형법 | `—` | `vb__newton1687.png` | 『프린키피아』의 평행사변형법 — 대응하는 통계 내용 없음 | 보류 |
| 286 | `ch05/plane_vectors/vector_lattice_inner_product.md` | 그림 | 「일」이라는 말과 $\tfrac12$ 을 함께 들여온 쪽 | `—` | `pv__coriolis1829_p3.png` | 코리올리의 「일」 — 대응하는 통계 내용 없음 | 보류 |
| 287 | `ch05/space_geometry/angle_distance_locus.md` | 그림 | 공간을 점이 아니라 직선으로 채우려 한 쪽 | `—` | `sg__plucker1865_p726.png` | 플뤼커 직선기하 — 대응하는 통계 내용 없음 | 보류 |
| 288 | `ch05/space_geometry/coord_vector.md` | 그림 | 두 차원짜리 종이에 세 차원을 적는 기술 | `—` | `sg__monge1798_p2.png` | 몽주의 화법기하 — 대응하는 통계 내용 없음 | 보류 |
| 289 | `ch05/space_geometry/coord_vector.md` | AI | 화면 밖의 것을 지우는 데 드는 곱셈 열여덟 번 | `—` | `—` | 절두체 컬링 — 대응하는 통계 내용 없음 | 보류 |
| 290 | `ch05/space_geometry/projection_dihedral.md` | 그림 | 길이는 제멋대로인데 각은 언제나 같았다 | `—` | `sg__steno1669_crystals.png` | 스테노의 면각 일정의 법칙 — 대응하는 통계 내용 없음 | 보류 |
| 291 | `ch05/space_geometry/shadow_projection.md` | 그림 | 해는 평행하게 비추고 눈은 한 점에서 본다 | `—` | `sp__lastsupper.png` | 평행사영 대 중심사영 — 이 책의 사영은 직교사영이다 | 보류 |
| 292 | `ch05/space_geometry/shadow_projection.md` | AI | 화면에 그리는 일은 나눗셈 한 번이다 | `—` | `—` | 원근 나눗셈 — 대응하는 통계 내용 없음 | 보류 |
| 293 | `ch05/space_geometry/solid_reasoning.md` | 그림 | 『원론』은 이 다섯으로 끝난다 | `—` | `sg__euclid1482_bk13.png` | 정다면체가 다섯뿐 — 대응하는 통계 내용 없음 | 보류 |
| 294 | `ch05/space_geometry/solid_rotation.md` | 그림 | 곧은 쇠막대만으로 세운 잘록한 탑 | `—` | `sg__shukhov1896.png` | 일엽쌍곡면 선직면 — 대응하는 통계 내용 없음 | 보류 |
| 295 | `ch05/space_geometry/solid_volume.md` | 그림 | 넓이를 맞추어 공정하게 만든 십사면체 | `—` | `sg__juryeonggu.png` | 공정한 주사위 — 같은 주제의 상자가 이미 sample_space.md 에 있다 | 보류 |
| 296 | `ch05/space_geometry/space_basics.md` | 그림 | 나사가 넷이면 흔들린다 | `—` | `sg__surveying1913_p196_197.png` | 세 점이 평면을 정한다 — 알맹이가 기하다 | 보류 |
| 297 | `ch05/space_geometry/space_projection_dot_product.md` | 그림 | 빛이 비스듬히 들면 왜 어두워지는가를 정사영으로 답한 쪽 | `—` | `sg__lambert1760_p27.png` | 람베르트 코사인 법칙 — 대응하는 통계 내용 없음 | 보류 |
| 298 | `ch05/space_geometry/space_projection_dot_product.md` | AI | 자료에 식을 맞추는 일은 수선의 발을 찾는 일이다 | `ch00/linalg_regression/square_matrices/orthogonal_projection.md` | `—` | — | **옮김** |
| 299 | `ch05/space_geometry/sphere_plane.md` | 그림 | 하늘을 고리 몇 개로 대신한 기계 | `—` | `sg__armillary_bhm2094c.png` | 혼천의와 대원 — 대응하는 통계 내용 없음 | 보류 |
| 300 | `ch06/arguments/chain_decomposition.md` | 그림 | 같은 한 줄을 뒤집으면 위쪽 경계가 나온다 | `—` | `ar__sperner1973.png` | 스페르너·딜워스 정리 — 순서집합을 다루는 쪽이 없다 | 보류 |
| 301 | `ch06/arguments/construction.md` | 그림 | 만우절의 유래를 묻는 물음들 사이에 끼어 있던 문제 | `—` | `ag__lgdiary1850_p48.png` | 슈타이너 삼중계 — 블록설계를 다루는 쪽이 없다 | 보류 |
| 302 | `ch06/arguments/exchange_argument.md` | 그림 | 좋아지는 교환이 하나도 없는 자리를 만들어 낸 논문 | `—` | `ar__gale2003.png` | 게일–섀플리 안정 짝짓기 — 대응하는 통계 내용 없음 | 보류 |
| 303 | `ch06/arguments/extremal.md` | 그림 | 한 줄짜리 원리를 어디에 쓸지 알아내는 데 오십 년 | `—` | `ar__sylvester.png` | 실베스터–갈라이 정리 — 대응하는 통계 내용 없음 | 보류 |
| 304 | `ch06/arguments/invariant.md` | 그림 | 도달할 수 없는 자리를 「풀었다」고 크게 써 붙였다 | `—` | `ag__fifteen_poster1880.png` | 15 퍼즐과 순열 불변량 — 대응하는 통계 내용 없음 | 보류 |
| 305 | `ch06/arguments/monovariant.md` | 그림 | 단조량을 아흔 해째 못 찾은 조작 | `—` | `ar__collatz1984.png` | 콜라츠 추측 — 대응하는 통계 내용 없음 | 보류 |
| 306 | `ch06/arguments/nim_game.md` | 그림 | 이름도 이 쪽에서 붙었다 | `—` | `nim__bouton1901_p35.png` | 님과 님-합 — 대응하는 통계 내용 없음 | 보류 |
| 307 | `ch06/arguments/optimum_bounds.md` | 그림 | 두 걸음이 아직 만나지 못한 수 | `—` | `ar__erdos1992.png` | 램지 수 R(5,5) — 대응하는 통계 내용 없음 | 보류 |
| 308 | `ch06/arguments/optimum_search.md` | 그림 | 합판 공장에서 물어 온 한 가지 물음 | `—` | `ag__kantorovich1975.png` | 칸토로비치와 선형계획법 — LP 를 다루는 쪽이 없다 | 보류 |
| 309 | `ch06/arguments/optimum_search.md` | AI | 글을 토막 내는 일도 표를 채우는 일이다 | `—` | `ag__dp_segment.png` | 동적계획법 토큰화 — 그것을 다루는 쪽이 없다 | 보류 |
| 310 | `ch06/arguments/pigeonhole.md` | 그림 | 비둘기집이 아니라 서랍이었다 | `—` | `ar__dirichlet.png` | 비둘기집 원리의 이름 내력 — 그 원리를 쓰는 쪽이 없다 | 보류 |
| 311 | `ch06/arguments/positional_construction.md` | 그림 | 접시가 둘이면 자릿수가 셋이 된다 | `—` | `ag__bachet1884_p154_155.png` | 바셰의 추 문제 — 대응하는 통계 내용 없음 | 보류 |
| 312 | `ch06/arguments/set_construction.md` | 그림 | 가장 멀리 벌린 끝에 놓인 자 | `—` | `ar__golomb2011.png` | 골롬 자·시돈 집합 — 대응하는 통계 내용 없음 | 보류 |
| 313 | `ch06/binomial/absorption_identity.md` | 그림 | 「대장을 반드시 넣어」 | `ch04/discrete_distributions/binomial.md` | `bn__whitworth1870_p45.png` | Internet Archive 스캔, PD | **옮김** |
| 314 | `ch06/binomial/absorption_identity.md` | AI | 기여를 나눌 때 무게가 층마다 꼭 $1/n$ 이 되는 까닭 | `—` | `bn__shapley_layers.png` | 섀플리값(SHAP) — 이 책의 특징선택은 벌점 기반이다 | 보류 |
| 315 | `ch06/binomial/absorption_identity.md` | 그림 | 느낌표가 생기기 전에 먼저 있던 것 | `—` | `bn__kramp.png` | 계승 기호의 내력 — 하강계승을 세우는 자리가 없다 | 보류 |
| 316 | `ch06/binomial/binomial_calculus.md` | 그림 | 미분을 쓸 수 없던 교실에서 같은 합을 구하면 | `—` | `bn__hallknight1891_p334.png` | Σr²C(n,r) — 이 합을 다루는 쪽이 없다 | 보류 |
| 317 | `ch06/binomial/binomial_calculus.md` | AI | 분배함수를 미분하면 확률이 떨어져 나온다 | `ch20/optimization/cross_entropy.md` | `bn__softmax_grad.png` | — | **옮김** |
| 318 | `ch06/binomial/binomial_identities.md` | 그림 | 수는 다 적혀 있는데 합은 적혀 있지 않다 | `—` | `bn__tartaglia1556.png` | 타르탈리아의 삼각형 — 양휘 쪽 상자가 같은 자리를 맡는다 | 보류 |
| 319 | `ch06/binomial/binomial_identities.md` | AI | 망 하나를 학습시키면서 $2^{\,n}$ 개를 학습시킨다 | `—` | `bn__dropout_subnets.png` | 드롭아웃 = 2^n 부분망 — 같은 설명이 연습문제 풀이 안에만 있다 | 보류 |
| 320 | `ch06/binomial/binomial_theorem.md` | 그림 | 지수를 자연수 밖으로 밀어낸 편지 | `—` | `bt__newton1676.png` | 뉴턴의 일반 이항정리 — negative_binomial.md 가 음이항급수를 다루지 않는다 | 보류 |
| 321 | `ch06/binomial/hockey_stick.md` | 그림 | 파스칼에게 이것은 「둘째 따름」이었다 | `—` | `bn__pascal_latin_f106.png` | 하키스틱 항등식 — 본문에서 쓰는 자리가 없다 | 보류 |
| 322 | `ch06/binomial/multinomial_theorem.md` | 그림 | 글자를 섞어 삼 년을 숨긴 문장 | `—` | `bn__huygens1659_p299.png` | 하위헌스의 애너그램 — 다항계수를 세우는 쪽이 없다 | 보류 |
| 323 | `ch06/binomial/multinomial_theorem.md` | AI | 다항계수의 로그가 곧 「글자 하나에 몇 비트」다 | `ch20/optimization/cross_entropy.md` | `bn__entropy_multinomial.png` | — | **옮김** |
| 324 | `ch06/binomial/vandermonde_convolution.md` | 그림 | 이름은 붙었는데 그의 것이 아닌 것 | `ch04/discrete_distributions/hypergeometric.md` | `bn__vandermonde.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 325 | `ch06/counting_applications/count_equation.md` | 그림 | 답 셋을 적어 두고 「그 이치를 다할 길이 없다」고 덧붙였다 | `—` | `ca__zhangqiujian_baiji.png` | 백계문제(부정방정식) — 대응하는 통계 내용 없음 | 보류 |
| 326 | `ch06/counting_applications/count_functions.md` | 그림 | 「함수를 센다」로 바꾸었더니 열두 칸이 되었다 | `—` | `ca__rota1970.png` | 로타의 십이로 — 세는 법을 정리하는 쪽이 없다 | 보류 |
| 327 | `ch06/counting_applications/distribute_balls.md` | 그림 | 틀린 셈이 맞는 답을 냈다 | `ch03/probability/sample_space.md` | `ca__bose1925.png` | Wikimedia Commons, PD-India | **옮김** |
| 328 | `ch06/counting_applications/lattice_paths.md` | 그림 | 세지 않고 재는 대신, 재지 않고 세기 | `—` | `ca__pick.png` | 픽의 정리 — 대응하는 통계 내용 없음 | 보류 |
| 329 | `ch06/counting_applications/lattice_shortest_path.md` | 그림 | 반사가 한 번도 나오지 않는 「반사원리」 | `—` | `ca__andre1887_p436.png` | 앙드레의 반사원리 — 확률보행을 다루는 쪽이 없다 | 보류 |
| 330 | `ch06/counting_techniques/case_split.md` | 그림 | 갈래가 이천 가지라 사람이 다 볼 수 없었다 | `—` | `ct__fourcolors1976.png` | 사색정리와 컴퓨터 증명 — 대응하는 통계 내용 없음 | 보류 |
| 331 | `ch06/counting_techniques/complement_counting.md` | 그림 | 오백 년 묵은 규칙에 이름이 붙기까지 | `ch00/math/logic_sets.md` | `ct__demorgan.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 332 | `ch06/counting_techniques/complement_counting.md` | AI | 「한 번만 맞으면 된다」를 재는 법 | `ch06/estimator_quality/bias_variance.md` | `ct__passk_estimator.png` | — | **옮김** |
| 333 | `ch06/counting_techniques/correspondence_count.md` | 그림 | 세지 않고 같음을 확인하던 나뭇조각 | `—` | `ct__tally_sticks.png` | 엄대와 전단사 — 대응하는 통계 내용 없음 | 보류 |
| 334 | `ch06/counting_techniques/graph_counting.md` | 그림 | 「행복한 결말 문제」라는 이름이 붙은 까닭 | `—` | `ct__szekeres2001.png` | 에르되시–세케레시 — 대응하는 통계 내용 없음 | 보류 |
| 335 | `ch06/counting_techniques/inclusion_exclusion.md` | 그림 | 이름이 둘 붙었는데 둘 다 처음이 아니다 | `ch03/probability/axioms.md` | `ct__dasilva.png` | Wikimedia Commons, PD-old-assumed | **옮김** |
| 336 | `ch06/counting_techniques/magic_square.md` | 그림 | 마방진 안에 제작연도를 숨긴 판화 | `—` | `ct__durer1514_square.png` | 뒤러의 마방진 — 대응하는 통계 내용 없음 | 보류 |
| 337 | `ch06/counting_techniques/magic_square.md` | 그림 | 첫 우표에 실린 사람이 마방진을 만들었다 | `—` | `ct__stamp_usa1847.png` | 프랭클린의 마방진 — 대응하는 통계 내용 없음 | 보류 |
| 338 | `ch06/permutation_combination/combination.md` | 그림 | 파스칼보다 삼백 년 앞선 그림 | `ch04/discrete_distributions/binomial.md` | `comb__yanghui.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 339 | `ch06/permutation_combination/count_sum_product.md` | 그림 | 경우의 수가 크다고 안전한 것은 아니다 | `—` | `cs__enigma.png` | 에니그마의 설정 수 — 대응하는 통계 내용 없음 | 보류 |
| 340 | `ch06/permutation_combination/permutation.md` | 그림 | 종치기 교본에 실린 계승표 | `—` | `pc__stedman1677_p13.png` | 체인지 링잉과 계승표 — 순열을 세우는 쪽이 없다 | 보류 |
| 341 | `ch06/permutation_combination/repeated_combination.md` | 그림 | 서른두 장인데 얼굴은 스물한 가지 | `—` | `rc__paigow32.png` | 골패와 중복조합 — 중복조합을 다루는 쪽이 없다 | 보류 |
| 342 | `ch06/permutation_combination/repeated_combination.md` | AI | 순서를 버리면 가짓수가 몇 자리나 줄어드는가 | `—` | `rc__bagofwords.png` | 낱말 가방 — 텍스트 표현을 다루는 쪽이 없다 | 보류 |
| 343 | `ch06/permutation_combination/repeated_permutation.md` | 그림 | 손끝 하나에 들어가는 개수가 글자 수를 정했다 | `—` | `rp__braille1829.png` | 점자와 2^6 — 대응하는 통계 내용 없음 | 보류 |
| 344 | `ch06/recurrence_generating/generating_function.md` | 그림 | 「A 를 나는 생성함수라 부른다」 | `ch03/rv/mgf.md` | `rg__laplace1886_p29_p40.png` | Internet Archive 스캔, 캘리포니아 대학 소장본 | **옮김** |
| 345 | `ch06/recurrence_generating/recurrence_count.md` | 그림 | 세 해를 기다리는 소 | `—` | `rg__ganita_kaumudi1936.png` | 나라야나의 소 문제 — 대응하는 통계 내용 없음 | 보류 |
| 346 | `ch06/special_numbers/catalan_numbers.md` | 그림 | 이름을 가장 늦게 얻은 수 | `—` | `sn__catalan.png` | 카탈란 수 — 대응하는 통계 내용 없음 | 보류 |
| 347 | `ch06/special_numbers/catalan_numbers.md` | 그림 | 다각형을 자르다가 나온 점화식 | `—` | `sn__segner.png` | 제그너의 점화식 — 대응하는 통계 내용 없음 | 보류 |
| 348 | `ch06/special_numbers/count_derangements.md` | 그림 | 카드놀이에서 나온 문제 | `—` | `sn__montmort.png` | 몽모르의 교란순열 — 맞춤문제를 본문에서 다루는 쪽이 없다 | 보류 |
| 349 | `ch06/special_numbers/integer_partition.md` | 그림 | 표를 손으로 $200$ 까지 채운 포병 소령 | `—` | `sn__macmahon.png` | 맥마흔의 분할표 — 대응하는 통계 내용 없음 | 보류 |
| 350 | `ch06/special_numbers/stirling_numbers.md` | 그림 | 묘비에 「베네치아 사람」이라 새긴 수학자 | `—` | `sn__stirling_grave.png` | 스털링 수 — 이 책에서 쓰는 자리가 없다 | 보류 |
| 351 | `ch07/basics/birthday_problem.md` | 그림 | 믿기지 않아서 발표하지 않았다 | `ch09/multiple_testing/fwer.md` | `pb__vonmises.png` | 프랑스 국립도서관 소장, Wikimedia Commons, PD-US-expired | **옮김** |
| 352 | `ch07/basics/case_division_probability.md` | 그림 | 포탄이 어디에 떨어지는지 모를 때 쓰는 표 | `ch04/continuous_distributions/normal.md` | `cs__fm640_1945_p19.png` | Internet Archive, 미국 정부 저작물이라 저작권 없음 | **옮김** |
| 353 | `ch07/basics/combination_probability.md` | 그림 | 열셋인지 열넷인지 아직 다투는 한 벌 | `—` | `pb__mamluk_cards.png` | 카드 한 벌 고증과 풀하우스 확률 — 조합 세기라 걸릴 정의·정리가 없다 | 보류 |
| 354 | `ch07/basics/complement_probability.md` | 그림 | 원숭이 백만 마리를 불러낸 까닭 | `ch03/probability/axioms.md` | `pb__borel1929.png` | Wikimedia Commons · 프랑스 국립도서관, 저작권 만료 | **옮김** |
| 355 | `ch07/basics/dice_pair_condition.md` | 그림 | 반듯하지 않은 주사위 | `—` | `prob__roman_dice.png` | 같은 쪽 상자 2(네 면으로만 서는 주사위)가 같은 말을 더 좋은 자료로 이미 한다 | 보류 |
| 356 | `ch07/basics/function_probability.md` | 그림 | 공을 칸에 던지면 함수가 하나 생긴다 | `—` | `fp__todhunter1889_p461.png` | 공·칸 배치와 전사함수 세기 — 대응하는 절이 없다 | 보류 |
| 357 | `ch07/basics/integer_solution_probability.md` | 그림 | 「특별한 재주」라 부르던 증명 | `—` | `is__netto1901_p21.png` | 중복조합과 별·막대 — 그것을 다루는 쪽이 없다 | 보류 |
| 358 | `ch07/basics/permutation_probability.md` | 그림 | 열네 글자로 만든 자물쇠 | `—` | `pp__hooke1676_p31.png` | 같은 것이 있는 순열 — 그것을 다루는 쪽이 없다 | 보류 |
| 359 | `ch07/basics/probability.md` | 그림 | 세 줄로 적은 확률 | `ch03/probability/axioms.md` | `prob__kolmogorov.png`, `prob__grundbegriffe1933.png` | Wikimedia Commons, 촬영 Konrad Jacobs, 오버볼파흐 수학연구소, CC BY-SA 2.0 DE / Wikimedia Commons, 스캔 KurtSchwitters, CC BY-SA 3.0 | **옮김** |
| 360 | `ch07/basics/probability_addition.md` | 그림 | 인쇄 오류를 평생 미워한 사람의 이름이 붙은 부등식 | `ch03/probability/axioms.md` | `pb__bonferroni.png` | Wikimedia Commons, PD-Italy | **옮김** |
| 361 | `ch07/basics/sample_space.md` | 그림 | 네 면으로만 서는 주사위 | `ch03/probability/sample_space.md` | `pb__ancient_dice.png` | Wikimedia Commons, 촬영 Xocolatl, PD-self / 메트로폴리탄 미술관 10.130.1158, CC0 | **옮김** |
| 362 | `ch07/conditional/bayes.md` | 그림 | 되짚기가 활자로 놓인 자리 | `ch03/probability/bayes.md` | `cd__by_bayes1763_prop5.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 363 | `ch07/conditional/bayes.md` | 그림 | 곱하기를 더하기로 바꾸어 암호를 풀다 | `ch03/probability/bayes.md` | `cd__turing1936.png` | Wikimedia Commons, PD-US-not renewed | **옮김** |
| 364 | `ch07/conditional/classification_table.md` | 그림 | 네 칸짜리 표에 이름이 붙던 날 | `ch19/evaluation/confusion_matrix.md` | `cd__pearson1904_p1.png` | Internet Archive · 코넬대 도서관 스캔, 저작권 없음 | **옮김** |
| 365 | `ch07/conditional/classification_table.md` | AI | 「정확도 $99$ 퍼센트」가 쓸모없을 수 있는 까닭 | `ch19/evaluation/confusion_matrix.md` | `—` | — | **옮김** |
| 366 | `ch07/conditional/conditional_applications.md` | 그림 | 「적어도 하나」를 비밀 문서에 적어 둔 까닭 | `ch03/probability/independence.md` | `cd__morse_kimball1946_p89.png` | Internet Archive, 미국 정부 저작물이라 저작권 없음 | **옮김** |
| 367 | `ch07/conditional/conditional_applications.md` | AI | 장치를 늘릴수록 더 자주 멈추는 까닭 | `ch03/probability/independence.md` | `—` | — | **옮김** |
| 368 | `ch07/conditional/conditional_independence.md` | 그림 | 「서로 다른 기록을 섞을 때 생기는 오류」 | `ch03/probability/conditional_independence.md` | `cd__yule1903_p132.png` | Internet Archive, 저작권 없음 | **옮김** |
| 369 | `ch07/conditional/conditional_probability.md` | 그림 | 곱셈정리가 활자로 놓인 자리 | `ch03/probability/conditional.md` | `cd__demoivre_doctrine.png` | Wikimedia Commons, PD-old-100-expired | **옮김** |
| 370 | `ch07/conditional/conditional_probability.md` | AI | 문장 하나의 확률을 낱말 하나씩 쪼개는 법 | `ch03/probability/conditional.md` | `cd__ai_chain_scale.png` | — | **옮김** |
| 371 | `ch07/conditional/conditional_state.md` | 그림 | 네 칸은 세어서 얻은 것이 아니라 $1$ 을 전개한 것이다 | `ch00/math/logic_sets.md` | `cd__boole1854_p76.png` | Internet Archive · 바이에른 주립도서관 스캔, 저작권 없음 | **옮김** |
| 372 | `ch07/conditional/conditional_two_stage.md` | 그림 | 사전확률을 적지 못한 채 돌린 계산 | `ch06/bayesian/prior_posterior.md` | `cd__bordereau1894.png` | 프랑스 국방사료원 소장, Wikimedia Commons, PD-old-70 | **옮김** |
| 373 | `ch07/conditional/game_probability.md` | 그림 | 모든 것을 모아 적은 책에, 풀지 못한 물음 하나 | `ch03/probability/conditional.md` | `cd__pacioli1523_summa.png` | Stockholms Universitetsbibliotek 촬영, CC BY 2.0 | **옮김** |
| 374 | `ch07/conditional/independent_binomial.md` | 그림 | 신의 존재를 증명하려고 쓴 첫 가설검정 | `ch16/one_sample_nonparametric/sign_test.md` | `cd__arbuthnot1710_p186.png` | Internet Archive, 저작권 없음 | **옮김** |
| 375 | `ch07/conditional/independent_events.md` | 그림 | 완두 오백쉰여섯 알로 잰 독립 | `ch03/probability/independence.md` | `cd__mendel1866_p19.png` | Internet Archive, 저작권 없음 | **옮김** |
| 376 | `ch07/conditional/monty_hall.md` | 그림 | 이름만 빌려준 사람 | `ch03/probability/conditional.md` | `cd__monty_hall.png` | Wikimedia Commons, PD-Pre1978 | **옮김** |
| 377 | `ch07/conditional/monty_hall.md` | 그림 | 편지 만 통이 쏟아진 칼럼 | `ch03/probability/conditional.md` | `cd__erdos.png` | Topsy Kretts 촬영, Wikimedia Commons, CC BY 3.0 | **옮김** |
| 378 | `ch07/conditional/monty_hall.md` | 그림 | 몬티보다 백 년 먼저 나온 문제 | `ch03/probability/bayes.md` | `cd__bertrand.png` | Wikimedia Commons, PD-old | **옮김** |
| 379 | `ch07/conditional/stratified_reversal.md` | 그림 | 심슨은 만화가 아니다 | `ch12/ecological_correlation/simpsons_paradox.md` | `cd__bletchley.png` | Wikimedia Commons, 촬영 DeFacto, CC BY-SA 4.0 | **옮김** |
| 380 | `ch07/conditional/stratified_reversal.md` | 그림 | 소송이 될 뻔한 입학 통계 | `ch12/ecological_correlation/simpsons_paradox.md` | `cd__berkeley.png` | Wikimedia Commons, 촬영 Daderot, CC0 | **옮김** |
| 381 | `ch07/conditional/total_probability.md` | 그림 | 바다를 칸으로 나누고 하나씩 지워 나간 기록 | `ch03/probability/conditional.md` | `cd__palomares1975_fig333.png` | 미국 공군 자료, PD-USGov | **옮김** |
| 382 | `ch07/descriptive/central_tendency.md` | 그림 | 눈금이 없는 그래프 | `ch02/center/mean_median_mode.md` | `dsc__stamp_indonesia1969.png` | Wikimedia Commons, PD | **옮김** |
| 383 | `ch07/descriptive/covariance_correlation.md` | 그림 | 「회귀」라는 이름이 나온 그림 | `ch03/rv/correlation.md` | `dc__galton1886_plate10.png` | Wikimedia Commons, PD-Art | **옮김** |
| 384 | `ch07/descriptive/dispersion.md` | 그림 | 백분위수를 종이에 그린 것 | `ch02/spread/robust.md` | `dsc__growth_chart_cdc2000.png` | Wikimedia Commons, PD-USGov-HHS-CDC | **옮김** |
| 385 | `ch07/descriptive/frequency_distribution.md` | 그림 | 통계학이라는 말보다 먼저 있던 도수분포표 | `ch02/exploration/histograms.md` | `fd__bills_of_mortality_1665.png` | Wikimedia Commons, Wellcome Collection, CC BY 4.0 | **옮김** |
| 386 | `ch07/descriptive/frequency_distribution.md` | 그림 | 히스토그램을 둥글게 말아 의회에 내민 그림 | `ch02/exploration/histograms.md` | `fd__nightingale1858.png` | Wikimedia Commons, PD-old | **옮김** |
| 387 | `ch07/descriptive/frequency_distribution.md` | 그림 | 날 수로 보면 가장 많은 쪽이 가장 적은 쪽이었다 | `ch02/exploration/histograms.md` | `fd__snow1855_p86.png` | Wellcome Library 스캔 | **옮김** |
| 388 | `ch07/distributions/binomial_distribution.md` | 그림 | 이름이 된 책 | `ch04/discrete_distributions/bernoulli.md` | `bin__ars1713.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 389 | `ch07/distributions/binomial_to_poisson.md` | 그림 | 말에 차여 죽은 사람을 센 표 | `ch04/discrete_distributions/poisson.md` | `binomial_to_poisson__bortkiewicz1898_p25.png` | Wikimedia Commons, PD-US-expired | **옮김** |
| 390 | `ch07/distributions/exponential_distribution.md` | 그림 | 전화선을 몇 개 깔 것인가 | `ch04/continuous_distributions/exponential.md` | `ds__molina1927_p463.png` | Internet Archive 스캔, 저작권 없음 | **옮김** |
| 391 | `ch07/distributions/geometric_distribution.md` | AI | 기다린 보람이 없으므로 간격을 늘린다 | `ch04/discrete_distributions/geometric.md` | `—` | — | **옮김** |
| 392 | `ch07/distributions/geometric_distribution.md` | 그림 | 담뱃갑에 든 쉰 장, 다 모으려면 | `ch04/discrete_distributions/geometric.md` | `ds__cope_warriors_card.png` | Wikimedia Commons, 저작권 만료 | **옮김** |
| 393 | `ch07/distributions/hypergeometric_distribution.md` | 그림 | 검사를 얼마나 할 것인가 | `ch04/discrete_distributions/hypergeometric.md` | `ds__dodge1929_p629.png` | Internet Archive 스캔, 저작권 없음 | **옮김** |
| 394 | `ch07/distributions/poisson.md` | 그림 | 이름이 붙기 전의 포아송분포 | `ch04/discrete_distributions/poisson.md` | `poisson__poisson1837_p206.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 395 | `ch07/distributions/poisson.md` | 그림 | 에펠탑에 새겨진 이름 | `ch04/discrete_distributions/poisson.md` | `poisson__eiffel.png` | Wikimedia Commons, 촬영 Victor R. Ruiz, CC BY-SA 2.0 | **옮김** |
| 396 | `ch07/expectation_applications/conditional_expectation.md` | 그림 | 「검사를 얼마나 할 것인가」에 수로 답한 쪽 | `ch03/rv/conditional_dist.md` | `ce__dodge_romig1929_p621_628_629.png` | Internet Archive · 저작권 없음 | **옮김** |
| 397 | `ch07/expectation_applications/conditional_expectation.md` | AI | 기계가 찾아내는 것은 결국 $E[Y\mid X]$ 하나다 | `ch03/rv/conditional_dist.md` | `—` | — | **옮김** |
| 398 | `ch07/expectation_applications/expectation_counting.md` | 그림 | 가장자리는 싸고 안쪽은 비싸다 | `ch03/rv/joint.md` | `ec__hollerith1890.png` | Wikimedia Commons · 저작권 없음 | **옮김** |
| 399 | `ch07/expectation_applications/expectation_counting.md` | AI | 자료를 섞어 쓰는 일이 괜찮은 까닭 | `ch05/foundations/repeated_sampling.md` | `—` | — | **옮김** |
| 400 | `ch07/expectation_applications/expectation_equation.md` | 그림 | 차수를 백 배로 올려도 실근은 셋쯤 늘 뿐이다 | `—` | `ea__ee_kac.png` | 난수 다항식의 실근 개수(카츠) — 걸릴 절이 없다 | 보류 |
| 401 | `ch07/expectation_applications/expectation_geometry.md` | 그림 | 제곱을 떼는 순간 손이 닿지 않는다 | `ch02/spread/variance_std.md` | `—` | 이 책에서 그렸다 | **옮김** |
| 402 | `ch07/expectation_applications/expectation_process.md` | 그림 | 「최대 $n$ 번까지만」을 떼어 내면 벌어지는 일 | `ch03/limits/gambler_paradox_lln.md` | `ea__dbernoulli.png` | ETH 취리히 도서관 소장, Wikimedia Commons, PD-old | **옮김** |
| 403 | `ch07/expectation_applications/expected_profit.md` | 그림 | 처음 이 셈을 한 곳은 가게가 아니라 은행 금고였다 | `ch05/foundations/financial_crisis_clt.md` | `ea__edgeworth1888_p113.png` | Internet Archive · Google Books 스캔, 저작권 없음 | **옮김** |
| 404 | `ch07/expectation_applications/order_statistics.md` | 그림 | 전차를 세지 않고 전차 수를 센 사람들 | `ch06/estimation_methods/moments_examples.md` | `ea__panther1944.png` | Archives Normandie, Wikimedia Commons, PD | **옮김** |
| 405 | `ch07/expectation_applications/variance_optimization.md` | 그림 | 「서로 독립이면」이라는 단서에 상을 준 까닭 | `ch03/rv/variance_covariance.md` | `ea__vo_markowitz.png` | 이 책에서 그렸다 | **옮김** |
| 406 | `ch07/expectation_variance/distribution_table_variance.md` | 그림 | 손에게 좋은 공식이 기계에게는 나쁜 공식이다 | `ch07/variance/naive.md` | `ev__welford.png` | 이 책에서 그렸다 | **옮김** |
| 407 | `ch07/expectation_variance/expectation.md` | 그림 | 확률보다 기댓값이 먼저 인쇄되었다 | `ch03/rv/expectation.md` | `ev__huygens1657.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 408 | `ch07/expectation_variance/expectation_coin_and_dice.md` | 그림 | 「공정하다」를 이만 육천 번 던져 재어 본 사람 | `ch10/test/gof.md` | `ev__weldon1906.png` | Wikimedia Commons, PD-US | **옮김** |
| 409 | `ch07/expectation_variance/indicator_linearity.md` | AI | 점수는 더해서 나오지만 믿음은 거기서 나오지 않는다 | `ch03/rv/expectation.md` | `ev__ai_indicator_dep.png` | — | **옮김** |
| 410 | `ch07/expectation_variance/indicator_linearity.md` | 그림 | 섞었다고 믿은 날 | `ch17/permutation/foundations.md` | `ev__lottery1969.png` | Wikimedia Commons, PD-USGov | **옮김** |
| 411 | `ch07/expectation_variance/indicator_variance.md` | AI | 답을 여러 개 모아도 줄지 않는 몫 | `ch11/assumptions/independence.md` | `ev__ai_ensemble_floor.png` | — | **옮김** |
| 412 | `ch07/expectation_variance/indicator_variance.md` | 그림 | 오리를 세지 않고 오리 수를 세는 법 | `ch06/mle/capture_recapture.md` | `ev__lincoln1930.png` | Internet Archive 스캔, 미국 정부 간행물 | **옮김** |
| 413 | `ch07/expectation_variance/variance.md` | 그림 | 표준편차가 아니라 분산에 이름을 붙인 까닭 | `ch03/rv/variance_covariance.md` | `ev__fisher1918_variance.png` | 애들레이드 대학 R. A. Fisher Digital Archive 의 스캔 | **옮김** |
| 414 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 금 같은 정리 | `ch03/limits/lln.md` | `lln__arsconjectandi1713.png` | Wikimedia Commons, PD-old-100 | **옮김** |
| 415 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 부등식 하나로 밀어붙인 사람 | `ch03/limits/lln.md` | `lln__stamp_russia2021.png` | Wikimedia Commons, PD | **옮김** |
| 416 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 바늘로 잰 원주율 | `ch03/limits/lln.md` | `lln__buffon_statue.png` | Wikimedia Commons, 촬영 Celette, $2012$, CC BY-SA 4.0 | **옮김** |
| 417 | `ch07/law_of_large_numbers/law_of_large_numbers.md` | 그림 | 카드 점을 치다 나온 계산법 | `ch03/limits/lln.md` | `lln__fermiac.png` | 로스앨러모스 국립연구소, Wikimedia Commons, PD-USGov | **옮김** |
| 418 | `ch07/normal_distribution/binomial_to_normal.md` | 그림 | 반 칸에 사람 이름이 붙은 자리 | `ch10/practice/validity_conditions.md` | `nd__yates_gain.png` | 이 책에서 그렸다 | **옮김** |
| 419 | `ch07/normal_distribution/central_limit_theorem.md` | 그림 | 종 모양이 활자로 놓인 쪽 | `ch03/limits/clt.md` | `clt__doctrine1756_p243.png` | Internet Archive 원문 스캔, PD-old-100 | **옮김** |
| 420 | `ch07/normal_distribution/normal_applications.md` | 그림 | 같은 높이로 튀었는데 한쪽만 이상이다 | `ch05/applications/phat_n.md` | `nd__shewhart1927_p726.png` | Internet Archive 스캔, 저작권 없음 | **옮김** |
| 421 | `ch07/normal_distribution/normal_distribution.md` | 그림 | 지폐에 실린 확률밀도함수 | `ch04/continuous_distributions/normal.md` | `nd__note.png` | — | **옮김** |
| 422 | `ch07/normal_distribution/normal_distribution.md` | 그림 | 구슬이 쌓아 올린 종 모양 | `ch04/discrete_distributions/binomial.md` | `nd__galton.png` | Wikimedia Commons, PD | **옮김** |
| 423 | `ch07/normal_distribution/normal_sum_percentile.md` | 그림 | 분산이 같은데 모양이 다르다 | `ch03/limits/clt.md` | `nd__lindeberg.png` | 이 책에서 그렸다 | **옮김** |
| 424 | `ch07/normal_distribution/normal_sum_percentile.md` | AI | 층을 쌓을 때 $\sqrt n$ 으로 나누는 까닭 | `ch04/continuous_distributions/normal.md` | `—` | — | **옮김** |
| 425 | `ch07/normal_distribution/normal_symmetry.md` | 그림 | 대칭을 눈금으로 새겨 둔 종이 | `ch14/graphical_methods/qq_plots.md` | `nd__hazen.png` | 이 책에서 그렸다 | **옮김** |
| 426 | `ch07/normal_distribution/normal_symmetry.md` | AI | 눈금을 맞추지 않으면 큰 눈금만 보게 된다 | `ch05/distributions/normal.md` | `—` | — | **옮김** |
| 427 | `ch07/normal_distribution/repeated_trial_normal.md` | 그림 | 그 표를 만든 사람들은 음수를 다룰 줄 몰랐다 | `ch05/distributions/normal.md` | `nd__tables.png` | 이 책에서 그렸다 | **옮김** |
| 428 | `ch07/random_variable/continuous_random_variable.md` | AI | 상수를 구하지 못한 채 쓰는 밀도 | `ch03/rv/continuous.md` | `rv__crv_ai_ratio.png` | — | **옮김** |
| 429 | `ch07/random_variable/continuous_random_variable.md` | 그림 | 대칭인데 평균이 없는 밀도 | `ch03/rv/continuous.md` | `rv__crv_cauchy.png` | 이 책에서 그렸다 | **옮김** |
| 430 | `ch07/random_variable/cumulative_distribution.md` | 그림 | 잡화상이 만든 첫 계단 | `ch03/rv/pmf_pdf_cdf.md` | `rv__graunt1662_p62.png` | $62$ / Internet Archive, 저작권 없음 | **옮김** |
| 431 | `ch07/random_variable/geometric_probability.md` | 그림 | 세 답을 나란히 적고 물음이 틀렸다고 한 쪽 | `ch03/probability/sample_space.md` | `rv__bertrand1907_p5.png` | Wikimedia Commons, PD-US-expired | **옮김** |
| 432 | `ch07/random_variable/probability_density_applications.md` | 그림 | 넓이를 지키려고 위도를 눌러 놓은 지도 | `—` | `rv__pda_equalarea1921.png` | 구면 띠 넓이와 등적 도법 — 걸릴 정의·정리가 없다 | 보류 |
| 433 | `ch07/random_variable/probability_density_applications.md` | AI | 공 위에 점을 고르게 뿌리려면 위도를 버려야 한다 | `ch04/continuous_distributions/uniform.md` | `rv__pda_ai_sphere.png` | — | **옮김** |
| 434 | `ch07/random_variable/probability_density_function.md` | AI | 어느 자 위에서 균등하게 뽑을 것인가 | `ch18/ridge/lambda_selection.md` | `rv__pdf_ai_logscale.png` | — | **옮김** |
| 435 | `ch07/random_variable/probability_density_function.md` | 그림 | 측량 오차를 따지다 나온 분포 | `ch04/continuous_distributions/chi_square.md` | `rv__pdf_helmert.png` | 이 책에서 그렸다 | **옮김** |
| 436 | `ch07/random_variable/probability_density_graph.md` | 그림 | 눈금을 바꾸면 보이는 직선 | `ch07/non_normal/heavy_tails.md` | `rv__pdg_pareto.png` | 이 책에서 그렸다 | **옮김** |
| 437 | `ch07/random_variable/probability_mass_function.md` | 그림 | 질량이 밀도로 건너가는 자리 | `ch03/rv/continuous.md` | `rv__simpson1757_p72.png` | Internet Archive 스캔, PD-old | **옮김** |
| 438 | `ch07/random_variable/probability_mass_function.md` | AI | 꼬리를 자르면 확률이 아니게 된다 | `ch03/rv/discrete.md` | `rv__pmf_ai_renorm.png` | — | **옮김** |
| 439 | `ch07/random_variable/probability_mass_function.md` | AI | 합이 $1$ 이라고 해서 확률인 것은 아니다 | `ch03/rv/discrete.md` | `—` | — | **옮김** |
| 440 | `ch07/random_variable/random_variable.md` | 그림 | 주사위 셋의 합을 칠백 해 전에 적어 둔 표 | `ch03/rv/discrete.md` | `rv__devetula1534.png` | Internet Archive 스캔, 퍼블릭 도메인 표시 | **옮김** |
| 441 | `ch07/random_variable/random_variable_geometry.md` | 그림 | 「그때에야 비로소 셀 수 있다」 | `ch03/probability/sample_space.md` | `rv__crofton1868_p181.png` | Internet Archive 스캔, PD-old | **옮김** |
| 442 | `ch07/random_variable/random_variable_process.md` | 그림 | 시 두 만 자를 손으로 세어 얻은 두 수 | `ch16/one_sample_nonparametric/runs_test.md` | `rv__markov1913_p153.png` | Internet Archive 스캔, PD-old | **옮김** |
| 443 | `ch07/random_variable/uniform_distribution.md` | 그림 | 고르게 뽑으려고 돌을 깎았다 | `ch04/continuous_distributions/uniform.md` | `rv__kleroterion.png` | 촬영 Zde, Wikimedia Commons, CC BY-SA 4.0 | **옮김** |
| 444 | `ch07/random_variable/uniform_distribution.md` | AI | 난수 하나로 다음 낱말을 고르는 법 | `ch04/continuous_distributions/uniform.md` | `rv__uni_ai_inverse.png` | — | **옮김** |
| 445 | `ch07/random_walk/conditional_sequential.md` | 그림 | 가장 센 패턴이 없는 놀이 | `—` | `rw__penney.png` | 페니 게임·비추이성 — 그것을 다루는 쪽이 없다 | 보류 |
| 446 | `ch07/random_walk/first_step_analysis.md` | 그림 | 「밑천이 무한하다면」이라고 적어 놓고 넘어간 쪽 | `—` | `fs__eb1911_p771.png` | 도박꾼의 파산·흡수확률 — 그것을 다루는 쪽이 없다 | 보류 |
| 447 | `ch07/random_walk/first_step_analysis.md` | AI | 기계는 이 점화식을 풀지 않고 외워 넣는다 | `ch01/paradigms/reinforcement.md` | `—` | — | **옮김** |
| 448 | `ch07/random_walk/graph_walk.md` | 그림 | 지나온 길은 묻지 않는다고 적어 둔 놀이판 | `—` | `gw__gyanchaupar.png` | 그래프 위 임의보행·마르코프 성질 — 그것을 다루는 쪽이 없다 | 보류 |
| 449 | `ch07/random_walk/graph_walk.md` | AI | 링크를 따라 걷는 사람이 매기는 순위 | `—` | `—` | 정상분포·페이지랭크 — 걸릴 정의·정리가 없다 | 보류 |
| 450 | `ch07/random_walk/lattice_walk.md` | 그림 | 모눈종이 위에 연필로 그린 임의 보행 | `ch05/applications/sample_mean.md` | `rw__perrin1916_p115.png` | Internet Archive, PD-US-expired | **옮김** |
| 451 | `ch07/random_walk/line_walk.md` | 그림 | 취한 사람은 집에 오고 취한 새는 못 온다 | `—` | `rw__polya.png` | 폴리아 재귀성 정리 — 걸릴 자리가 없다 | 보류 |
| 452 | `ch07/random_walk/trial_state.md` | 그림 | 홀짝이 매번 뒤집히는 항아리로 물리학의 다툼을 가라앉히다 | `—` | `rw__ehrenfest.png` | 에렌페스트 연쇄·되돌아옴 시간 — 그것을 다루는 쪽이 없다 | 보류 |
| 453 | `ch07/statistical_estimation/population_sample.md` | 그림 | 돌아온 비행기만 세었다 | `ch01/classical/survivorship_bias.md` | `se__wald1943.png` | 미국 정부 문서, 공개 배포 승인 | **옮김** |
| 454 | `ch07/statistical_estimation/population_sample.md` | 그림 | 눈을 가리면 고르게 섞이는가 | `ch01/classical/surveys.md` | `se__draft_lottery.png` | 미국 국립문서보관소 $165$-WW-$420$(P$379$), Wikimedia Commons, PD-USGov | **옮김** |
| 455 | `ch07/statistical_estimation/population_sample.md` | 그림 | 사람을 똑같은 그림으로 세면 | `ch01/classical/population_sample.md` | `se__stamp_india1977.png` | Wikimedia Commons, GODL-India | **옮김** |
| 456 | `ch07/statistical_estimation/proportion_difference.md` | 그림 | 이백삼십팔만 명에게 물어보고 틀렸다 | `ch05/applications/sample_proportion.md` | `se__truman1948.png` | Wikimedia Commons, PD-US-no notice | **옮김** |
| 457 | `ch07/statistical_estimation/proportion_difference.md` | 그림 | 두 병동의 수가 갈라져 있었다 | `ch05/applications/diff_proportions.md` | `se__semmelweis1861_p3.png` | Wellcome Library 스캔 | **옮김** |
| 458 | `ch07/statistical_estimation/sampling_estimation.md` | 그림 | 「확률」이라 부르지 않으려고 | `ch08/foundations/interpretation.md` | `se__neyman.png` | Wikimedia Commons, 촬영 Konrad Jacobs, 오버볼파흐 수학연구소, CC BY-SA 2.0 DE | **옮김** |
| 459 | `ch07/statistical_estimation/standard_error.md` | 그림 | 맥주 회사가 이름을 쓰지 못하게 했다 | `ch05/applications/standard_error.md` | `se__student1908_p1.png` | Internet Archive 스캔, $1908$ 년 발행 | **옮김** |
| 460 | `ch08/algorithms/algorithms.md` | 그림 | 기계가 밟을 절차를 표로 적은 쪽 | `—` | `al__lovelace1843_noteg.png` | 러브레이스의 주석 G — 의사코드를 세우는 쪽이 없다 | 보류 |
| 461 | `ch08/algorithms/algorithms.md` | 그림 | 사람 육만 사천 명이 밟을 절차 | `—` | `al__richardson1922_p184.png` | 리처드슨의 수치예보 — 대응하는 통계 내용 없음 | 보류 |
| 462 | `ch08/algorithms/recurrences.md` | 그림 | 정리를 장난감으로 만들어 팔았다 | `—` | `recurrences__lucas1883.png` | 하노이의 탑 — 대응하는 통계 내용 없음 | 보류 |
| 463 | `ch08/algorithms/searching.md` | 그림 | 열여섯 해 동안 아무도 제대로 적지 못한 네 줄 | `—` | `al__knuth2005.png` | 이분탐색의 불변식 — 알고리즘 정확성을 다루는 쪽이 없다 | 보류 |
| 464 | `ch08/graphs/graphs.md` | 그림 | 다리 일곱 개에서 시작한 것 | `—` | `gr__euler1741_p128.png` | 쾨니히스베르크의 다리 — ch12 의 DAG 는 인과그래프다 | 보류 |
| 465 | `ch08/logic/logic.md` | 그림 | 「모두」를 원 안에 집어넣은 사람 | `ch00/math/logic_sets.md` | `lg__euler_circles_1768.png` | — | **옮김** |
| 466 | `ch08/number_theory/binary_digits.md` | 그림 | 셈하려고 그린 것이 아닌 $0$ 부터 $63$ 까지 | `—` | `nt__fuxi_leibniz1701.png` | 소옹의 육십사괘 — 대응하는 통계 내용 없음 | 보류 |
| 467 | `ch08/number_theory/combinatorial_nt.md` | 그림 | 상자도 비둘기도 없이, 머리카락으로 적은 증명 | `—` | `nt__recreation1629_p205.png` | 비둘기집 원리의 가장 이른 논증 — 그 원리가 이 책에 없다 | 보류 |
| 468 | `ch08/number_theory/congruence.md` | 그림 | 기호 $\equiv$ 가 처음 인쇄된 쪽 | `—` | `cong__gauss1801_p2.png` | 가우스의 합동 기호 — 대응하는 통계 내용 없음 | 보류 |
| 469 | `ch08/number_theory/congruence.md` | 그림 | 사람이 끼어들 자리를 없앤 개혁 | `—` | `nt__fasti_antiates.png` | 율리우스·그레고리력 — 대응하는 통계 내용 없음 | 보류 |
| 470 | `ch08/number_theory/congruence.md` | 그림 | 이름에 나라가 들어간 까닭 | `—` | `nt__sunzi_suanjing.png` | 중국인의 나머지 정리 — 대응하는 통계 내용 없음 | 보류 |
| 471 | `ch08/number_theory/digits_bases.md` | 그림 | 점 하나가 수가 되기까지 | `—` | `nt__bakhshali.png` | 바흐샬리 사본의 0 — 대응하는 통계 내용 없음 | 보류 |
| 472 | `ch08/number_theory/digits_bases.md` | 그림 | 자릿값으로 셈은 했는데 적지는 못했다 | `—` | `nt__roman_abacus.png` | 로마 손주판 — 대응하는 통계 내용 없음 | 보류 |
| 473 | `ch08/number_theory/digits_bases.md` | 그림 | $\dfrac{1}{10}$ 을 끝까지 적지 못한 기계 | `ch20/softmax_regression/numerical_stability.md` | `nt__patriot1992_p15.png` | 미국 정부 보고서, PD | **옮김** |
| 474 | `ch08/number_theory/diophantine.md` | 그림 | 등호를 지워 버린 빨간 줄 | `—` | `nt__stamp_czech2000.png` | 페르마의 마지막 정리 우표 — 대응하는 통계 내용 없음 | 보류 |
| 475 | `ch08/number_theory/diophantine.md` | 그림 | 정사각형의 대각선을 적어 본 사천 년 전 사람 | `—` | `nt__ybc7289.png` | YBC 7289 의 √2 — 대응하는 통계 내용 없음 | 보류 |
| 476 | `ch08/number_theory/diophantine.md` | 그림 | 지어낸 초상, 지어낸 이야기 | `—` | `nt__hippasus.png` | 히파소스 일화의 허구성 — 대응하는 통계 내용 없음 | 보류 |
| 477 | `ch08/number_theory/divisors_primes.md` | 그림 | 시시한 수는 없다 | `—` | `nt__stamp_india1962.png` | 1729 와 라마누잔 — 대응하는 통계 내용 없음 | 보류 |
| 478 | `ch08/number_theory/divisors_primes.md` | 그림 | 바닥 기호가 생기기 백오십 년 전에 적힌 바닥함수의 합 | `—` | `nt__legendre1808_p9.png` | 르장드르의 소인수 지수 공식 — 대응하는 통계 내용 없음 | 보류 |
| 479 | `ch09/game_theory/constant_sum_game.md` | 그림 | 신분증 사진 한 장과 미니맥스 정리 | `—` | `gt__vonneumann.png` | 게임이론·미니맥스를 다루는 장이 없다 | 보류 |
| 480 | `ch09/game_theory/dominant_strategy.md` | 그림 | 스물여섯 쪽짜리 학위논문 | `—` | `gt__nash.png` | 내시균형을 다루는 장이 없다 | 보류 |
| 481 | `ch09/game_theory/game_theory.md` | 그림 | 두 곡선이 만나는 점, 내시보다 백열두 해 먼저 | `—` | `gt__cournot1838_fig2.png` | 최적반응·쿠르노 균형을 다루는 장이 없다 | 보류 |
| 482 | `ch09/game_theory/mixed_strategy.md` | 그림 | 「얼마나 더 자주인지는 계산이 알려 주지 않는다」 | `—` | `gt__montmort1713_p409.png` | 혼합전략·무차별 원리를 다루는 장이 없다 | 보류 |
| 483 | `ch09/game_theory/payoff_utility.md` | 그림 | 곱하지 않은 목록 | `ch03/rv/variance_covariance.md` | `gt__bentham_ch4.png` | Internet Archive, 저작권 없음 | **옮김** |
| 484 | `elementary_geometry/angles/angles.md` | 그림 | $360$ 은 땅이 아니라 하늘에서 왔다 | `—` | `ang__mulapin.png` | 각의 단위 360 의 내력 — 대응하는 통계 내용 없음 | 보류 |
| 485 | `elementary_geometry/area/circle_area.md` | 그림 | 「둘레는 서른, 지름은 열」 | `—` | `ca__molten_sea.png` | 솔로몬의 놋대야와 π=3 — 대응하는 통계 내용 없음 | 보류 |
| 486 | `elementary_geometry/area/circle_area.md` | 그림 | 묘비에 새긴 것이 이름이 아니라 두 개의 부등호였다 | `—` | `ca__ceulen_tomb.png` | 판 쾰런의 35 자리 π — 대응하는 통계 내용 없음 | 보류 |
| 487 | `elementary_geometry/area/circle_area.md` | 그림 | 머리카락 굵기만큼만 틀리게 | `—` | `ca__kashi_muhitiyya.png` | 알카시의 π — 대응하는 통계 내용 없음 | 보류 |
| 488 | `elementary_geometry/area/circle_area.md` | 그림 | 신사에 걸어 둔 문제판 | `—` | `ca__sangaku.png` | 산가쿠 — 대응하는 통계 내용 없음 | 보류 |
| 489 | `elementary_geometry/area/perimeter.md` | 그림 | 벽에 걸어 둔 자 | `—` | `pm__metre.png` | 미터원기 — 척도를 세우는 쪽이 없다 | 보류 |
| 490 | `elementary_geometry/area/polygon_area.md` | 그림 | 우표에 실린 「거의」 정사각형 | `—` | `ar__stamp_germany1998.png` | 정사각형 조각내기 — 대응하는 통계 내용 없음 | 보류 |
| 491 | `elementary_geometry/centers/centroid.md` | 그림 | 지구를 들어 올리는 지렛대 | `—` | `cen__archimedes_lever1824.png` | 아르키메데스의 지렛대 — 알맹이가 지렛대 원리다 | 보류 |
| 492 | `elementary_geometry/centers/euler_nine_point.md` | 그림 | 여섯 거리를 한자리에 늘어놓았더니 보인 것 | `—` | `cen__euler1767_p114.png` | 오일러 직선 — 대응하는 통계 내용 없음 | 보류 |
| 493 | `elementary_geometry/centers/orthocenter.md` | 그림 | 「그렇게 부르면 편리할 것이다」 | `—` | `cen__ferrers1866_p119.png` | 수심 명명 — 대응하는 통계 내용 없음 | 보류 |
| 494 | `elementary_geometry/centers/tangent_centers.md` | 그림 | 이름 넷을 청해 셋이 살아남았다 | `—` | `cen__hudson1883_nature.png` | 외접원·내접원 명명 — 대응하는 통계 내용 없음 | 보류 |
| 495 | `elementary_geometry/circle/circle.md` | 그림 | 이 원 위에 서면 자기 위치를 알 수 없다 | `—` | `ci__station_pointer.png` | 위험원(후방교회법) — 기제가 원주각이다 | 보류 |
| 496 | `elementary_geometry/circle/circle_packing.md` | 그림 | 대리석에서 원기둥 셋을 파내는 법을 적어 놓고, 틀렸다 | `—` | `cp__malfatti_p235.png` | 말파티의 원 — 대응하는 통계 내용 없음 | 보류 |
| 497 | `elementary_geometry/congruence/congruence.md` | 그림 | 포개어 보는 것이 증명이 되는가 | `—` | `cong__hilbert1899_p12.png` | 힐베르트의 합동 공리 — 대응하는 통계 내용 없음 | 보류 |
| 498 | `elementary_geometry/construction/construction.md` | 그림 | 여백에 그림을 앉힌 첫 인쇄본 | `—` | `cons__euclid1482.png` | 1482 년 『원론』 인쇄본 — 대응하는 통계 내용 없음 | 보류 |
| 499 | `elementary_geometry/construction/construction.md` | 그림 | 둘 가운데 하나는 없어도 된다 | `—` | `cons__mascheroni1797.png` | 모어–마스케로니 정리 — 대응하는 통계 내용 없음 | 보류 |
| 500 | `elementary_geometry/folding/fold_angle.md` | 그림 | 한 동작으로 펴지는 접기 무늬 | `—` | `fold__miura.png` | 미우라 접기 — 대응하는 통계 내용 없음 | 보류 |
| 501 | `elementary_geometry/folding/fold_area.md` | 그림 | 반으로 접어도 닮은 종이 | `—` | `fold__a_series.png` | A 판형과 √2 — 대응하는 통계 내용 없음 | 보류 |
| 502 | `elementary_geometry/pythagoras/pythagoras.md` | 그림 | 얼굴이 아니라 정리를 새겼다 | `—` | `pyth__stamp_greece1955.png` | 피타고라스 정리 우표 — 이 책의 직교분해와 다른 이야기다 | 보류 |
| 503 | `elementary_geometry/pythagoras/pythagoras.md` | 그림 | 빗변 위의 정사각형 하나로 끝낸 그림 | `—` | `pyth__zhoubi_xiantu.png` | 조상의 구고원방도 — 대응하는 통계 내용 없음 | 보류 |
| 504 | `elementary_geometry/quadrilateral/quadrilateral.md` | 그림 | 밧줄로는 변의 길이밖에 잴 수 없다 | `—` | `qd__menna.png` | 이집트 사각형 넓이 공식 — 결정론적 기하 부등식이다 | 보류 |
| 505 | `elementary_geometry/similarity/similarity.md` | 그림 | 그림자로 잰 피라미드 | `—` | `sim__pyramid.png` | 탈레스의 닮음 — 대응하는 통계 내용 없음 | 보류 |
| 506 | `elementary_geometry/similarity/similarity.md` | 그림 | 닿을 수 없으면 두 번 재면 된다 | `—` | `sim__haidao_wanghaidao.png` | 『해도산경』의 중차술 — 대응하는 통계 내용 없음 | 보류 |
| 507 | `elementary_geometry/similarity/similarity.md` | 그림 | 이름은 지름을 따랐고 물은 넓이를 따랐다 | `—` | `sim__frontinus1722_p109.png` | 프론티누스의 수도관 — 대응하는 통계 내용 없음 | 보류 |
| 508 | `elementary_geometry/triangle_angles/triangle_angles.md` | 그림 | 증명하지 못하고 「본성에 어긋난다」고 적은 쪽 | `—` | `tri__saccheri_prop33.png` | 사케리와 평행선 공준 — 대응하는 통계 내용 없음 | 보류 |
| 509 | `elementary_geometry/triangle_sides/triangle_sides.md` | 그림 | 글자를 버리고 색으로 적은 증명 | `—` | `side__byrne1847_prop5.png` | 바인의 색판 『원론』 — 대응하는 통계 내용 없음 | 보류 |
| 510 | `elementary_geometry/trig_ratio/trig_ratio.md` | 그림 | 「사인」이라는 말에는 뜻이 없다 | `—` | `trg__aryabhatiya1874_p17.png` | 사인의 어원 — 대응하는 통계 내용 없음 | 보류 |
| 511 | `elementary_geometry/trig_ratio/trig_ratio.md` | 그림 | 산 하나에 올라 지구를 쟀다 | `—` | `trg__stamp_uzbekistan2023.png` | 알비루니의 지구 반지름 — 본체가 삼각비다 | 보류 |
| 512 | `elementary_geometry/volume/volume_basics.md` | 그림 | 사천 년 전에 이미 적혀 있던 $\dfrac13$ | `—` | `vb__moscow_p14.png` | 모스크바 파피루스 제14문제 — 대응하는 통계 내용 없음 | 보류 |
| 513 | `elementary_geometry/volume/volume_basics.md` | 그림 | 묘비에 새겨 달라던 그림 | `—` | `vb__tomb.png` | 아르키메데스의 구와 원기둥 — 대응하는 통계 내용 없음 | 보류 |
| 514 | `elementary_geometry/volume/volume_basics.md` | 그림 | 거인이 있을 수 없는 까닭을 그린 쪽 | `—` | `vb__galileo_bone.png` | 갈릴레오의 제곱–세제곱 법칙 — 대응하는 통계 내용 없음 | 보류 |
| 515 | `numbers/arithmetic/rational_arithmetic.md` | 그림 | $+$ 와 $-$ 가 처음 인쇄된 쪽 | `—` | `ra__widmann1489.png` | + · − 기호의 내력 — 대응하는 통계 내용 없음 | 보류 |
| 516 | `numbers/arithmetic/square_root.md` | 그림 | 잴 수 없는 것을 비교하는 법 | `—` | `sq__euclid_bookV.png` | 에우독소스의 비례론 — 실수의 구성을 다루는 쪽이 없다 | 보류 |
| 517 | `numbers/arithmetic/square_root.md` | 그림 | 소수점이 없던 시절의 소수 | `—` | `sq__thiende1585.png` | 스테빈의 『디 틴더』 — 대응하는 통계 내용 없음 | 보류 |
| 518 | `numbers/arithmetic/unit_fractions.md` | 그림 | 욕심쟁이는 일곱 가운데 마지막이었다 | `—` | `uf__fibonacci1857_p82.png` | 피보나치의 단위분수 — 대응하는 통계 내용 없음 | 보류 |
| 519 | `numbers/expression/letters_expressions.md` | 그림 | 모음은 모르는 것, 자음은 아는 것 | `—` | `ex__viete.png` | 비에트의 문자 사용 — 대응하는 통계 내용 없음 | 보류 |
| 520 | `numbers/expression/letters_expressions.md` | 그림 | 모르는 수에 글자 대신 자리를 준 사람들 | `—` | `le__siyuan_tianyuan.png` | 천원술 — 대응하는 통계 내용 없음 | 보류 |
| 521 | `numbers/expression/polynomial_operations.md` | 그림 | 곱셈 기호가 셋이던 쪽 | `—` | `po__oughtred1667_p10.png` | 오트레드의 곱셈 기호 — 대응하는 통계 내용 없음 | 보류 |

합계 **521** 개 — 옮김 **170**, 보류 **351**, 대기 **0**
