# 시뮬레이션 실행 설명



## 실행 순서



```

1. extract_solar_weather.py    →  data/solar/ + data/weather/

2. extract_prices_data.py      →  data/prices/

3. merge_for_simulation.py     →  data/merged_for_simulation.csv

4. run_simulation.py           →  results/simulation_output/

```



각 스크립트를 위 순서대로 실행해야 한다. 앞 단계 출력이 뒤 단계 입력이 된다.



---



## 1. extract_solar_weather.py



GEFCom2014 Solar Track 원본 파일 하나에서 solar 발전량과 기상 변수 두 dataset을 분리해 뽑아낸다.



| 항목 | 내용 |

|------|------|

| 입력 | `data_raw/gefcom2014/GEFCom2014/GEFCom2014 Data/GEFCom2014-S_V2/Solar/Task 15/predictors15.csv` |

| 출력 | `data/solar/solar-energy-generation.csv` (TIMESTAMP + solar_power) |

| 출력 | `data/weather/weather_data.csv` (TIMESTAMP + 기상 변수 12개) |



- `predictors15.csv`는 ZONEID + 기상 변수 12개 + POWER 가 모두 함께 있는 파일

- Zone 1 만 사용

- solar_power는 0~1 로 이미 정규화됨

- 기상 변수 12개: VAR78, VAR79, VAR134, VAR157, VAR164, VAR165, VAR166, VAR167, VAR169, VAR175, VAR178, VAR228



---



## 2. extract_prices_data.py



MISO Settlement Point Prices Excel 파일에서 MISO System 허브의 가격 데이터만 뽑아냄.



| 항목 | 내용 |

|------|------|

| 입력 | `data_raw/miso/raw/201*_da_pr_xls/` (하루 당 Excel 1개) |

| 입력 | `data_raw/miso/raw/201*_rt_pr_xls/` (하루 당 Excel 1개) |

| 출력 | `data/prices/day_ahead_prices.csv` (TIMESTAMP + DA_LMP) |

| 출력 | `data/prices/real_time_prices.csv` (TIMESTAMP + RT_LMP) |



- Excel 파일마다 행에서 'MISO System' 문자를 찾아 가격 시작 위치를 동적으로 결정

- 파일 이름이 아닌 Excel 내부 Market Date 를 기준으로 날짜 추출

- glob 패턴 `201*_da_pr_xls` / `201*_rt_pr_xls` 로 2010~2019 년 모든 월별 폴더 자동 포함

- DA 와 RT 는 구조가 같아 `_extract_miso_prices()` 공통 함수 하나로 처리



---



## 3. merge_for_simulation.py



위 두 단계에서 만든 4개 파일을 하나의 CSV 로 합친다.



| 항목 | 내용 |

|------|------|

| 입력 | `data/solar/solar-energy-generation.csv` |

| 입력 | `data/prices/day_ahead_prices.csv` |

| 입력 | `data/prices/real_time_prices.csv` |

| 입력 | `data/weather/weather_data.csv` |

| 출력 | `data/merged_for_simulation.csv` |



- timestamp 기준으로 4개 파일 left merge

- weather 파일의 익명 변수명 (VAR78 등) 을 실제 ECMWF 명칭으로 변경 (VAR78 → tclw, VAR79 → tciw 등)

- RT 가격 마지막 날 누락 시 forward fill 로 보정

- 최종 컬럼: timestamp, solar_power, da_price, rt_price, tclw, tciw, sp, r, tcc, u10, v10, t2m, ssrd, strd, tsr, tp



---



## 4. run_simulation.py



합쳐진 데이터로 AR, MLR, 제안 방식 세 모델을 학습하고 결과를 낸다.



| 항목 | 내용 |

|------|------|

| 입력 | `data/merged_for_simulation.csv` |

| 출력 | `results/simulation_output/ar_results.csv` |

| 출력 | `results/simulation_output/mlr_results.csv` |

| 출력 | `results/simulation_output/proposed_results.csv` |



- UTC 00:00~11:00 하루 12시간 만 사용 (호주 현지 낮 시간)

- 1~3 월 학습, 4 월 테스트

- AR 모델: direct multi-step 방식으로 시간대 당 별도 LinearRegression (과거 12일 같은 시간 사용)

- MLR 모델: 기상 변수 12개를 입력으로 시간대 당 별도 LinearRegression

- 제안 방식: 최적화 기반 AR — optimality gap 와 forecasting error 을 통합 목적함수로 계수 학습

- penalty rate 0%~100% 11 단계 스윕

- 제안 방식은 W1/W2 비율 10 가지 조합으로 실행



### 평가지표



- **nRMSE**: 정규화 RMSE, 예측 정확도

- **Optimality Gap**: (최적 이익 - 실제 이익) / 최적 이익 × 100, commitment 전략의 이익 손실률



---



## 논문 결과와 다른 원인 분석



현재 merged dataset 은 구조적으로 올바르다. 데이터 자체의 오류는 확인되지 않았다.



| 확인 항목 | 결과 |

|-----------|------|

| Solar 패턴 | UTC 00:00~05:00 고값, 11:00~20:00 0 → 호주 여름 패턴 정상 |

| Price 패턴 | DA 27~50, RT 23~42 → MISO 실제 가격 범위 정상 |

| Weather 변수 | 12개 변수, 값은 GEFCom2014 원본 그대로 |

| merge | timestamp 기준으로 4개 파일 정확히 합쳐짐 |

| 결측치 | RT price 4/30 하루 누락 → forward fill 로 보정됨 |

| UTC 필터링 | `run_simulation.py` 에서 `.between(0, 11)` 으로 낮 시간 정상 추출 |



그럼에도 불구하고 실행 결과가 논문과 2~2.5배 차이난다. 근본 원인은 다음과 같다.



### 1. 학습 기간 부족 (주 원인)



| | 논문 | 우리 |

|--|------|------|

| 학습 | 300일 | **90일** |

| 테스트 | 100일 | **30일** |



MISO 가격 데이터가 2014년 1~4월 4개월로 제한되어 있다. GEFCom2014 Solar/Weather 데이터는 2012.04~2014.07 전체 기간 존재하지만, 가격 데이터가 없으면 optimality gap 계산이 불가능하므로 학습 기간을 늘릴 수 없다.



### 2. MISO 허브/기간 불명



논문이 "MISO System" 전체가 아닌 특정 허브를 썼을 수 있고, 사용 기간도 "Data will be made available on request" 라고만 명시되어 있어 정확히 비교할 수 없다.



### 3. MLR backward stepwise 미적용



논문은 12개 기상 변수 중 SSRD, TSR, Hour 3개만 최종 선택했다. 우리는 12개 전 변수를 사용하므로 불필요한 변수가 예측 정확도를 떨어뜨릴 수 있다.



### 4. nRMSE 계산 방식 차이



하루 12시간에 대해 하루별 nRMSE 계산 후 평균하는지와, 전체 테스트 기간을 하나로 합쳐 계산하는지 논문이 명시하지 않았다.



---



## 실험 결과 요약 (2026-08-22, Run #1~#3)



### Run #1 (90일 학습)



| 모델 | 논문 | 실행 nRMSE | 실행 Gap |

|------|------|------------|----------|

| AR | 34.76% / 15.04% | 86.55% | 28.16% |

| MLR(12vars) | 21.76% / 12.59% | 50.49% | 16.06% |



**차이 원인**: 학습 데이터 수 부족, MISO 가격 데이터 기간 부족, nRMSE 계산 방식, AR 모델 구현, GEFCom2014 시간대



### Run #2 (90일 학습, MLR backward stepwise 적용 + nRMSE 두 가지 방식)



**변경사항**:

- MLR 변수: 12개 → ssrd, tsr, hour 3개 (backward stepwise, 논문과 동일)

- nRMSE 계산: 전체 합산 + 하루별 평균 후 전체 평균, 두 가지 모두 보고



| 모델 | 논문 기대 | 실행 (전체) | 실행 (day-avg) | Gap |

|------|-----------|-------------|----------------|-----|

| AR | 34.76% / 15.04% | 86.55% | 142.18% | 28.16% |

| MLR(12vars) | 21.76% / 12.59% | 50.49% | 68.94% | 16.06% |

| MLR(sw-3vars) | 21.76% / 12.59% | **45.07%** | **62.09%** | **15.02%** |



**결과**:

- MLR backward stepwise: nRMSE 50.49% → 45.07% (10.7% 개선), Gap 16.06% → 15.02% (4% 개선)

- 하지만 논문 결과와 여전히 2배 차이



### Run #3 (300일 학습, GEFCom2014 전체 기간, MISO 없이 solar+weather 만)



| 모델 | 논문 | 90일 | 300일 | 개선 |

|------|------|------|-------|------|

| AR | 34.76% | 86.55% | **49.33%** | ↓ 43% |

| MLR(12vars) | 21.76% | 50.49% | **35.67%** | ↓ 29% |

| MLR(sw-3vars) | 21.76% | 45.07% | **36.62%** | ↓ 19% |



**핵심: 학습 기간 부족이 가장 큰 원인.**



**nRMSE 계산 방식 3가지 비교 (300일 학습 기준)**:



| 모델 | 전체 합산 | day-avg | hour-avg |

|------|-----------|---------|----------|

| AR | 49.33% | 60.96% | **284.94%** |

| MLR(sw) | 36.62% | 40.67% | **69.83%** |



- hour-avg: night 시간(평균 발전량 ≈ 0)으로 분모가 너무 작아 극단적

- **전체 합산 방식**이 논문 값과 가장 가까움 → 논문의 계산 방식일 가능성 가장 높음



**MLR backward stepwise 효과学习环境 차이**:



| 학습 기간 | MLR(12vars) | MLR(sw) | 비교 |

|-----------|-------------|---------|------|

| 90일 | 50.49% | 45.07% | 3개 더 좋음 |

| 300일 | **35.67%** | 36.62% | **12개 더 좋음** |



- 학습 데이터가 부족하면 변수 3개만 남기는 것이 과적합 방지 효과

- 학습 데이터가 충분하면 12개 변수가 더 나은 예측 성능



---



## 참고



- `check_hourly.py`: merged 데이터의 시간대별 solar_power 패턴을 확인하는 일회성 디버깅 스크립트 (필수는 아님)