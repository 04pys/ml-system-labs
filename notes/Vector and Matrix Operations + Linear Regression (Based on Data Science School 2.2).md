# 벡터·행렬 연산 & 선형회귀 핵심 요약 (DataScienceSchool 2.2)

> 목표: 이후 ML/논문에서 나오는 선형대수 표기(내적, 행렬곱, 잔차)를 “보자마자 의미가 떠오르게” 만들기

## 0. 큰 그림
- 벡터/행렬도 사칙연산을 하며, 이를 통해 대량 계산을 간단한 수식으로 표현 :contentReference[oaicite:34]{index=34}
- NumPy로 동일 연산을 실습하며 감각을 붙임 :contentReference[oaicite:35]{index=35}

---

## 1. 덧셈/뺄셈: element-wise
- 같은 크기(차원)의 벡터/행렬끼리만 가능
- 같은 위치 원소끼리 더하고 빼는 “요소별(element-wise) 연산” :contentReference[oaicite:36]{index=36}

---

## 2. 스칼라 곱 & 브로드캐스팅
- 스칼라×벡터/행렬: 모든 원소에 스칼라가 곱해짐 :contentReference[oaicite:37]{index=37}
- 브로드캐스팅: 원래는 차원이 같아야 하는데, 관례적으로 스칼라를 1-벡터처럼 확장해 연산 허용 :contentReference[oaicite:38]{index=38}

---

## 3. 내적(dot product) = 벡터×벡터(핵심)
- 다루는 곱셈은 “내적(inner product)” 중심 :contentReference[oaicite:39]{index=39}
- 표기: `x^T y`, `x · y`, `<x, y>`는 같은 의미로 쓰임 :contentReference[oaicite:40]{index=40}
- 내적 조건(개념적으로)
  - 길이(차원)가 같아야 함
  - (표기상) 앞은 행벡터, 뒤는 열벡터 :contentReference[oaicite:41]{index=41}

---

## 4. 유사도(내적의 응용)
- 내적은 두 벡터가 “얼마나 닮았는지”를 측정하는 데도 사용 :contentReference[oaicite:42]{index=42}
- 대표: 코사인 유사도(cosine similarity)로 확장 가능 :contentReference[oaicite:43]{index=43}

---

## 5. 선형회귀(Linear Regression) = 가중합/내적
- 선형회귀는 독립변수 x로 종속변수 y를 예측하는 방법 중 하나 :contentReference[oaicite:44]{index=44}
- 예측치: 가중치 벡터 w와 입력 x의 가중합
  - 1) 스칼라 표기:  ŷ = w1 x1 + ... + wN xN :contentReference[oaicite:45]{index=45}
  - 2) 벡터 표기:  ŷ = w^T x (내적으로 표현) :contentReference[oaicite:46]{index=46}
- 신경망 관점에서 “가중치 곱 → 합” 구조와 동일한 뼈대 :contentReference[oaicite:47]{index=47}

---

## 6. 행렬곱(행렬×행렬) 정의
- C = AB일 때, c_ij는 A의 i번째 “행벡터”와 B의 j번째 “열벡터”의 내적 :contentReference[oaicite:48]{index=48}
- 성립 조건: A의 열 수 = B의 행 수 :contentReference[oaicite:49]{index=49}
- 이 정의 하나로 “대량의 선형결합을 동시에 계산” 가능해짐

---

## 7. 잔차(residual): 예측과 정답의 차이
- 회귀 결과는 w로 나타나고, 예측치는 ŷ_i = w^T x_i 형태 :contentReference[oaicite:50]{index=50}
- 잔차(오차): e_i = y_i - ŷ_i = y_i - w^T x_i :contentReference[oaicite:51]{index=51}
- 전체 데이터로 묶으면 잔차 벡터 e를 만들고, (표기상) y - Xw 형태로 간단히 표현 가능 :contentReference[oaicite:52]{index=52}
