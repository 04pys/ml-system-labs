# Pandas / NumPy / Matplotlib 핵심 요약 (Wikidocs 217157)

> 목표: Colab에서 데이터 다루기/계산/시각화의 최소 핵심을 빠르게 복습하기

## 0. 임포트 규칙(관례)
- Pandas: `import pandas as pd`
- NumPy: `import numpy as np`
- Matplotlib: `import matplotlib.pyplot as plt`

---

## 1. Pandas 핵심: Series / DataFrame

### 1) Series
- 1차원 값(values) + 인덱스(index) 구조
- `sr.values`, `sr.index`로 값/인덱스 확인

### 2) DataFrame
- 2차원(행 index, 열 columns, 값 values) 자료구조
- 생성: `pd.DataFrame(values, index=..., columns=...)`
- 구성요소 확인: `df.index`, `df.columns`, `df.values`
- 리스트/딕셔너리로도 생성 가능

### 3) DataFrame 조회(가장 자주 쓰는 3개)
- `df.head(n)` 앞 n개
- `df.tail(n)` 뒤 n개
- `df['열이름']` 특정 열 보기

### 4) 외부 데이터 읽기
- `pd.read_csv('file.csv')`로 CSV 읽기
- 읽으면 인덱스가 자동 부여될 수 있고 `df.index`로 확인 가능

---

## 2. NumPy 핵심: ndarray / shape / 생성 / 인덱싱 / 연산

### 1) ndarray 개념
- NumPy 핵심 구조: 다차원 배열 `ndarray`
- 딥러닝에서 shape/ndim 감각이 매우 중요(축 개수/크기)
  - `arr.ndim`, `arr.shape`

### 2) 배열 생성
- `np.array([...])` 1D, `np.array([[...],[...]])` 2D
- 초기화 유틸
  - `np.zeros((r,c))`, `np.ones((r,c))`, `np.full((r,c), v)`
  - `np.eye(n)` 항등행렬
  - `np.random.random((r,c))` 랜덤 배열
- 범위 생성: `np.arange(n)` / `np.arange(i, j, k)`

### 3) reshape
- `reshape`는 “데이터는 그대로 + 모양만 변경”

### 4) 슬라이싱/정수 인덱싱
- 슬라이싱: `mat[0, :]`(행), `mat[:, 1]`(열)
- 정수 인덱싱: 연속적이지 않은 원소 선택에 필요

### 5) 연산: element-wise vs dot(행렬곱)
- `+ - * /`는 기본적으로 요소별(element-wise) 연산
- 벡터/행렬곱은 `np.dot(A, B)` 사용

---

## 3. Matplotlib 핵심: 최소로 그릴 줄 알기
- 목적: 데이터 이해/결과 시각화
- 라인 플롯 기본
  - `plt.title(...)`, `plt.plot(x, y)`, `plt.show()`
- 축 라벨
  - `plt.xlabel(...)`, `plt.ylabel(...)`



Progress: documents/references/PROGRESS.md
