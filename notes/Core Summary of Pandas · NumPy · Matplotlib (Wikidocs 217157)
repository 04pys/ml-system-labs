# Pandas / NumPy / Matplotlib 핵심 요약 (Wikidocs 217157)

> 목표: Colab에서 데이터 다루기/계산/시각화의 최소 핵심을 빠르게 복습하기

## 0. 임포트 규칙(관례)
- Pandas: `import pandas as pd` :contentReference[oaicite:0]{index=0}
- NumPy: `import numpy as np` :contentReference[oaicite:1]{index=1}
- Matplotlib: `import matplotlib.pyplot as plt` :contentReference[oaicite:2]{index=2}

---

## 1. Pandas 핵심: Series / DataFrame

### 1) Series
- 1차원 값(values) + 인덱스(index) 구조 :contentReference[oaicite:3]{index=3}
- `sr.values`, `sr.index`로 값/인덱스 확인 :contentReference[oaicite:4]{index=4}

### 2) DataFrame
- 2차원(행 index, 열 columns, 값 values) 자료구조 :contentReference[oaicite:5]{index=5}
- 생성: `pd.DataFrame(values, index=..., columns=...)` :contentReference[oaicite:6]{index=6}
- 구성요소 확인: `df.index`, `df.columns`, `df.values` :contentReference[oaicite:7]{index=7}
- 리스트/딕셔너리로도 생성 가능 :contentReference[oaicite:8]{index=8}

### 3) DataFrame 조회(가장 자주 쓰는 3개)
- `df.head(n)` 앞 n개 :contentReference[oaicite:9]{index=9}
- `df.tail(n)` 뒤 n개 :contentReference[oaicite:10]{index=10}
- `df['열이름']` 특정 열 보기 :contentReference[oaicite:11]{index=11}

### 4) 외부 데이터 읽기
- `pd.read_csv('file.csv')`로 CSV 읽기 :contentReference[oaicite:12]{index=12}
- 읽으면 인덱스가 자동 부여될 수 있고 `df.index`로 확인 가능 :contentReference[oaicite:13]{index=13}

---

## 2. NumPy 핵심: ndarray / shape / 생성 / 인덱싱 / 연산

### 1) ndarray 개념
- NumPy 핵심 구조: 다차원 배열 `ndarray` :contentReference[oaicite:14]{index=14}
- 딥러닝에서 shape/ndim 감각이 매우 중요(축 개수/크기) :contentReference[oaicite:15]{index=15}
  - `arr.ndim`, `arr.shape` :contentReference[oaicite:16]{index=16}

### 2) 배열 생성
- `np.array([...])` 1D, `np.array([[...],[...]])` 2D :contentReference[oaicite:17]{index=17}
- 초기화 유틸
  - `np.zeros((r,c))`, `np.ones((r,c))`, `np.full((r,c), v)` :contentReference[oaicite:18]{index=18}
  - `np.eye(n)` 항등행렬 :contentReference[oaicite:19]{index=19}
  - `np.random.random((r,c))` 랜덤 배열 :contentReference[oaicite:20]{index=20}
- 범위 생성: `np.arange(n)` / `np.arange(i, j, k)` :contentReference[oaicite:21]{index=21}

### 3) reshape
- `reshape`는 “데이터는 그대로 + 모양만 변경” :contentReference[oaicite:22]{index=22}

### 4) 슬라이싱/정수 인덱싱
- 슬라이싱: `mat[0, :]`(행), `mat[:, 1]`(열) :contentReference[oaicite:23]{index=23}
- 정수 인덱싱: 연속적이지 않은 원소 선택에 필요 :contentReference[oaicite:24]{index=24}

### 5) 연산: element-wise vs dot(행렬곱)
- `+ - * /`는 기본적으로 요소별(element-wise) 연산 :contentReference[oaicite:25]{index=25}
- 벡터/행렬곱은 `np.dot(A, B)` 사용 :contentReference[oaicite:26]{index=26}

---

## 3. Matplotlib 핵심: 최소로 그릴 줄 알기
- 목적: 데이터 이해/결과 시각화 :contentReference[oaicite:27]{index=27}
- 라인 플롯 기본
  - `plt.title(...)`, `plt.plot(x, y)`, `plt.show()` :contentReference[oaicite:28]{index=28}
- 축 라벨
  - `plt.xlabel(...)`, `plt.ylabel(...)` :contentReference[oaicite:29]{index=29}

---

## 4. 오늘의 체크(스스로 점검)
- [ ] DataFrame을 만들고 `head/tail`, 특정 열 선택을 바로 할 수 있나 :contentReference[oaicite:30]{index=30}
- [ ] ndarray의 `shape/ndim`을 보고 연산 가능 여부를 판단할 수 있나 :contentReference[oaicite:31]{index=31}
- [ ] `*`(요소별) vs `dot`(행렬곱)을 실수 없이 구분하나 :contentReference[oaicite:32]{index=32}
- [ ] plot의 title/xlabel/ylabel/show 기본 흐름을 외웠나 :contentReference[oaicite:33]{index=33}
