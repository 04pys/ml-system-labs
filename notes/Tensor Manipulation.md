# Tensor Manipulation

## 1. tensor 기본 개념

- **scalar**: 숫자 1개
- **vector**: 1차원 tensor
- **matrix**: 2차원 tensor
- **tensor**: 다차원 배열 전체를 넓게 부르는 말

즉,

- vector = 1D tensor
- matrix = 2D tensor
- 3D 이상은 보통 tensor라고 많이 부름

딥러닝에서는 1D, 2D도 전부 그냥 tensor라고 생각해도 됨.

---

## 2. NLP에서의 tensor

여기가 특히 중요함.

NLP에서는 보통 다음 shape를 자주 봄:

`(batch size, length, dim)`

의미:

- `batch size`: 한 번에 넣는 문장 개수
- `length`: 문장 길이(토큰 개수)
- `dim`: 각 토큰을 표현하는 벡터 차원

예:
- 문장 32개
- 각 문장 길이 10
- 각 단어 벡터 차원 300

→ shape: `(32, 10, 300)`

즉 NLP tensor는 보통  
**문장 묶음 × 문장 길이 × 각 단어의 벡터 크기**  
로 이해하면 됨. :contentReference[oaicite:1]{index=1}

---

## 3. shape를 읽는 습관

항상 먼저 볼 것:

- `t.ndim` : 몇 차원인지
- `t.shape` : 각 차원의 크기

tensor manipulation은 결국  
**shape를 읽고, 원하는 shape로 바꾸는 작업**임.

---

## 4. dim의 의미

내가 가장 헷갈리지 말아야 할 부분.

`dim=n`은  
**n번째 차원을 기준으로 연산한다**는 뜻이다.

그리고 `mean`, `sum`, `max` 같은 연산에서는  
보통 **그 차원이 사라진다(제거된다)**고 이해하면 된다.

예를 들어 shape가 `(2, 2)`인 tensor가 있을 때:

abc
[[1, 2],
 [3, 4]]
abc

### `dim=0`
- **세로 방향으로 내려가면서** 묶는 느낌
- 같은 열끼리 연산
- 결과 shape는 `(2,)`

예:
- 첫 번째 열: 1, 3
- 두 번째 열: 2, 4

### `dim=1`
- **가로 방향으로** 묶는 느낌
- 같은 행끼리 연산
- 결과 shape는 `(2,)`

예:
- 첫 번째 행: 1, 2
- 두 번째 행: 3, 4

즉,

- `dim=0` → 행 차원을 없애면서 열별 계산
- `dim=1` → 열 차원을 없애면서 행별 계산

---

## 5. `mean`, `sum`, `max`에서 dim 이해

### `mean()`
- 전체 평균을 구할 수도 있고
- `dim`을 주면 그 차원을 따라 평균을 구함
- 그 결과 **해당 차원은 제거됨**

예:
abc
t.mean()
abc
→ 전체 평균

abc
t.mean(dim=0)
abc
→ 열별 평균, `dim=0` 제거

abc
t.mean(dim=1)
abc
→ 행별 평균, `dim=1` 제거

### `sum()`
- `mean()`과 똑같은 방식으로 이해하면 됨
- 지정한 차원을 따라 더하고, 그 차원은 사라짐

### `max()`
- 전체 최댓값도 가능
- `dim`을 주면 그 차원 기준으로 최대값을 찾음
- `max(dim=...)`는 **최댓값과 argmax 인덱스 둘 다 반환**함

예:
abc
t.max(dim=0)
abc

→ 열별 최대값, 열별 argmax 반환

abc
t.max(dim=1)
abc

→ 행별 최대값, 행별 argmax 반환

또,
- `[0]` : max 값
- `[1]` : argmax 값

으로 꺼낼 수 있었음.

`dim=-1`은 **마지막 차원**을 의미함.  
그래서 2D tensor에서는 `dim=1`과 같은 의미가 될 수 있음. :contentReference[oaicite:2]{index=2}

---

## 6. tensor를 다루는 핵심 방법들

### 1) 생성
- `torch.tensor()`
- `torch.FloatTensor()`
- numpy 배열로부터 생성

### 2) 확인
- `.ndim`
- `.shape`
- `.size()`

### 3) 인덱싱 / 슬라이싱
- 특정 원소 접근
- 특정 행, 열, 구간 자르기

### 4) 형태 변형
- `view()` 사용
- `-1`은 PyTorch가 자동으로 계산하게 맡기는 것

예:
abc
t.view(-1, 3)
abc

→ 두 번째 차원을 3으로 맞추고, 첫 번째 차원은 자동 계산

중요:
- `view()`는 **원소 수는 유지한 채 shape만 바꾸는 것**
- 총 원소 개수는 반드시 같아야 함 :contentReference[oaicite:3]{index=3}

---

## 7. `squeeze()` / `unsqueeze()`

### `squeeze()`
- 크기가 1인 차원을 제거

예:
- shape `(3, 1)` → `(3,)`

즉, 쓸데없이 붙어 있는 1차원 축을 없애는 느낌. :contentReference[oaicite:4]{index=4}

### `unsqueeze(dim)`
- 지정한 위치에 크기 1인 차원을 추가

예:
- `(3,)`에서 `unsqueeze(0)` → `(1, 3)`
- `(3,)`에서 `unsqueeze(1)` → `(3, 1)`
- `(3,)`에서 `unsqueeze(-1)` → 마지막에 추가 → `(3, 1)`

핵심:
- `squeeze`는 차원 제거
- `unsqueeze`는 차원 추가
- 둘 다 **원소 수는 그대로**이고 shape만 바뀜 :contentReference[oaicite:5]{index=5}

---

## 8. `dim`을 "제거"와 "확장" 관점에서 기억하기

이 부분을 제일 실전적으로 기억하면 좋음.

### 감소 연산 (`mean`, `sum`, `max`)
- `dim`은 **어느 축을 따라 줄일지**
- 그 축은 결과에서 **사라진다**

예:
- `(2, 3)`에서 `mean(dim=0)` → `(3,)`
- `(2, 3)`에서 `mean(dim=1)` → `(2,)`

### 차원 추가 (`unsqueeze(dim)`)
- `dim`은 **어디에 새 축을 끼워 넣을지**
- 그 위치에 크기 1짜리 차원이 생김

예:
- `(3,)`에서 `unsqueeze(0)` → `(1, 3)`
- `(3,)`에서 `unsqueeze(1)` → `(3, 1)`

즉,

- `mean/sum/max`의 `dim` = **그 차원을 줄인다**
- `unsqueeze`의 `dim` = **그 위치에 차원을 만든다**

이렇게 기억하면 헷갈림이 많이 줄어듦.

---

## 9. 행렬 곱과 원소별 곱

이 부분도 실습에서 중요했음.

- `.matmul()` : 행렬 곱
- `.mul()` : 원소별 곱

즉 둘은 전혀 다름.

- 행렬 곱: 선형대수의 행렬 곱셈
- 원소별 곱: 같은 위치끼리 곱하기 :contentReference[oaicite:6]{index=6}

---

## 10. 이번 실습에서 기억할 것

- tensor는 다차원 배열
- NLP에서는 주로 `(batch size, length, dim)` 형태를 많이 봄
- tensor manipulation의 핵심은 **shape 이해**
- `dim`은 축(axis) 개념
- `mean`, `sum`, `max`에서는 그 축이 **줄어들며 제거됨**
- `unsqueeze(dim)`에서는 그 위치에 **새 차원이 추가됨**
- `view()`, `squeeze()`, `unsqueeze()`는 **원소 수는 그대로 두고 shape만 바꾸는 연산**

---

## 한 줄 정리

Tensor manipulation은  
**shape를 읽고, dim이 어떤 축인지 이해한 뒤, 그 축을 줄이거나 추가하거나 바꾸는 작업**이라고 보면 됨.
