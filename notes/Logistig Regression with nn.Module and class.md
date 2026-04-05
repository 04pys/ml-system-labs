# nn.Module과 클래스로 구현하는 로지스틱 회귀 복습 정리

## 1. 로지스틱 회귀의 핵심 가설식

선형 회귀의 가설식은 다음과 같았다.

H(x) = Wx + b

반면 로지스틱 회귀의 가설식은 다음과 같다.

H(x) = sigmoid(Wx + b)

즉, **선형 변환 결과에 sigmoid를 한 번 더 통과시킨 것**이 로지스틱 회귀의 핵심이다.

---

## 2. PyTorch에서 이를 어떻게 구현하는가

PyTorch에서는 다음 두 모듈을 조합하면 된다.

- `nn.Linear(입력차원, 출력차원)`  
  -> `Wx + b`를 계산하는 선형 변환
- `nn.Sigmoid()`  
  -> 선형 변환 결과를 0과 1 사이 값으로 바꾸는 시그모이드 함수

즉,

- `nn.Linear(2, 1)` : 입력 2개를 받아 출력 1개 생성
- `nn.Sigmoid()` : 그 출력값을 확률처럼 해석 가능한 값으로 변환

이 둘을 이어 붙이면 로지스틱 회귀 모델이 된다.

---

## 3. nn.Sequential()의 의미

`nn.Sequential()`은 **여러 `nn.Module`을 순서대로 연결해 주는 컨테이너**이다.

예를 들어

```python
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
```

라고 쓰면 의미는 다음과 같다.

x -> Linear(2, 1) -> Sigmoid -> output

즉,

1. 입력 `x`가 먼저 `nn.Linear(2, 1)`을 통과하고
2. 그 결과가 다시 `nn.Sigmoid()`를 통과해서
3. 최종 출력이 만들어진다

이 흐름이다.

### 복습 포인트
`nn.Sequential()`은 "모듈을 그냥 모아두는 리스트"가 아니라,  
**앞 모듈의 출력을 뒤 모듈의 입력으로 자동 연결해 주는 하나의 `nn.Module`**이다.

그래서

```python
model(x_train)
```

처럼 호출하면 내부적으로 순서대로 계산이 진행된다.

### 내가 헷갈리지 않게 기억할 말
`nn.Sequential()` = "nn.Module들을 차례대로 쌓아서 한 덩어리 모델처럼 만드는 것"

---

## 4. 예제 데이터의 의미

예제에서는 다음과 같이 학습 데이터를 만든다.

```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
```

- `x_train` : 입력 데이터
- `y_train` : 각 입력에 대한 정답 레이블
- 정답은 `0` 또는 `1`이므로 이진 분류 문제임

---

## 5. 모델 정의

```python
model = nn.Sequential(
   nn.Linear(2, 1),
   nn.Sigmoid()
)
```

이 모델은 내부적으로 다음 수식을 계산한다고 보면 된다.

z = w1*x1 + w2*x2 + b  
H(x) = sigmoid(z)

즉 최종적으로는

H(x) = sigmoid(w1*x1 + w2*x2 + b)

를 계산하는 구조이다.

---

## 6. model(x_train)이 왜 가능한가

`model`은 `nn.Sequential(...)`로 만든 **하나의 `nn.Module` 객체**이다.

따라서

```python
model(x_train)
```

이라고 하면, 입력 `x_train`이 모델의 `forward` 계산을 따라 통과한다.

여기서는

1. `x_train`이 `nn.Linear(2, 1)`에 들어가고
2. 그 결과가 `nn.Sigmoid()`로 들어간다

그래서 출력은 각 샘플마다 **0과 1 사이의 값**이 된다.

이 값은 "1일 확률"처럼 해석할 수 있다.

---

## 7. 처음 예측값이 의미 없는 이유

처음에는 `W`와 `b`가 랜덤하게 초기화되어 있다.

따라서 학습 전 `model(x_train)`의 출력은 단지 랜덤 초기값에 따른 결과일 뿐,  
아직 정답 패턴을 학습한 상태가 아니므로 큰 의미가 없다.

즉, 처음 출력은 **"현재 모델이 아직 아무것도 배우지 않은 상태에서 내는 확률값"**이다.

---

## 8. 학습 과정의 핵심 흐름

학습 코드는 다음 흐름으로 진행된다.

```python
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

이 흐름을 말로 풀면 다음과 같다.

### (1) hypothesis 계산
현재 모델이 입력 `x_train`에 대해 예측한 값  
즉 `H(x)`를 구한다.

### (2) cost 계산
예측값 `hypothesis`와 실제 정답 `y_train`의 차이를  
`binary_cross_entropy`로 측정한다.

### (3) zero_grad()
이전 step에서 누적된 gradient를 초기화한다.

### (4) backward()
cost를 기준으로 각 파라미터에 대한 기울기(gradient)를 계산한다.

### (5) step()
optimizer가 gradient를 보고 `W`, `b`를 업데이트한다.

---

## 9. binary_cross_entropy의 의미

`F.binary_cross_entropy(hypothesis, y_train)`은  
**이진 분류용 손실 함수**이다.

로지스틱 회귀에서 배우는 다음 식과 같은 의미로 보면 된다.

- [ y log(H(x)) + (1-y) log(1-H(x)) ]

정확히는 보통 배치 전체에 대해 평균을 내어 사용한다.

### 중요한 점
이 함수에 넣는 `hypothesis`는 **이미 sigmoid를 지난 확률값**이어야 한다.

즉 이 예제에서는

- `nn.Linear`가 먼저 선형 변환을 하고
- `nn.Sigmoid()`가 확률값으로 바꾼 뒤
- 그 결과를 `binary_cross_entropy`에 넣는 구조이다.

---

## 10. 정확도 계산 방식

예제에서는 예측값이 0.5 이상이면 1, 미만이면 0으로 간주한다.

```python
prediction = hypothesis >= torch.FloatTensor([0.5])
```

즉,

- 예측 확률 >= 0.5 이면 True -> 1로 간주
- 예측 확률 < 0.5 이면 False -> 0으로 간주

그 다음 실제 정답 `y_train`과 비교해서 맞은 개수를 세고,  
전체 개수로 나누어 정확도를 계산한다.

---

## 11. 꼭 기억할 핵심 차이

### 선형 회귀
H(x) = Wx + b

### 로지스틱 회귀
H(x) = sigmoid(Wx + b)

즉, 로지스틱 회귀는  
**선형 회귀의 출력에 sigmoid를 덧붙인 구조**라고 이해하면 된다.

---

## 12. 이번 페이지에서 반드시 기억할 트리거

### 트리거 1
`nn.Linear + nn.Sigmoid = 로지스틱 회귀`

### 트리거 2
`nn.Sequential()`은 여러 `nn.Module`을 **순서대로 연결**한다.

### 트리거 3
`model(x_train)`은  
입력이 모델을 통과하며 차례대로 계산된 결과를 반환한다.

### 트리거 4
`binary_cross_entropy`는  
로지스틱 회귀에서의 이진 분류 cost 함수이다.

### 트리거 5
학습 루프의 기본 흐름은 항상

예측 -> 손실 계산 -> gradient 초기화 -> 역전파 -> 파라미터 업데이트

이다.

---

## 13. 한 줄 요약

이 페이지는  
**PyTorch에서 `nn.Linear`와 `nn.Sigmoid`를 `nn.Sequential()`로 연결해 로지스틱 회귀를 구현하고, `binary_cross_entropy`로 학습시키는 기본 구조**를 익히는 페이지이다.
