# 선형 회귀와 자동 미분 복습 정리

## 1. 선형 회귀의 핵심

선형 회귀는 **훈련 데이터를 가장 잘 설명하는 직선**을 찾는 문제다.  
기본 가설은 다음과 같다.

H(x) = Wx + b

- `W`: 가중치(weight), 직선의 기울기
- `b`: 편향(bias), y절편

PyTorch 예제에서는 보통 공부 시간 `x_train`과 점수 `y_train`처럼, 입력과 출력을 각각 텐서로 두고 선형 관계를 학습한다.

---

## 2. 손실 함수: MSE(Mean Squared Error)

평균 제곱 오차(MSE)는 **실제값과 예측값의 차이를 제곱한 뒤 평균낸 값**이다.

MSE = (1/n) * Σ[ y^(i) - H(x^(i)) ]^2    (i = 1 ~ n)

즉,

- 오차를 그냥 더하면 양수/음수가 상쇄될 수 있으므로
- 제곱해서 모두 양수로 만든 뒤
- 평균을 구해 손실로 사용한다

선형 회귀에서는 이 MSE를 최소로 만드는 `W`와 `b`를 찾는 것이 목표다.

정리하면:

cost(W, b) = (1/n) * Σ[ y^(i) - H(x^(i)) ]^2    (i = 1 ~ n)

---

## 3. 경사 하강법(Gradient Descent)

손실 함수가 작아지는 방향으로 파라미터를 조금씩 조정하는 방법이다.

기본 아이디어:

W := W - α * (∂cost / ∂W)

여기서

- `α`: learning rate
- `(∂cost / ∂W)`: `W`에 대한 gradient

의미는 다음과 같다.

- gradient가 양수이면 `W`를 줄여야 cost가 감소
- gradient가 음수이면 `W`를 늘려야 cost가 감소

즉, **현재 위치에서 손실이 가장 가파르게 증가하는 방향의 반대로 이동**하는 방식이다.

learning rate에 대한 감각도 중요하다.

- 너무 크면 최솟값 근처를 지나쳐 발산할 수 있음
- 너무 작으면 학습이 매우 느려짐

---

## 4. 왜 `b`도 미분 가능한가

많이 헷갈리는 부분이다.

가설이

H(x) = Wx + b

이므로, `b`도 예측값 계산에 직접 들어간다.  
즉 `b`를 바꾸면 `hypothesis`가 바뀌고, 그러면 오차와 cost도 바뀐다.  
그래서 `cost`를 `b`에 대해서도 미분할 수 있다.

직관적으로 보면:

- `W`는 직선의 **기울기**를 조정
- `b`는 직선을 **위아래로 평행이동**

데이터에 잘 맞추려면 기울기뿐 아니라 위치도 맞아야 하므로 `b`도 학습 대상이다.

중요한 점은:

**`backward()`는 `W`만 미분하는 것이 아니라, cost 계산에 참여한 모든 학습 가능한 파라미터에 대해 gradient를 계산한다는 점**이다.

즉 `W.grad`뿐 아니라 `b.grad`도 계산된다.

---

## 5. `requires_grad=True`의 의미

PyTorch에서 어떤 텐서를 학습 대상 파라미터로 쓰려면 보통 `requires_grad=True`로 둔다.

```python
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

이 뜻은:

- 이 텐서에 대해 연산 그래프를 추적하고
- 최종적으로 `backward()`를 호출했을 때
- 이 텐서에 대한 미분값을 계산하겠다는 의미다

즉, `requires_grad=True`가 붙은 텐서는 “학습 가능한 변수”라고 생각하면 된다.

---

## 6. 자동 미분(Autograd)의 핵심

예를 들어:

```python
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 2*y + 5
z.backward()
```

이면 PyTorch는 연산 그래프를 따라가며 체인 룰로 미분한다.

수식으로는:

y = w^2  
z = 2y + 5 = 2w^2 + 5

따라서

dz/dw = 4w

`w = 2`이면

dz/dw = 8

그래서 `w.grad`는 8이 된다.

여기서 중요한 점:

- `backward()`는 최종 결과(`z` 또는 `cost`)를 기준으로
- 그 값에 영향을 준 모든 `requires_grad=True` 텐서에 대해
- gradient를 계산한다

---

## 7. `y.grad`가 `None`인 이유

이 부분도 자주 헷갈린다.

예를 들어:

```python
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 2*y + 5
z.backward()
```

이때

- `w.grad`는 값이 나옴
- `y.grad`는 `None`일 수 있음

이유는 PyTorch가 기본적으로 **leaf tensor의 gradient만 `.grad`에 저장**하기 때문이다.

- `w`: 사용자가 직접 만든 leaf tensor
- `y`: `w**2`라는 연산 결과로 나온 intermediate tensor

즉, `y`에 대한 gradient 계산은 내부적으로 사용되지만, 기본적으로 `.grad`에 저장하지 않는다.

`y.grad`까지 보고 싶으면:

```python
y.retain_grad()
```

를 `backward()` 전에 호출해야 한다.

---

## 8. 선형 회귀 코드 흐름의 핵심

선형 회귀 학습 코드는 보통 이런 구조를 가진다.

```python
optimizer.zero_grad()
cost.backward()
optimizer.step()
```

이 3줄이 학습 루프의 핵심이다.

---

## 9. `optimizer.zero_grad()`

역할: **이전 step에서 저장된 gradient를 0으로 초기화**

PyTorch는 `backward()`를 호출할 때 gradient를 덮어쓰지 않고 **누적(accumulate)** 한다.  
그래서 이전 반복에서 계산된 gradient가 남아 있으면 다음 반복 결과와 섞인다.

즉, 학습 루프에서 gradient를 새로 계산하기 전에 반드시 초기화해야 한다.

정리:

- `backward()` -> gradient를 계산해서 `.grad`에 더함
- `zero_grad()` -> `.grad`를 0으로 비움

그래서 보통 순서는:

1. grad 초기화
2. backward로 새 grad 계산
3. step으로 업데이트

---

## 10. `cost.backward()`

역할: **현재 cost를 기준으로 연산 그래프를 역추적하며 gradient 계산**

선형 회귀에서는 보통

```python
hypothesis = x_train * W + b
cost = torch.mean((y_train - hypothesis) ** 2)
```

이고, 여기서 `cost.backward()`를 하면 PyTorch가 자동으로 다음을 계산한다.

- `cost`를 `W`로 미분한 값 -> `W.grad`
- `cost`를 `b`로 미분한 값 -> `b.grad`

중요한 점:

`backward()`는 단순히 `W`만 미분하는 것이 아니다.  
**cost 계산에 참여한 모든 학습 가능한 파라미터에 대해 gradient를 계산**한다.

즉 `W`뿐 아니라 `b`도 학습 대상이므로 같이 미분된다.

---

## 11. `optimizer.step()`

역할: **계산된 gradient를 사용해 실제 파라미터를 업데이트**

예를 들어 SGD라면 내부적으로 대략 다음과 비슷하게 동작한다.

W <- W - η * (∂cost / ∂W)

b <- b - η * (∂cost / ∂b)

즉 optimizer가 관리하는 파라미터 목록에 따라 `W`, `b` 둘 다 갱신된다.

예:

```python
optimizer = torch.optim.SGD([W, b], lr=0.01)
```

이면 `step()`은 `W.grad`와 `b.grad`를 읽어서 둘 다 업데이트한다.

핵심 직관:

- `W`는 직선의 기울기를 조정
- `b`는 직선을 위아래로 이동
- 둘 다 cost를 줄이는 방향으로 조금씩 바뀜

---

## 12. 세 줄을 한 흐름으로 이해하기

```python
optimizer.zero_grad()
cost.backward()
optimizer.step()
```

이 세 줄을 말로 풀면:

1. **기존 gradient를 비운다**
2. **현재 손실 기준으로 W, b의 gradient를 계산한다**
3. **그 gradient를 이용해 W, b를 cost가 줄어드는 방향으로 업데이트한다**

이걸 반복하면 `W`, `b`가 점점 더 좋은 값으로 이동한다.

---

## 13. `manual_seed` 짧은 정리

`torch.manual_seed()`는 **난수 발생 순서를 고정**한다.

즉 seed를 고정하면 프로그램을 다시 실행해도 같은 순서의 난수가 생성된다.  
그래서 실험 재현성이 좋아진다.

예:

```python
torch.manual_seed(1)
```

의미:

- 매번 같은 초기값, 같은 난수 흐름을 사용하게 되어
- 실행 결과를 재현하기 쉬워진다

---

## 14. 빠르게 떠올려야 할 복습 트리거

### 선형 회귀
- 목표: 데이터를 가장 잘 설명하는 직선 찾기
- 식: `H(x) = Wx + b`

### 손실 함수
- MSE 사용
- 실제값과 예측값 차이의 제곱 평균
- `W`, `b`를 잘 찾을수록 MSE가 작아짐

### 자동 미분
- `requires_grad=True`면 gradient 추적
- `backward()`가 연산 그래프를 따라 gradient 계산
- `W.grad`, `b.grad`에 저장

### 학습 루프 3단계
- `optimizer.zero_grad()` -> grad 초기화
- `cost.backward()` -> grad 계산
- `optimizer.step()` -> 파라미터 업데이트

### `b`도 왜 학습?
- `b`도 hypothesis에 직접 들어감
- 직선의 위치를 조정하므로 cost에 영향
- 따라서 `cost`를 `b`로도 미분 가능

### `y.grad`가 None인 이유
- intermediate tensor는 기본적으로 `.grad` 저장 안 함
- leaf tensor만 `.grad` 기본 저장
- 필요하면 `retain_grad()` 사용

### `manual_seed`
- 난수 발생 순서 고정
- 재현성 확보

---

## 15. 한 번에 기억할 요약

선형 회귀는 `H(x) = Wx + b` 형태의 직선을 데이터에 맞추는 문제이며, 손실 함수로는 보통 MSE를 쓴다. 학습에서는 `requires_grad=True`로 설정된 `W`, `b`에 대해 `cost.backward()`가 자동으로 gradient를 계산하고, `optimizer.step()`이 그 gradient를 사용해 파라미터를 업데이트한다. 이때 gradient는 누적되므로 매 step마다 `optimizer.zero_grad()`로 초기화해야 한다. `W`는 기울기, `b`는 직선의 위치를 조정하므로 둘 다 학습 대상이다. `manual_seed`는 난수 순서를 고정해 실험 재현성을 높인다.
