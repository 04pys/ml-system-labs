# nn.Module과 클래스로 구현하기 복습 정리

## 1. 이 실습의 핵심이 무엇이었나

이 실습의 핵심은 **선형 회귀 모델을 더 “PyTorch답게” 구현하는 방식**을 익히는 것이었다.

이전까지는 가설 함수와 비용 함수를 직접 정의했다면, 여기서는 PyTorch가 이미 제공하는 구성요소를 사용했다.

- 선형 회귀 모델: `nn.Linear(...)`
- 평균 제곱 오차: `F.mse_loss(...)`
- 모델 구조화: `class ... (nn.Module)`

즉, 이번 파트는 단순히 “선형 회귀를 다시 해봤다”가 아니라,

- `nn.Linear`가 파라미터를 어떻게 들고 있는지 이해하고
- `model.parameters()`가 왜 중요한지 확인하고
- `forward`:contentReference[oaicite:0]{index=0}다시 연결하고
- 나아가 **PyTorch 모델을 클래스로 만드는 기본 형식**을 익히는 실습이었다. :contentReference[oaicite:1]{index=1}

---

## 2. 단순 선형 회귀에서 기억할 것

단순 선형 회귀는 입력 1개, 출력 1개인 모델이다.

수식으로 쓰면 다음과 같다.

- `H(x) = Wx + b`

PyTorch에서는 이것을 직접 `W`, `b`를 선언해서 만들 수도 있지만, 이번 실습에서는 다음처럼 더 간단히 만들었다.

abcpython
model = nn.Linear(1, 1:contentReference[oaicite:2]{index=2}미는 다음과 같다.

- 입력 차원: 1
- 출력 차원: 1

즉, `x` 하나를 받아서 `y` 하나를 예측하는 구조라는 뜻이다. :contentReference[oaicite:3]{index=3}

### 복습 트리거

- `nn.Linear(1, 1)`은 결국 `Wx + b`를 내부적으로 수행하는 레이어임
- 이 안에는 이미 `weight`와 `bias`가 들어 있음
- 그래서 내가 직접 `W`, `b`를 만들지 않아도 됨

---

## 3. model.parameters()가 왜 중요한가

`nn.Linear(1, 1)`로 모델을 만들면 내부에 학습 대상 파라미터가 자동으로 생성된다.

- 첫 번째: `weight`
- 두 번째: `bias`

실습에서는 `list(model.parame:contentReference[oaicite:4]{index=4}은 랜덤이며, 둘 다 `requires_grad=True` 상태였다. 이는 두 값이 경사 하강법으로 업데이트될 대상이라는 뜻이다. :contentReference[oaicite:5]{index=5}

### 복습 트리거

- optimizer는 무엇을 업데이트하는가?  
  → `model.parameters()`
- 왜 직접 `W`, `b`를 안 넘겨도 되는가?  
  → `nn.Linear` 내부에 이미 등록되어 있기 때문
- `requires_grad=True`는 무엇을 뜻하는가?  
  → 미분 추적 대상이라는 뜻

---

## 4. 학습 루프의 흐름 다시 연결하기

실습의 학습 코드는 다음 순서로 흘러갔다.

abcpython
prediction = model(x_train)
cost = F.mse_loss(prediction, y_train)

optimizer.zero_grad()
cost.backward()
optimizer.step()
abc

이 5줄은 정말 중요하다.

:contentReference[oaicite:6]{index=6}model(x_train)`

입력 `x_train`을 모델에 넣어서 예측값을 얻는다.  
이것이 **forward 연산**이다. :contentReference[oaicite:7]{index=7}

### (2) `cost = F.:contentReference[oaicite:8]{index=8}제값의 차이를 평균 제곱 오차로 계산한다.  
즉, “지금 모델이 얼마나 틀렸는가”를 수치로 만든다. :contentReference[oaicite:9]{index=9}

### (3) `optimizer.zero_grad()`

기존 gradient를 초기화한다.

PyTorch는 gradient를 자동으로 **누적(accumulate)** 하기 때문에, 이전 step의 gradient를 지우지 않으면 계속 더해진다. 그래서 매 step마다 먼저 0으로 비워줘야 한다.

###:contentReference[oaicite:10]{index=10}

비용 함수 `cost`를 기준으로 각 파라미터에 대한 기울기(미분값)를 계산한다.  
이것이 **backward 연산**이다. :contentReference[oaicite:11]{index=11}

### (5) `optimizer.step()`

계산된 gradient를 이용해서 파라미터를 실제로 업데이트한다.  
즉, `W`, `b`가 조금 더 좋은 방향으로 움직인다.

### 복습 트리거

- `forward`는 예측값을 만드는 단계
- `backward`는 기울기를 계산하는 단계
- `step`은 실제 파라미터를 바꾸는 단계
- `zero_grad`를 빼면 gradient가 누적됨

---

## 5. 왜 예측값이 점점 정답에 가까워졌나

실습 데이터는 사실상 `y = 2x` 형태였다.

- `x = [[1], [2], [3]]`
- `y = [[2], [4], [6]]`

즉, 정답은 대략

- `W = 2`
- `b = 0`

에 가까워야 한다.

학습 전에는 `W`, `b`가 랜덤이지만, 반복 학습을 거치면서 cos:contentReference[oaicite:12]{index=12}으로

- `W`는 2에 가깝게
- `b`는 0에 가깝게

수렴했다. 또한 `x=4`를 넣었을 때 예측값이 8에 매우 가까워졌다. :contentReference[oaicite:13]{index=13}

### 복습 트리거

- 선형 회귀 학습의 본질  
  → `W`, `b`를 정답에 가까운 값으로 찾아가는 과정
- cost가 줄어든다는 뜻  
  → 예측이 실제값에 가까워진다는 뜻

---

## 6. 다중 선형 회귀에서 바뀌는 점

다중 선형 회귀는 입력이 여러 개다.

예를 들어 실습에서는 입력이 3개였다.

- `x1, x2, x3 -> y`

수식은 다음처럼 볼 수 있다.

- `H(x) = w1x1 + w2x2 + w3x3 + b`

PyTorch에서는 이것을 다음처럼 표현했다.

abcpy:contentReference[oaicite:14]{index=14}(3, 1)
abc

즉,

- 입력 차원 3
- 출력 차원 1

이라는 뜻이다.  
여기서는 weight도 3개를 가지게 된다. :contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}또한 학습률을 `0.01`이 아니라 `1e-5` 정도로 낮췄는데, 이는 너무 큰 학습률을 쓰면 기울기가 발산할 수 있기 때문이다. :contentReference[oaicite:17]{index=17}

### 복습 트리거

- `nn.Linear(3, 1)`이면 weight가 3개 필요함
- 입력 차원이 커지면 학습률 설정이 더 민감해질 수 있음
- 구조는 바뀌어도 학습 루프 자체는 거의 동일함

---

## 7. 클래스 기반 모델 구현의 의미

실습의 가장 중요한 부분 중 하나는 다음 코드였다.

abcpython
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
abc

그리고

abcpython
model = LinearRegressionModel()
:contentReference[oaicite:18]{index=18}  
이 형식은 PyTorch에서 매우 표준적인 모델 정의 방식이다. 위키독스도 이 형식을 반드시 숙지할 필요가 있다고 설명한다. :contentReference[oaicite:19]{index=19}

---

## 8. `super().__init__()`는 왜 필요한가

이 부분은 따로 꼭 기억해야 한다.

처음 보면 그냥 관습처럼 보이지만, 실제로는 매우 중요한 코드다.

### 정확한 의미

`super().__init__()`는 **부모 클래스인 `nn.Module`의 생성자**를 호출하는 것이다.

즉, 현재 클래스 `LinearRegressionModel`이 `nn.Module`을 상속받았기 때문에, 부모인 `nn.Module`이 가지고 있는 초기화 작업을 먼저 수행하는 것이다.

여기서 중요한 점은 표기가 `super.__init__()`가 아니라 **`super().__init__()`**라는 점이다.

---

## 9. `nn.Module`의 속성들로 초기화된다는 말의 뜻

`nn.Module`은 단순히 “forward 함수만 정의하게 해주는 클래스”가 아니다.  
PyTorch가 모델을 **정식 신경망 모듈로 관리**할 수 있게 하는 기반 클래스다.

`super().__init__()`를 호출하면 내부적으로 이런 관리 구조들이 준비된다.

- `_parameters`
  - 이 모듈의 학습 파라미터를 저장하는 공간
- `_modules`
  - 이 모듈 안에 들어있는 하위 모듈을 저장하는 공간
- `_buffers`
  - 파라미터는 아니지만 모델 상태로 관리해야 하는 값들을 저장하는 공간
- 그 밖에 hook 관련 내부 구조들

즉, `super().__init__()`는  
**“이 객체를 PyTorch가 추적 가능한 모델 객체로 만드는 초기 설정”**을 수행한다고 이해하면 된다.

---

## 10. 그 초기화가 실제로 왜 의미가 큰가

다음 코드가 있다고 하자.

abcpython
self.linear = nn.Linear(1, 1)
abc

이 코드가 단순한 파이썬 속성 저장으로 끝나지 않고,  
PyTorch가 **“아, 이건 이 모델의 하위 레이어구나”**라고 인식하게 하려면 `nn.Module`의 초기화가 선행되어 있어야 한다.

즉, `super().__init__()`가 먼저 실행되어야 이후의

- `self.linear = nn.Linear(1, 1)`

가 단순 변수 저장이 아니라 **submodule 등록**으로 처리된다.

그 결과 다음이 가능해진다.

- `model.parameters()` 호출 시 내부 파라미터를 자동 수집
- `optimizer = SGD(model.parameters(), ...)` 가능
- `model.to(device)` 시 내부 파라미터까지 함께 이동
- `model.state_dict()`로 저장 가능
- `model.train()`, `model.eval()`이 하위 모듈까지 적용됨

즉, `super().__init__()`는  
**“이 객체를 그냥 파이썬 객체가 아니라 PyTorch 모델로 정식 등록하는 준비 작업”**이다.

---

## 11. 한 줄씩 다시 읽는 클래스 코드

abcpython
class LinearRegressionModel(nn.Module):
abc

- `nn.Module`을 상속받음
- 즉, 이 클래스는 PyTorch 모델의 규칙을 따름

abcpython
def __init__(self):
    super().__init__()
abc

- 객체 생성 시 부모 클래스 초기화
- `nn.Module`의 내부 관리 구조 세팅

abcpython
self.linear = nn.Linear(1, 1)
abc

- 선형 레이어 하나를 생성
- 입력 1개, 출력 1개
- 이 레이어는 `self.linear`라는 하위 모듈로 등록됨

abcpython
def forward(self, x):
    return self.linear(x)
abc

- 입력 `x`가 들어오면 선형 레이어를 통과시켜 결과 반환
- 즉, 모델의 실제 계산식을 정의하는 부분

### 복습 트리거

- `__init__`는 “구성품 만들기”
- `forward`는 “실제 계산 정의”
- `super().__init__()`는 “PyTorch 모델로서의 기본 공사”
- `self.linear = nn.Linear(...)`는 “구성품 장착”

---

## 12. 내가 헷갈리지 않기 위한 핵심 구분

### `nn.Linear(...)`
이미 구현된 선형 변환 레이어

### `nn.Module`
여러 레이어와 파라미터를 관리하는 모델의 기반 클래스

### `super().__init__()`
부모 클래스 `nn.Module`의 초기화 수행

### `forward(...)`
입력이 들어왔을 때 어떤 계산을 할지 정의

### `model(x)`
겉으로는 이렇게 쓰지만 내부적으로는:contentReference[oaicite:20]{index=20}스도 다중 선형 회귀 부분에서 `model(x_train)`은 `model.forward(x_train)`와 동일하다고 설명한다. :contentReference[oaicite:21]{index=21}

---

## 13. 이번 파트를 한 번에 떠올리는 복습 트리거

아래 질문에 바로 답할 수 있으면 복습이 잘 된 것이다.

1. `nn.Linear(1, 1)`의 의미는 무엇인가?  
   → 입력 1개, 출력 1개인 선형 레이어

2. `model.parameters()`는 왜 필요한가?  
   → optimizer가 업데이트할 파라미터를 넘기기 위해

3. `optimizer.zero_grad()`를 왜 먼저 하는가?  
   → gradient 누적을 막기 위해

4. `cost.backward()`는 무엇을 하는가?  
   → 비용 함수 기준으로 gradient 계산

5. `optimizer.step()`은 무엇을 하는가?  
   → 계산된 gradient로 파라미터 업데이트

6. `class ... (nn.Module)`로 만드는 이유는 무엇인가?  
   → PyTorch 방식으로 모델을 구조화하고 관리하기 위해

7. `super().__init__()`가 왜 필요한가?  
   → `nn.Module`의 내부 관리 구조를 초기화해서 레이어, 파라미터, 저장/이동 기능이 정상 동작하게 하기 위해

8. `__init__`와 `forward`의 차이는 무엇인가?  
   → `__init__`는 레이어를 만들고, `forward`는 계산 흐름을 정의함

---

## 14. 요약

- `nn.Linear(a, b)`는 입력 차원 `a`, 출력 차원 `b`의 선형 레이어
- 선형 회귀는 결국 `Wx + b`
- 파라미터는 `model.parameters()`로 가져옴
- 학습 순서:
  - 예측
  - cost 계산
  - `zero_grad`
  - `backward`
  - `step`
- `forward`는 예측 계산
- `backward`는 기울기 계산
- `nn.Module`은 PyTorch 모델의 기반 클래스
- `super().__init__()`는 그 기반 기능을 활성화하는 필수 초기화
- 클래스 기반 모델 정의는 앞으로 거의 모든 PyTorch 구현의 기본 형식

---

## 15. 한 문장으로 요약

이번 파트는 **`nn.Linear`, `mse_loss`, `optimizer`, `nn.Module`, `forward`, `super().__init__()`가 서로 어떻게 연결되어 하나의 PyTorch 모델 학습 구조를 이루는지 익히는 단계**라고 정리하면 된다.
