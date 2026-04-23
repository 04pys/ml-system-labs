# 05-04 소프트맥스 회귀로 MNIST 데이터 분류하기

## 1. 이 실습이 하는 일

- MNIST 손글씨 숫자 이미지를 0~9 중 하나로 분류하는 실습
- 모델은 가장 단순한 다중 클래스 분류기인 소프트맥스 회귀 사용
- CNN이 아니라, 이미지를 펼쳐서 선형층에 넣는 방식이라는 점이 핵심

---

## 2. MNIST를 먼저 기억하기

- 훈련 데이터: 60,000개
- 테스트 데이터: 10,000개
- 각 이미지는 흑백 28×28
- 따라서 한 장을 펼치면 784차원 벡터가 됨

복습 포인트:

- 원본 이미지 shape: `(1, 28, 28)`
- 펼친 뒤 shape: `(784,)`
- 배치 입력 shape: `(batch_size, 784)`

즉 이 실습은  
**28×28 이미지를 784차원 벡터로 펴서 10개 클래스 확률을 예측하는 문제**로 볼 수 있음

---

## 3. 모델 구조

모델의 핵심은 아래 한 줄로 요약 가능

    nn.Linear(784, 10)

뜻:

- 입력 784차원
- 출력 10차원
- 각 출력값은 숫자 0~9 각각에 대한 score라고 생각하면 됨

이후 softmax를 통해 10개 클래스에 대한 확률처럼 해석할 수 있음

정리하면:

- 입력: `X`
- 선형 변환: `WX + b`
- 출력: 10개 클래스에 대한 점수
- 손실 함수: Cross Entropy

---

## 4. 왜 소프트맥스 회귀라고 부르나

이진 분류에서는 sigmoid를 많이 쓰지만,  
MNIST처럼 정답 후보가 10개인 문제는 다중 클래스 분류이므로 softmax를 사용함

softmax의 역할:

- 10개의 score를 받아
- 각 클래스에 대한 확률처럼 바꿔줌
- 전체 합은 1이 됨

즉,  
**이 문제는 하나만 고르는 다중 클래스 분류 문제이므로 softmax 회귀를 사용**하는 것임

---

## 5. 하이퍼파라미터

실습에서 자주 보는 설정:

    training_epochs = 15
    batch_size = 100

### 하이퍼파라미터가 무엇인가

하이퍼파라미터는  
**모델이 학습 중에 자동으로 배우는 값이 아니라, 사람이 학습 전에 미리 정하는 설정값**임

구분:

- 모델이 학습으로 바꾸는 값: `W`, `b`
- 사람이 미리 정하는 값: epoch 수, batch size, learning rate 등

### `training_epochs = 15`

뜻:

- 전체 훈련 데이터를 15번 반복해서 학습

예를 들어 훈련 데이터가 60,000개면

- 1 epoch = 60,000개 전체를 한 번 다 보는 것
- 15 epochs = 그 과정을 15번 반복

왜 15인가:

- MNIST는 비교적 단순해서 너무 큰 epoch가 없어도 학습이 잘 됨
- 너무 적으면 덜 배울 수 있고
- 너무 많으면 시간만 더 들 수 있음
- 15는 예제용으로 무난한 값

### `batch_size = 100`

뜻:

- 한 번에 데이터 100개씩 묶어서 학습

예를 들어 훈련 데이터 60,000개라면

- 한 epoch당 600개의 batch
- 즉 한 epoch 동안 600번 업데이트 기회가 생김

왜 100인가:

- 너무 작으면 gradient가 들쭉날쭉할 수 있음
- 너무 크면 메모리 부담이 커지고 업데이트 횟수가 줄어듦
- 100은 속도와 안정성의 균형이 괜찮은 편

핵심 한 줄:

- `epoch`는 전체 데이터를 몇 번 반복해서 볼지
- `batch size`는 한 번에 몇 개씩 끊어서 볼지

---

## 6. DataLoader가 하는 일

실습 코드에서 자주 나오는 형태:

    data_loader = DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

### DataLoader가 무엇인가

`DataLoader`는  
**dataset에서 데이터를 배치 단위로 꺼내고, 섞고, 반복문에서 쉽게 쓰게 해주는 도구**임

쉽게 말하면:

- `dataset` = 원본 데이터 보관소
- `DataLoader` = 학습할 때 꺼내주는 공급기

### 각 옵션 의미

#### `dataset=mnist_train`

- 학습용 MNIST 데이터셋을 사용하겠다는 뜻

#### `batch_size=batch_size`

- 한 번에 100개씩 꺼냄

#### `shuffle=True`

- 매 epoch마다 데이터를 섞음
- 훈련 데이터는 보통 섞는 것이 좋음

#### `drop_last=True`

- 마지막 batch의 개수가 100보다 작으면 버림
- 배치 크기를 일정하게 유지하기 위해 사용

### 실제 학습에서의 모습

    for X, Y in data_loader:
        ...

이 반복문에서 매번

- `X`: 이미지 100개
- `Y`: 정답 100개

를 받아 학습하게 됨

---

## 7. forward가 무엇이었는지 다시 정리

클래스로 모델을 만들 때 보던 `forward`가 바로 **순전파**임

예시:

    class SoftmaxClassifierModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(784, 10)

        def forward(self, x):
            return self.linear(x)

### 의미

- `__init__` : 사용할 부품 정의
- `forward` : 입력이 들어왔을 때 어떤 계산을 할지 정의

즉

    def forward(self, x):
        return self.linear(x)

는  
입력 `x`를 선형층에 통과시켜 출력하라는 뜻

### 헷갈리지 말 것

아래는 틀린 형태:

    return self.linear(self, x)

정확한 것은:

    return self.linear(x)

### `model(X)`와 `forward(X)`의 관계

PyTorch에서는 보통 직접 `forward(X)`를 쓰지 않고

    model(X)

처럼 쓰는데, 내부적으로 `forward(X)`가 호출됨

즉:

- `model(X)` = 순전파 실행
- 이 과정에서 예측값 계산

---

## 8. 순전파, 손실 계산, 역전파의 흐름

전체 학습 흐름은 아래처럼 이해하면 됨

    optimizer.zero_grad()
    prediction = model(X)
    cost = criterion(prediction, Y)
    cost.backward()
    optimizer.step()

### 1) `model(X)`

- 순전파
- 입력 `X`를 받아 예측값 계산

### 2) `cost = criterion(prediction, Y)`

- 예측값과 정답을 비교해 손실 계산

### 3) `cost.backward()`

- 역전파
- 손실함수를 기준으로 계산 그래프를 거꾸로 따라가며
- `W`, `b`에 대한 미분값을 자동 계산해서 `.grad`에 저장

즉 질문에서 이해한 대로  
**손실 함수에 대해 `W`, `b`의 미분계수를 구해 저장하는 단계**가 맞음

다만 중요한 점:

- `backward()`는 gradient를 계산해서 저장만 함
- 실제 파라미터 업데이트는 `optimizer.step()`에서 일어남

### 4) `optimizer.step()`

- 저장된 gradient를 이용해 `W`, `b` 업데이트

---

## 9. 손실 함수: CrossEntropyLoss

MNIST 다중 클래스 분류에서는 보통

    nn.CrossEntropyLoss()

를 사용

이 손실 함수는

- softmax
- negative log likelihood

를 합쳐 생각할 수 있는 손실 함수

실무적으로는  
출력에 직접 softmax를 먼저 씌우기보다,  
**raw score를 그대로 `CrossEntropyLoss`에 넣는 경우가 많다**는 점 기억

즉 모델 출력은 확률이 아니라 score여도 됨

---

## 10. 예측은 어떻게 하나

모델이 10개의 score를 내면,  
그중 가장 큰 값을 가진 index를 예측 클래스로 선택함

예:

- 출력이 `[score0, score1, ..., score9]`
- 가장 큰 값의 위치가 7이면
- 예측 숫자는 7

즉 최종 분류는  
**10개 중 가장 점수가 큰 클래스를 고르는 것**임

---

## 11. shape 감각 꼭 잡기

MNIST 실습에서 shape이 자주 헷갈림

### 입력

- 원본 이미지 1장: `(1, 28, 28)`
- 펼친 뒤: `(784,)`

### 배치 입력

- 배치 크기 100이면: `(100, 784)`

### 출력

- 클래스가 10개이므로: `(100, 10)`

즉 한 배치에 대해

- 입력: 100장 × 784차원
- 출력: 100장 × 10개 클래스 점수

---

## 12. 이 실습의 본질

이 실습은 복잡한 신경망을 배우는 것이 아니라,  
**PyTorch로 다중 클래스 분류의 가장 기본 구조를 익히는 실습**임

배울 것:

1. 이미지를 벡터로 펼쳐 입력 만들기
2. `Linear(784, 10)`으로 score 계산하기
3. `CrossEntropyLoss`로 손실 구하기
4. `backward()`로 gradient 계산하기
5. optimizer로 파라미터 업데이트하기

즉,  
**다중 클래스 분류의 가장 기본적인 학습 파이프라인을 익히는 실습**이라고 보면 됨

---

## 13. 이 실습이 단순한 이유

이 모델은 CNN이 아니라 단순 선형 모델이므로

- 이미지의 공간 정보 활용이 약함
- 성능은 아주 높지 않을 수 있음
- 하지만 구조가 단순해서 학습 흐름 이해에는 매우 좋음

즉 이 실습의 목적은  
**최고 성능보다는 분류 모델의 기본 동작 원리 이해**에 가까움

---

## 14. 최종 한 번에 복습

### 핵심 흐름

- MNIST 이미지를 28×28에서 784차원으로 펼침
- `Linear(784, 10)`으로 10개 클래스 score 계산
- `CrossEntropyLoss`로 손실 계산
- `backward()`로 gradient 계산
- `optimizer.step()`으로 파라미터 업데이트

### 핵심 개념

- 하이퍼파라미터: 사람이 미리 정하는 학습 설정값
- epoch: 전체 데이터를 몇 번 반복할지
- batch size: 한 번에 몇 개씩 학습할지
- DataLoader: 데이터를 배치 단위로 꺼내는 도구
- forward: 순전파, 입력으로부터 예측값 계산
- backward: 역전파, 손실로부터 gradient 계산

### 이 실습을 한 문장으로

**MNIST 이미지를 784차원 벡터로 펼쳐 소프트맥스 회귀로 10개 숫자 중 하나를 분류하는 기본 다중 클래스 분류 실습**

---

## 15. 헷갈리기 쉬운 포인트만 마지막으로

### 1) `forward`는 순전파인가?
- 맞음
- 입력을 넣어서 예측값을 계산하는 과정

### 2) `model(X)`를 하면 `forward(X)`가 호출되나?
- 맞음
- 보통 직접 `forward`를 부르지 않고 `model(X)`처럼 사용

### 3) `cost.backward()`는 역전파인가?
- 맞음
- `W`, `b`에 대한 gradient를 자동 계산해 `.grad`에 저장

### 4) `backward()`가 파라미터를 직접 바꾸나?
- 아님
- 실제 업데이트는 `optimizer.step()`에서 일어남

### 5) `DataLoader`는 dataset 그 자체인가?
- 아님
- dataset을 학습용 배치로 꺼내주는 도구임

---

## 16. 초압축 복습 버전

- MNIST: 28×28 손글씨 숫자 이미지, 10개 클래스 분류
- 입력 이미지를 784차원 벡터로 펼침
- 모델은 `Linear(784, 10)`
- `forward`는 순전파
- `DataLoader`는 데이터를 batch 단위로 공급
- `training_epochs=15`는 전체 데이터를 15번 반복 학습
- `batch_size=100`은 한 번에 100개씩 학습
- `cost.backward()`는 역전파, gradient 계산
- `optimizer.step()`은 실제 파라미터 업데이트
