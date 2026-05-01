# 06-06 다층 퍼셉트론으로 MNIST 분류하기 복습 트리거

## 0. 이 실습의 핵심

이전 MNIST 소프트맥스 회귀는 입력층과 출력층만 있는 단층 퍼셉트론에 가까웠음.

이번 실습은 은닉층을 추가한 다층 퍼셉트론, 즉 MLP로 MNIST 손글씨 숫자 이미지를 분류하는 실습임.

핵심 흐름은 다음과 같음.

데이터 로드
→ 정규화
→ 훈련/테스트 분리
→ Tensor 변환
→ DataLoader 구성
→ MLP 모델 정의
→ loss 계산
→ 역전파
→ optimizer로 파라미터 업데이트
→ 테스트 정확도 평가
→ 개별 이미지 예측 확인

---

## 1. MNIST 데이터 구조

MNIST는 손글씨 숫자 이미지 데이터셋임.

각 이미지는 28 x 28 크기의 흑백 이미지이고, 이를 펼치면 784개의 숫자 벡터가 됨.

따라서 MLP의 입력 차원은 다음과 같음.

28 x 28 = 784

즉, 모델에는 이미지가 2차원 그림 그대로 들어가는 것이 아니라, 길이 784짜리 벡터로 들어감.

시스템 관점에서 보면, 이미지는 결국 float 배열이고, MLP는 이 배열에 대해 행렬곱을 수행하는 구조임.

---

## 2. 데이터 로드

sklearn의 fetch_openml을 사용해 MNIST 데이터를 불러옴.

중요한 옵션:

as_frame=False

이 옵션을 넣으면 pandas DataFrame이 아니라 numpy array 형태로 데이터를 받을 수 있음.

MNIST 데이터의 첫 번째 샘플을 출력하면 0~255 범위의 픽셀값들이 나옴.

첫 번째 레이블은 문자열 '5'로 되어 있으므로, 이후 정수형으로 변환함.

핵심 정리:

- mnist.data: 이미지 데이터
- mnist.target: 정답 레이블
- 이미지 하나: 784개 픽셀값
- 레이블: 0~9 중 하나

---

## 3. 정규화

원래 픽셀값은 0~255 범위임.

이를 0~1 범위로 바꾸기 위해 255로 나눔.

X = mnist.data / 255

정규화를 하는 이유:

- 값의 범위가 너무 크면 학습이 불안정해질 수 있음
- 입력값 범위를 작게 맞추면 gradient 기반 학습이 더 안정적으로 진행됨
- 신경망이 더 빠르게 학습될 수 있음

복습 포인트:

정규화는 모델 구조를 바꾸는 것이 아니라, 학습이 잘 되도록 입력 데이터의 스케일을 조정하는 과정임.

---

## 4. 훈련 데이터와 테스트 데이터 분리

train_test_split을 사용해서 데이터를 훈련용과 테스트용으로 나눔.

test_size=1/7로 설정하면 전체 70,000개 중 약 60,000개는 훈련 데이터, 10,000개는 테스트 데이터가 됨.

훈련 데이터는 모델 파라미터를 학습하는 데 사용하고, 테스트 데이터는 학습이 끝난 뒤 모델 성능을 평가하는 데 사용함.

주의:

테스트 데이터는 학습에 사용하면 안 됨.
테스트 데이터는 모델이 처음 보는 데이터에 얼마나 잘 맞는지 확인하기 위한 것임.

---

## 5. Tensor 변환

PyTorch 모델에 넣기 위해 numpy 데이터를 torch Tensor로 변환함.

이미지 데이터:

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)

레이블 데이터:

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

여기서 레이블을 LongTensor로 바꾸는 것이 중요함.

이유:

CrossEntropyLoss는 정답 레이블을 one-hot vector가 아니라 class index 형태로 받음.

예를 들어 숫자 5의 정답은 다음처럼 들어감.

정답: 5

다음처럼 one-hot으로 넣는 것이 아님.

[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

---

## 6. TensorDataset과 DataLoader

TensorDataset은 입력 데이터와 정답 레이블을 하나의 데이터셋으로 묶어줌.

DataLoader는 이 데이터셋을 미니배치 단위로 꺼내줌.

이 실습에서는 batch_size=64를 사용함.

즉, 한 번에 이미지 64개씩 모델에 넣어서 학습함.

훈련용 DataLoader:

- batch_size=64
- shuffle=True

테스트용 DataLoader:

- batch_size=64
- shuffle=False

shuffle=True를 쓰는 이유:

훈련 데이터 순서가 고정되면 학습이 특정 순서에 영향을 받을 수 있으므로, 매 epoch마다 데이터를 섞어주는 것이 좋음.

---

## 7. MLP 모델 구조

모델은 nn.Sequential로 구성됨.

구조는 다음과 같음.

입력층: 784
은닉층 1: 100
활성화 함수: ReLU
은닉층 2: 100
활성화 함수: ReLU
출력층: 10

흐름:

784차원 입력
→ Linear(784, 100)
→ ReLU
→ Linear(100, 100)
→ ReLU
→ Linear(100, 10)

출력층의 10은 숫자 0~9에 대한 점수임.

중요:

마지막에 Softmax를 직접 붙이지 않음.

이유:

PyTorch의 CrossEntropyLoss는 내부적으로 softmax 계열 처리를 포함하기 때문에, 모델은 raw score, 즉 logits를 출력하면 됨.

따라서 모델의 마지막 출력은 확률이라기보다 클래스별 점수라고 이해하는 것이 더 정확함.

---

## 8. ReLU의 역할

ReLU는 음수는 0으로 만들고, 양수는 그대로 통과시키는 활성화 함수임.

ReLU를 넣는 이유:

선형층만 여러 개 쌓으면 결국 하나의 큰 선형 변환과 비슷해짐.

비선형 활성화 함수가 있어야 모델이 복잡한 패턴을 학습할 수 있음.

즉, MLP가 단순 소프트맥스 회귀보다 강력한 이유는 은닉층과 ReLU를 통해 비선형 패턴을 학습할 수 있기 때문임.

---

## 9. 손실 함수와 옵티마이저

손실 함수:

CrossEntropyLoss

역할:

모델이 출력한 10개 클래스 점수와 실제 정답 레이블을 비교해서 loss를 계산함.

옵티마이저:

Adam

역할:

loss.backward()로 계산된 gradient를 이용해 모델의 weight와 bias를 업데이트함.

학습률:

lr=0.01

학습률은 파라미터를 한 번에 얼마나 크게 수정할지 결정하는 값임.

너무 크면 학습이 불안정해질 수 있고, 너무 작으면 학습이 느려질 수 있음.

---

## 10. 학습 루프

학습은 epoch 단위로 반복됨.

이 실습에서는 총 3번의 epoch 동안 학습함.

각 미니배치마다 다음 순서가 반복됨.

1. optimizer.zero_grad()
2. y_pred = model(data)
3. loss = loss_fn(y_pred, targets)
4. loss.backward()
5. optimizer.step()

각 단계의 의미:

optimizer.zero_grad()

이전 미니배치에서 계산된 gradient를 초기화함.
PyTorch는 기본적으로 gradient를 누적하므로, 매 학습 step마다 초기화가 필요함.

model(data)

순전파.
입력 데이터를 모델에 넣어 예측값을 계산함.

loss_fn(y_pred, targets)

예측값과 정답을 비교해서 loss를 계산함.

loss.backward()

역전파.
loss를 줄이기 위해 각 파라미터를 어느 방향으로 바꿔야 하는지 gradient를 계산함.

optimizer.step()

계산된 gradient를 이용해 실제 weight와 bias를 업데이트함.

핵심:

backward는 gradient 계산.
step은 실제 파라미터 업데이트.

---

## 11. 평가 모드

학습 후에는 테스트 데이터로 모델을 평가함.

먼저 model.eval()을 호출함.

model.eval()의 의미:

모델을 학습 모드가 아니라 추론 모드로 전환함.

이번 MLP에서는 dropout이나 batch normalization이 없어서 차이가 크게 느껴지지 않을 수 있지만, 일반적으로 평가할 때는 반드시 호출하는 습관을 들이는 것이 좋음.

그다음 torch.no_grad()를 사용함.

torch.no_grad()의 의미:

평가 단계에서는 gradient 계산이 필요 없으므로, gradient 추적을 끔.

효과:

- 메모리 사용량 감소
- 계산 속도 향상
- 불필요한 연산 방지

---

## 12. 정확도 계산

테스트 데이터에 대해 모델 출력을 얻음.

outputs = model(data)

outputs의 shape는 대략 다음과 같음.

batch_size x 10

예를 들어 batch_size가 64라면:

64 x 10

각 행은 하나의 이미지에 대한 10개 클래스 점수임.

torch.max(outputs.data, 1)을 사용하면 각 이미지마다 가장 큰 점수를 가진 클래스 인덱스를 구할 수 있음.

dim=1인 이유:

outputs가 [배치 크기, 클래스 수] 형태이기 때문.
각 행 안에서 가장 큰 클래스 점수를 찾아야 하므로 클래스 차원인 dim=1을 기준으로 max를 수행함.

이후 predicted와 targets를 비교해 맞은 개수를 correct에 더함.

최종 정확도:

맞은 개수 / 전체 테스트 데이터 개수

실습에서는 약 96% 정확도가 나옴.

---

## 13. 개별 이미지 예측

테스트 데이터 중 하나의 인덱스를 선택해서 모델에 넣고 예측을 확인함.

이때 data = X_test[index]는 이미지 하나임.

즉, shape가 batch_size x 10이 아니라 10개 클래스 점수만 가진 1차원 벡터가 됨.

그래서 torch.max(output.data, 0)을 사용함.

dim=0인 이유:

개별 이미지 하나의 출력은 [10] 형태이기 때문.
유일한 차원에서 가장 큰 값을 찾으면 됨.

정리:

배치 예측 outputs: [batch_size, 10] → torch.max(outputs, 1)
개별 예측 output: [10] → torch.max(output, 0)

---

## 14. 소프트맥스 회귀와 MLP의 차이

소프트맥스 회귀:

입력 784
→ 출력 10

중간 은닉층이 없음.

MLP:

입력 784
→ 은닉층 100
→ 은닉층 100
→ 출력 10

은닉층과 ReLU가 있기 때문에 더 복잡한 비선형 패턴을 학습할 수 있음.

MNIST처럼 단순한 이미지 분류 문제에서도 MLP는 소프트맥스 회귀보다 더 높은 표현력을 가짐.

하지만 MLP는 이미지를 1차원 벡터로 펼쳐서 보기 때문에, 이미지의 공간적 구조를 직접 활용하지는 못함.

이 한계 때문에 이후 CNN이 중요해짐.

---

## 15. 시스템 관점에서의 의미

MLP의 핵심 연산은 Linear layer임.

Linear layer는 본질적으로 행렬곱임.

예를 들어 첫 번째 층은 다음과 같은 연산을 수행함.

입력: batch_size x 784
가중치: 784 x 100
출력: batch_size x 100

즉, MLP 학습과 추론은 대부분 행렬 연산 중심으로 이루어짐.

시스템 관점에서 중요한 질문:

- 입력 데이터는 메모리에 어떻게 저장되는가?
- batch size가 커지면 행렬곱 크기는 어떻게 변하는가?
- weight matrix는 얼마나 자주 재사용되는가?
- GPU/NPU에서는 이 행렬곱을 어떻게 병렬화하는가?
- cache locality와 memory bandwidth가 성능에 어떤 영향을 주는가?

윤수님의 목표인 메모리 시스템, PIM, AI workload 최적화 관점에서는 MLP를 단순 분류 모델로만 보지 말고, dense matrix multiplication workload의 기본 예시로 보면 좋음.

---

## 16. 이 실습에서 반드시 기억할 것

1. MNIST 이미지는 28 x 28이지만, MLP에는 784차원 벡터로 들어감.

2. 픽셀값은 0~255에서 0~1로 정규화함.

3. 레이블은 CrossEntropyLoss를 위해 LongTensor class index로 넣음.

4. DataLoader는 데이터를 미니배치 단위로 꺼내줌.

5. MLP 구조는 784 → 100 → 100 → 10임.

6. ReLU는 비선형성을 추가해 단순 선형 모델보다 복잡한 패턴을 학습하게 해줌.

7. CrossEntropyLoss를 쓸 때 마지막에 Softmax를 직접 붙이지 않음.

8. backward는 gradient 계산이고, optimizer.step은 실제 파라미터 업데이트임.

9. 평가할 때는 model.eval()과 torch.no_grad()를 사용함.

10. 배치 예측에서는 torch.max(outputs, 1), 개별 예측에서는 torch.max(output, 0)을 사용함.

---

## 17. 한 줄 요약

MNIST 이미지를 784차원 벡터로 정규화한 뒤, DataLoader로 미니배치 학습을 수행하고, 784 → 100 → 100 → 10 구조의 MLP를 CrossEn
::contentReference[oaicite:1]{index=1}
tropyLoss와 Adam으로 학습하여 손글씨 숫자를 분류하는 실습.
