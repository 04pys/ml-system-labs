# 미니 배치와 DataLoader 복습 정리

## 1. 왜 미니 배치를 쓰는가

전체 데이터를 한 번에 다 받아와서 cost를 계산하고 경사 하강법을 수행하는 방법은 **배치 학습(batch gradient descent)** 이다.  
하지만 실제 데이터는 매우 많을 수 있어서, 전체 데이터를 한 번에 사용하는 방식은 계산량이 크고 느리며, 경우에 따라서는 메모리 한계 때문에 어렵기도 하다. 그래서 전체 데이터를 더 작은 묶음으로 나누어 학습하는 **미니 배치 학습(mini-batch gradient descent)** 을 사용한다.

내가 기억할 핵심은 다음과 같다.

- 전체 데이터를 한 번에 쓰는 방법 = **배치 학습**
- 여러 개의 작은 배치로 나누고, **각 배치마다 cost 계산 → backward → optimizer.step()** 을 수행하는 방법 = **미니 배치 학습**
- 전체 데이터를 한 번 전부 다 쓰면 **1 epoch** 이다.

---

## 2. 배치 크기, 에포크, 이터레이션

### (1) batch size
한 번의 업데이트에서 사용하는 샘플 수이다.  
예를 들어 `batch_size=2`면 한 번에 샘플 2개씩 묶어서 학습한다.

### (2) epoch
전체 훈련 데이터를 **한 번 모두 사용한 주기**이다.  
데이터가 100개이고 `batch_size=10`이면, 10번의 배치 학습이 끝나야 1 epoch다.

### (3) iteration
**한 epoch 안에서 매개변수가 업데이트되는 횟수**이다.  
예를 들어 전체 데이터가 2000개이고 `batch_size=200`이면, 한 epoch 안에서 총 10번 업데이트가 일어나므로 iteration은 10이다.

### 복습 트리거
- **epoch = 전체 데이터 1바퀴**
- **iteration = 한 epoch 안의 update 횟수**
- **batch size = 한 번에 묶어서 넣는 샘플 수**

---

## 3. Dataset과 DataLoader의 역할 구분

PyTorch는 `torch.utils.data.Dataset`과 `torch.utils.data.DataLoader`를 제공하며, 이를 통해 미니 배치 학습, 셔플, 병렬 처리를 쉽게 할 수 있다. 기본 흐름은 **Dataset을 정의하고, 그것을 DataLoader에 전달하는 것**이다.

### Dataset
Dataset은 **데이터를 보관하고, i번째 샘플을 꺼내는 규칙을 정의한 객체**다.

핵심 메서드

- `__len__()` : 전체 샘플 수 반환
- `__getitem__(idx)` : idx번째 샘플 1개 반환

즉, Dataset은  
**데이터 하나를 어떻게 꺼낼 것인가**를 정의한다.

### DataLoader
DataLoader는 Dataset을 받아서

- 배치 단위로 묶고
- 필요하면 섞고
- 반복문에서 순서대로 꺼내게 해주는 도구다.

즉, DataLoader는  
**Dataset에서 샘플들을 읽어와 mini-batch 단위로 넘겨주는 객체**라고 이해하면 된다.

---

## 4. TensorDataset이란

`TensorDataset`은 **텐서를 입력으로 받아 Dataset 형태로 저장해주는 도구**다.  
즉 이미 `x_train`, `y_train`이 텐서로 준비되어 있다면, 간단히 Dataset으로 묶을 수 있다.

예시:

abcpython
dataset = TensorDataset(x_train, y_train)
abc

여기서 `dataset[i]`는 사실상 `(x_train[i], y_train[i])`를 돌려주는 것과 같다.

### 복습 트리거
- 텐서가 이미 준비됨 → **TensorDataset**
- 파일 읽기, 전처리, 복잡한 로직 필요 → **CustomDataset 고려**

---

## 5. DataLoader의 핵심 인자

예시는 다음과 같다.

abcpython
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
abc

핵심 인자:

- `dataset` : 어떤 데이터셋을 쓸지
- `batch_size` : 미니 배치 크기
- `shuffle` : epoch마다 순서를 섞을지 여부

특히 `shuffle=True`이면 **epoch마다 데이터셋 순서가 섞인다.**  
이는 모델이 데이터 순서에 과도하게 익숙해지는 것을 줄이는 데 도움을 준다.

### 복습 트리거
- `shuffle=True`는 **배치 내부만 섞는 게 아니라**, 보통 **전체 데이터 순서를 섞은 뒤 batch로 자른다**
- 그래서 **epoch마다 학습 순서가 달라질 수 있다**

---

## 6. DataLoader가 iterator처럼 동작한다는 말의 의미

학습 코드는 보통 이렇게 돈다.

abcpython
for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
abc

이 말은 `dataloader`가 **반복문에서 한 배치씩 꺼내주는 객체**처럼 동작한다는 뜻이다.  
즉 리스트 전체를 한 번에 반환하는 것이 아니라, **반복할 때마다 다음 배치를 내놓는다**는 점에서 iterator처럼 이해하면 된다.

내가 기억할 포인트:

- `dataset[i]`는 **샘플 1개**
- `dataloader`를 반복하면 **배치 1개씩**
- 그래서 `for x_batch, y_batch in dataloader:` 같은 코드가 가능함

---

## 7. batch 차원이 추가된다는 뜻

헷갈렸던 부분 정리.

예를 들어

abcpython
x_train.shape = (5, 3)
y_train.shape = (5, 1)
abc

이면 의미는 보통 다음과 같다.

- 전체 샘플 수 = 5개
- 입력 샘플 하나의 shape = `(3,)`
- 정답 샘플 하나의 shape = `(1,)`

이때 `batch_size=2`로 DataLoader가 묶으면:

- `x_batch.shape = (2, 3)`
- `y_batch.shape = (2, 1)`

이 된다.

즉 **샘플 하나의 shape 앞에 batch 크기 축이 붙는다**는 뜻이다.

### 복습 트리거
- 전체 데이터 shape와 샘플 하나의 shape를 구분할 것
- `x_train.shape=(5,3)`에서 `5`는 **샘플 수**
- 따라서 batch로 2개를 꺼내면 `(2,5,3)`이 아니라 `(2,3)`이 됨

---

## 8. 학습 흐름

핵심 순서는 다음과 같다.

1. `x_train`, `y_train`을 텐서로 준비
2. `TensorDataset(x_train, y_train)`으로 dataset 생성
3. `DataLoader(dataset, batch_size=2, shuffle=True)` 생성
4. `for batch_idx, samples in enumerate(dataloader):`
5. 배치별로 `prediction = model(x_train)`
6. `cost = F.mse_loss(prediction, y_train)`
7. `optimizer.zero_grad()`
8. `cost.backward()`
9. `optimizer.step()`

즉 **배치별로 cost를 계산하고, 그 배치 기준으로 파라미터를 업데이트하는 구조**다.  
이것이 미니 배치 학습의 핵심이다.

---

## 9. CustomDataset이란

PyTorch의 `Dataset`을 상속하여 직접 데이터셋 클래스를 만드는 방법이다.  
기본적으로 `__init__`, `__len__`, `__getitem__`을 오버라이드한다.

예시 핵심:

abcpython
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = ...
        self.y_data = ...

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
abc

의미:

- `__init__` : 데이터 준비 / 전처리
- `__len__` : 총 샘플 개수
- `__getitem__` : idx번째 샘플 반환

---

## 10. CustomDataset이 성능이 더 좋다는 주장은 왜 근거가 없는가

내가 같은 값의 `x_train`, `y_train`으로

- `TensorDataset(x_train, y_train)`
- `CustomDataset()`

두 방식을 비교했을 때 cost 차이가 난 적이 있었음

그런데 **같은 값, 같은 shape의 샘플을 반환한다면**, `TensorDataset`과 `CustomDataset`은 학습 관점에서 본질적으로 같은 데이터를 공급한다.

따라서 다음 결론이 맞다.

- **CustomDataset을 쓴다고 해서 본질적으로 더 학습이 잘되는 것은 아님**
- 성능 차이가 났다면 보통 원인은
  - 모델 초기 가중치 차이
  - `shuffle=True`에 따른 배치 순서 차이
  - optimizer / model 재사용 여부
  - epoch 수, learning rate 차이
  - 실험 실행 순서 차이
  등이다

즉 CustomDataset은 **성능 향상 도구**가 아니라 **데이터 로딩 방식을 직접 정의하기 위한 도구**다.

### 복습 트리거
- `CustomDataset`의 장점 = **유연성**
- `TensorDataset`의 장점 = **간단함**
- 둘 중 무엇이 더 "정확도가 높다" 또는 "cost가 잘 내려간다"는 일반론은 없음

---

## 11. 언제 TensorDataset, 언제 CustomDataset인가

### TensorDataset을 쓰면 좋은 경우
- 입력과 정답이 이미 텐서로 준비되어 있음
- 간단한 실습
- 표 형태 데이터, 작은 실험

### CustomDataset이 필요한 경우
- 파일에서 이미지를 읽어와야 함
- 텍스트를 토크나이즈해야 함
- 샘플마다 전처리가 다름
- 입력이 여러 종류임
- 메타데이터를 함께 반환해야 함

즉, **복잡한 데이터 처리 때문에 CustomDataset을 만든다**고 이해하면 된다.

---

## 12. 최종 암기 포인트

### 핵심 개념 1
- 전체 데이터 한 번 사용 = **1 epoch**
- 한 번의 파라미터 업데이트 = **1 iteration**
- 한 번에 묶는 샘플 수 = **batch size**

### 핵심 개념 2
- `Dataset` = 샘플 1개를 어떻게 꺼낼지 정의
- `DataLoader` = 그것을 batch로 묶어서 반복문에서 꺼내게 해줌

### 핵심 개념 3
- `DataLoader(..., shuffle=True)`면 epoch마다 순서가 섞임

### 핵심 개념 4
- `TensorDataset`은 텐서를 바로 Dataset으로 묶는 도구

### 핵심 개념 5
- `CustomDataset`은 성능 향상용이 아니라 **직접 데이터 로딩 규칙을 정의하기 위한 도구**
- 같은 데이터를 같은 shape로 공급하면 `TensorDataset`보다 본질적으로 더 좋다고 말할 근거는 없음

---

## 13. 마지막 한 줄 요약

**Dataset은 샘플 1개를 정의하고, DataLoader는 그것을 mini-batch 단위로 반복해서 꺼내게 해주는 도구이며, CustomDataset은 성능 향상용이 아니라 데이터 로딩을 유연하게 직접 설계하기 위한 방법이다.**
