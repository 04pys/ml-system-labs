# Python Class 기초

## 왜 배웠는가

PyTorch의 많은 구현은 **python class 기반**이다.

예를 들어 앞으로 자주 보게 될 것들:
- `nn.Module`
- `Dataset`
- `DataLoader`와 연결되는 사용자 정의 데이터셋
- 모델 클래스
- 학습에 필요한 여러 구성요소 묶기

즉, class를 알아야  
**PyTorch 코드가 왜 저런 구조인지** 이해할 수 있다.

---

## 핵심 개념

### 1. 함수와 클래스의 차이

함수는 동작 하나를 수행하는 데 적합하다.  
하지만 **상태(state)를 따로따로 가지는 여러 객체**를 만들고 싶으면 클래스가 더 적합하다.

위키독스 예시에서는 덧셈기를 함수로 만들면 전역변수를 따로 관리해야 하고,  
여러 개의 독립적인 덧셈기를 만들려면 함수/전역변수도 여러 개 필요해진다.  
반면 클래스로 만들면 객체마다 자기 `result`를 따로 가질 수 있다. :contentReference[oaicite:1]{index=1}

---

### 2. class = 설계도, object = 실제 인스턴스

- **class**: 틀, 설계도
- **object(instance)**: 그 틀로 만든 실제 대상

예:
- `Calculator`는 클래스
- `cal1 = Calculator()`는 객체 생성

즉 같은 클래스에서 여러 객체를 만들 수 있고,  
각 객체는 자기 상태를 따로 가진다. :contentReference[oaicite:2]{index=2}

---

### 3. `__init__`

`__init__`은 **객체를 만들 때 자동으로 호출되는 초기화 함수**다.  
보통 객체의 초기 상태를 여기서 정한다. 위키독스도 이를 생성자라고 설명한다. :contentReference[oaicite:3]{index=3}

예:
abc
class Calculator:
    def __init__(self):
        self.result = 0
abc

→ 객체가 생성되면 `result = 0`으로 시작

---

### 4. `self`

`self`는 **객체 자기 자신**을 가리킨다.

예:
abc
self.result
abc

은  
“이 객체의 result”라는 뜻이다.

즉 클래스 안에서는 전역변수 대신  
`self.변수명`으로 객체의 상태를 저장한다고 이해하면 된다.

---

### 5. method

클래스 안에 정의된 함수는 보통 **메서드(method)** 라고 부른다.

예:
abc
def add(self, num):
    self.result += num
    return self.result
abc

이 메서드는  
“이 객체의 result에 num을 더하는 동작”이다.

---

## 이번 실습에서 기억할 것

- 함수는 단일 동작 구현에는 편함
- 클래스는 **상태와 동작을 함께 묶을 때** 좋음
- 객체마다 독립적인 값을 가질 수 있음
- `__init__`은 초기화
- `self`는 객체 자신의 데이터 접근
- PyTorch 코드는 이런 구조를 전제로 많이 작성됨

---

## PyTorch / ML 공부에서의 시사점

### 1. 모델도 결국 class로 만든다
앞으로 자주 보게 될 형태:

abc
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...
abc

즉 모델은 단순 함수가 아니라  
**파라미터와 동작을 함께 가진 객체**다.

---

### 2. 데이터셋도 class로 만든다
직접 데이터셋을 만들 때도 보통 class를 쓴다.

이유:
- 데이터 저장
- 길이 반환
- 특정 인덱스의 샘플 반환

이런 걸 하나로 묶어야 하기 때문

---

### 3. ML 코드는 “구조화”가 중요하다
처음에는 함수만으로도 코드를 짤 수 있지만,  
모델/데이터/학습과정이 커질수록 class가 훨씬 관리하기 쉽다.

즉 class는 단순 문법이 아니라  
**복잡한 ML 코드를 정리하는 기본 도구**다.

---

## 복습 트리거

이 파트를 배운 이유는  
“파이썬 class 자체”가 목적이 아니라,  
**PyTorch 코드가 왜 class 기반으로 작성되는지 이해하기 위해서**였다.

앞으로 PyTorch에서 class를 보면 이렇게 생각하면 됨:

- `__init__` : 필요한 부품/변수 준비
- `self.xxx` : 이 객체가 가진 상태
- 메서드 : 이 객체가 하는 일
- 객체 생성 : 실제 모델/데이터셋 하나 만들기

---

## 한 줄 정리

Python class는  
**상태와 동작을 하나로 묶는 방법**이고,  
PyTorch와 이후 ML 코드 대부분은 이 구조 위에서 돌아간다.
