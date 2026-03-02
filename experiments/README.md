# experiments

재현 가능한 실험 단위를 보관합니다.  
노트북에서 검증한 내용을 “스크립트 + 설정 + 재현 방법” 형태로 고정하는 목적입니다.

## 기본 구조
각 실험은 아래 구조를 기본으로 사용합니다.

- `exp_0001_<name>/`
  - `README.md`  (핵심)
  - `train.py`
  - `eval.py` (또는 train에 포함)
  - `config.yaml` (실험이 늘어나면 필수에 가까움)
  - `requirements.txt` (또는 `environment.txt`)

## exp README 필수 항목
- 목표(Problem)
- 가설(Hypothesis)
- 데이터(Data): 다운로드/전처리 방법(명령/링크)
- 방법(Method): 모델/변경점 핵심
- 설정(Config): lr, batch, epochs, seed 등
- 결과(Results): metric 표 + 핵심 그래프(또는 링크)
- 해석(Analysis): 왜 이렇게 나왔는지, 다음 액션
- 재현(How to run): 복붙 가능한 실행 명령

## 번호/분리 기준
- 목표/데이터/모델이 크게 바뀌면 새 exp로 분리
- 단순 튜닝은 같은 exp 안에서 config 버전으로 관리
- 공통 코드가 반복되기 시작하면 그때 `src/` 도입 고려
