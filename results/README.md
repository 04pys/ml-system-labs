# results

실험 결과를 모아서 비교/회고하기 위한 폴더입니다.  
코드(experiments)와 결과를 분리해두면, 실험이 늘어도 비교가 쉬워집니다.

## 저장 원칙
- `results/exp_xxxx_<name>/` 단위로 정리
- 체크포인트 파일은 Git에 올리지 않음
  - 대신 저장 위치 링크 + 메타데이터(학습 step, metric, 해시 등) 기록
- 로그도 필요한 최소만 남김
  - 최종/best metric, 학습 시간, 환경 정보 위주

## 권장 구조
- `results/exp_0001_<name>/`
  - `metrics.json`
  - `figures/`
  - `logs/` (필요한 최소)
  - `README.md` (결과 요약/비교)

## metrics.json 예시 항목
- `task`
- `model`
- `best_metric`
- `final_metric`
- `train_time_sec`
- `seed`
- `environment` (Colab GPU 종류 등)
- `notes` (중요 코멘트)

## 비교할 때 체크
- metric 정의/데이터 split이 같을 때만 비교
- Baseline 대비 변화 1줄 요약 남기기
- 속도 측정은 환경(런타임/GPU)을 반드시 함께 기록
