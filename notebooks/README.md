# notebooks

Colab에서 바로 실행하는 노트북을 모아둡니다.  
빠르게 검증하고 배우는 용도이며, 결과가 의미 있어지면 `experiments/`로 옮겨 재현 가능 형태로 정리합니다.

## 운영 흐름
1) notebooks에서 아이디어/가설을 빠르게 검증  
2) 결과가 의미 있으면 experiments로 스크립트화  
3) 최종 비교/결과는 results에 정리

## 네이밍 규칙
- `01_pytorch_train_loop.ipynb`
- `02_mnist_mlp_baseline.ipynb`
- `03_cnn_regularization.ipynb`
- `04_attention_shapes.ipynb`
- `05_hf_finetune_lora.ipynb`

## 최소 체크리스트
- 첫 셀: 목적/가설 3줄 요약
- 런타임 정보 기록(Colab, GPU 종류)
- seed 고정(가능하면)
- 마지막 셀: 결론/다음 액션 5줄 요약
- 노트북이 길어지면 experiments로 이동

## 주의
- 체크포인트(.pt/.ckpt/.safetensors), 대용량 로그/데이터는 Git 커밋 금지
- 필요하면 링크(Drive 등) + 메타데이터로만 관리
