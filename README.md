# DFC_DEMO

EV6 폴더에 있는 총 808개의 파일에 대해 진행하였습니다.

# ⚡ DFC_DEMO
전기차(EV) 데이터 필터링 및 시각화 데모 프로젝트입니다.

## 🛠️ 주요 기능
* **EV_data_filter.py**: 전기차 데이터에서 SOC missing gap >= 10, Time missing gap >= 12h인 데이터를 필터링합니다. CSV 파일 명을 따로 저장합니다.

* **DFC_apply_total.py / DFC_apply.py**: DFC 모델/알고리즘을 적용합니다. DFC 로직은 Google Drive에 따로 정리하였습니다. (BSG/Members/서민성/DFC 로직.docx)
* ㄴ 계속해서 수정 중입니다.

* **Total_visualization.py**: 기존 데이터를 그래프로 시각화합니다.

* + Compare.py 파일로 DFC 적용 전과 후의 그래프 비교를 plot 해주는 코드 수정 중입니다.

## 🚀 시작하기
이 프로젝트는 Python 기반으로 작동합니다.

```bash
python Total_visualization.py
