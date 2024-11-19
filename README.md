# Interview-models

- 기획의도 
  - 취준생을 위한 맞춤 면접 컨설팅을 위한 언어모델을 생성하는 것이 목적 
  - LLM으로 끝나는 것이 아닌, 카메라 및 TTS STT 기술을 활용하여 컨설팅을 해주는 봇을 만드는 것을 목적으로 함 

---
1. 진행사항 
   - 현재 코드 구성 완료, 데이터셋 구성을 위한 허깅페이스 기획 단계 
   - 2024.11.13일 기준, 데이터셋 구분 완료 출처 : `AIHub` 채용면접 데이터셋 https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71592
   - 허깅페이스 업로드 경로 
     - `val` :  https://huggingface.co/datasets/UICHEOL-HWANG/InterView_Datasets_Val/blob/main/README.md
     - `train` : https://huggingface.co/datasets/UICHEOL-HWANG/InterView_Datasets
   

2. 추후 계획 
   - 데이터셋을 처음 다운로드 받았을 때는, 음성 파일 떄문인진 몰라도 50GB가 넘었던 것으로 파악됨 
   - 막상 정제를 끝내니 50MB 수준으로 내려진 것을 보아하니 데이터셋 정제는 수월할 것으로 파악 
   - 프로젝트 아키텍처 추후 작성 예정