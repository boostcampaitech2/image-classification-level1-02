# 1 데이터 분석 및 ToDoList

*./mask_dataset_V100_server/EDA.ipynb 파일 참고*

## ToDoList
  - [ ] 18개의 라벨을 모두 예측하는 것과 성별/나이/마스크 라벨을 따로 예측하는 것중에서 어떤것이 더 효과적인가?
    - [ ] 18개의 라벨을 모두 예측하는 모델을 hparams tunning으로 훈련해 보기
      - 바르게 착용 : 바르지 않음 : 미착용 = 5 : 1 : 1
      - 바르게 착용을 subsample 해서 훈련할 것
      - 바르지 않음, 미착용 기준으로 60대 이상의 남녀 모두 수가 현저히 적음
      - 60세 이상 여성 109명 , 남성 83명
      - 모든 라벨을 100명씩 subsample 하면 1800장의 사진
        - [ ] 1800장으로 일단 훈련 돌려보기 -> 결과 : 
    - [ ] 성별 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기
    - [ ] 나이 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기
    - [ ] 마스크 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기


## DataLoader 설계
  - pre-transform과 tranform을 나눠서 동작시키면 feed에서 더 빠르게 연산이 가능하다.
  - 
