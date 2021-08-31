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
        - [x] 1800장으로 일단 훈련 돌려보기
          - 결과 : acc 50% 반반 찍기
        - [x] 1800장으로 CrossEntropyLoss + F1Loss 로 돌려보기
          -  p = 0.4 에서 F1score가 0.8 까지 오르는 것을 볼 수 있음
        - [x] Augmentation을 이용해서 데이터 경우의 수 늘려보기
          - 성능 향상 없음, [0,1] 로 normalize 한 후 사용하는 것이 가장 좋음
          - normalization, mean subtraction에서도 성능 향상 발견 없음.
        - [ ] LabelSmoothing, MixUP, CutMix 사용해보기
          - [x] MixUp 성능 안좋음 (코드상의 실수는 없었나?)
          - [x] CutMix 적용
            - CrossEntropyLoss가 지속적으로 떨어지는 경향을 보이지만 전체적인 성능향상은 없었다. 그리고 Adam을 사용한 경우 overfitting이 생기기도 했다. 만약 beta나 lam 등의 값ㅇ르 조절하면 좋아질 가능성이 있다.
          - [ ] LabelSmoothing
        - [ ] validation dataset이 한쪽으로 편향되어있을 가능성이 크다.
          - 이거 꼭 점검 할 것 
        - [ ] 제출 모델 용 SGD 훈련
          - FromScratch : F1score가 0.5 정도로 찍기 수준이다. acc도 60% 수준
          - pretrained  : F1score 0.8 / acc 80% 정도로 훈련됨
            모델 이름<exp_pretrained_SGD2021-08-30_21_26_10>
    - [ ] 성별 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기
    - [ ] 나이 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기
    - [ ] 마스크 라벨을 예측하는 모델을 hparams tunning으로 훈련해 보기


## DataLoader 설계
  - pre-transform과 tranform을 나눠서 동작시키면 feed에서 더 빠르게 연산이 가능하다.
