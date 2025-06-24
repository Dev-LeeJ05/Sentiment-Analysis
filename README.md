## 한국어 다중 감성 분석 모델
한국어 문장을 입력하면 7가지의 다중 감석 분석이 되는 모델입니다.
## DataSets
``datasets/`` 내부에 위치하고 있어야합니다.
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978)  ``corpus_train_1.xlsx`` ``corpus_train_2.xlsx``
- [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=270) ``single_train.xlsx``
- [한국어 감정 정보가 포함된 연속성 대화 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271) ``dialogue_train``   


## Model
- **Tokenization** : ``KoELECTRA의 Tokenization``기법과 유사한 Tokenization을 구현하였습니다.
- ``LSTM``을 사용하여 모델을 구축하였습니다.

## About Project
[Sentiment Analysis Project - Notion](https://www.notion.so/Sentiment-Analysis-217d2848aeaf80cca2dfdca6d9bfa089)
## Reference
[Korean-Sentiments-Classification](https://github.com/JH-lee95/Korean-Sentiments-Classification)   
[KoELECTRA](https://github.com/monologg/KoELECTRA)   
