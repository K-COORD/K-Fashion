# K-Fashion 이미지 | AI 허브 공개 데이터

본 레포를 통해 [2021년 인공지능 데이터 활용 경진대회: "너의 AIdea가 보여"](http://aihub-competition.or.kr/aidea) 공모전 본선 진출팀 K-COORD팀이 코드 베이스 중 [K-Fashion 이미지](https://aihub.or.kr/aidata/7988) 데이터 활용 코드를 공개함.

K-COORD팀은 Fashion Retrieval를 위한 패션 상품 매칭 모델을 개발하고 Fashion Analysis를 위한 패션 상품 속성 분석을 하는 Attribute R-CNN을 개발하였는데 본 레포는 후자와 관련된 코드를 공개함.


### 이 레포가 제공하는 것:
- 공모전에 사용한 모델 두개 중 Attribute R-CNN 모델 설계 및 학습 코드 공개
- K-Fashion 이미지 데이터를 Pytorch Dataloader로 학습에 사용 할 수 있도록 변환하는 코드
- 추후 사용자를 위한 수정 방식 제공: 본 레포는 K-COORD팀의 목적으로 특정한 방식으로 데이터 활용과 모델을 구현하였지만 쉽게 다른 정보도 활용 할 수 있도록 유연하게 만들어짐.

### 목차:
1. 데이터셋 정보 (예시 원천 데이터 / 라벨링)
2. 코드 설명 / 예시
   * 데이터 다운로드
   * 사용 예시 (데이터 -> dataloader 변환)
   * 설계 설명
   * 환경 설정

### 개발 환경

```
numpy==1.21.2
Pillow==8.4.0
torch==1.7.1+cu101
torchvision==0.8.2+cu101
```

## I. 데이터셋 정보

| 데이터셋명         | K-Fashion 이미지                                                                                                                                                         |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 버전               | 1.1                                                                                                                                                                      |
| 소개               | 패션 영역과 속성, 스타일 정보를 인식 및 도출할 수 있도록 학습용 이미지 데이터셋을 구축하고, 한국형 패션 인지 및 트렌드 파악과 AI기반 시각지능 기술 및 서비스 개발에 활용 |
| 저작권 및 이용정책 | 본 데이터는 과학기술정보통신부가 주관하고 한국지능정보사회진흥원이 지원하는 '인공지능 학습용 데이터 구축사업'으로 구축된 데이터입니다.   [데이터 이용정책 상세보기]      |
| 구축기관           | 오피니언라이브                                                                                                                                                           |
| 가공기관           | 이화여대 산학협력단                                                                                                                                                      |
| 검수기관           | 웨얼리, 이화여대 산학협력단, 오피니언라이브, 한국패션산업연구원                                                                                                          |


### 예시 원천 데이터

Example 1             |  Example 2
:-------------------------:|:-------------------------:
![alt text](https://github.com/codeandproduce/K-Fashion-Dataset/blob/master/examples/100026.jpg?raw=true)  |  ![alt text](https://github.com/codeandproduce/K-Fashion-Dataset/blob/master/examples/1070263.jpg?raw=true)



### 예시 원천 데이터 + Bounding Box 라벨링 시각화

Example 1             |  Example 2
:-------------------------:|:-------------------------:
![alt text](https://github.com/codeandproduce/K-Fashion-Dataset/blob/master/examples/100026_box.jpg?raw=true)  |  ![alt text](https://github.com/codeandproduce/K-Fashion-Dataset/blob/master/examples/1070263_box.jpg?raw=true))



### 예시 라벨링 / 어노테이션

<details><summary>과한 길이로 인하여 숨겨짐</summary>
<p>

```json
{
    "이미지 정보": {
        "이미지 식별자": 353924,
        "이미지 높이": 1066,
        "이미지 파일명": "u_154892233694411000_400400624.jpg",
        "이미지 너비": 800
    },
    "데이터셋 정보": {
        "파일 생성일자": "2020-09-14 05:16:46",
        "데이터셋 상세설명": {
            "렉트좌표": {
                "아우터": [
                    {
                        "X좌표": 69.5,
                        "Y좌표": 0.499625,
                        "가로": 641,
                        "세로": 1043
                    }
                ],
                "하의": [
                    {}
                ],
                "원피스": [
                    {}
                ],
                "상의": [
                    {}
                ]
            },
            "폴리곤좌표": {
                "아우터": [
                    {
                        "X좌표39": 213.0,
                        "X좌표38": 284.0,
                        "X좌표37": 344.0,
                        "X좌표36": 396.0,
                        "X좌표35": 524.0,
                        "X좌표34": 564.0,
                        "X좌표33": 602.0,
                        "X좌표32": 606.0,
                        "X좌표31": 605.0,
                        "X좌표30": 608.0,
                        "X좌표49": 72.0,
                        "X좌표48": 70.0,
                        "X좌표47": 89.0,
                        "X좌표46": 122.0,
                        "X좌표45": 140.0,
                        "X좌표44": 183.0,
                        "X좌표43": 172.0,
                        "X좌표42": 172.0,
                        "X좌표41": 168.0,
                        "X좌표40": 172.0,
                        "Y좌표26": 783.106,
                        "Y좌표25": 756.116,
                        "Y좌표28": 795.102,
                        "Y좌표27": 802.099,
                        "Y좌표22": 552.193,
                        "Y좌표21": 471.823,
                        "Y좌표24": 708.134,
                        "Y좌표23": 650.156,
                        "Y좌표20": 298.888,
                        "X좌표58": 300.0,
                        "X좌표57": 268.0,
                        "X좌표56": 214.0,
                        "X좌표55": 176.0,
                        "X좌표54": 169.0,
                        "X좌표53": 124.0,
                        "X좌표52": 96.0,
                        "X좌표51": 80.0,
                        "X좌표50": 79.0,
                        "Y좌표29": 779.108,
                        "Y좌표15": 69.9737,
                        "Y좌표14": 60.9771,
                        "Y좌표17": 108.959,
                        "Y좌표16": 78.9704,
                        "Y좌표11": 24.9906,
                        "Y좌표10": 7.997,
                        "Y좌표13": 45.9827,
                        "Y좌표12": 28.9891,
                        "Y좌표19": 231.913,
                        "Y좌표18": 136.949,
                        "Y좌표48": 760.115,
                        "Y좌표47": 779.108,
                        "Y좌표49": 730.126,
                        "Y좌표44": 689.141,
                        "Y좌표43": 769.111,
                        "Y좌표46": 788.104,
                        "Y좌표45": 798.1,
                        "Y좌표40": 1011.02,
                        "Y좌표42": 863.076,
                        "Y좌표41": 937.048,
                        "Y좌표37": 1035.01,
                        "Y좌표36": 1034.01,
                        "X좌표8": 470.0,
                        "Y좌표39": 1033.01,
                        "X좌표9": 498.0,
                        "Y좌표38": 1044.01,
                        "Y좌표33": 984.031,
                        "Y좌표32": 954.042,
                        "Y좌표35": 1044.01,
                        "Y좌표34": 1022.02,
                        "Y좌표31": 898.063,
                        "Y좌표30": 819.093,
                        "X좌표2": 328.0,
                        "X좌표3": 370.0,
                        "X좌표1": 304.0,
                        "X좌표6": 437.0,
                        "X좌표7": 457.0,
                        "X좌표4": 377.0,
                        "X좌표5": 398.0,
                        "Y좌표9": 0.999625,
                        "X좌표19": 672.0,
                        "X좌표18": 647.0,
                        "X좌표17": 631.0,
                        "X좌표16": 595.0,
                        "X좌표15": 569.0,
                        "X좌표14": 558.0,
                        "X좌표13": 558.0,
                        "X좌표12": 544.0,
                        "X좌표11": 523.0,
                        "Y좌표4": 39.985,
                        "X좌표10": 524.0,
                        "Y좌표3": 12.9951,
                        "Y좌표2": 0.999625,
                        "Y좌표1": 6.99737,
                        "Y좌표8": 21.9917,
                        "Y좌표7": 44.9831,
                        "Y좌표6": 55.979,
                        "Y좌표5": 50.9809,
                        "Y좌표58": 3.9985,
                        "Y좌표55": 110.958,
                        "Y좌표54": 175.934,
                        "Y좌표57": 47.982,
                        "Y좌표56": 67.9745,
                        "Y좌표51": 628.164,
                        "Y좌표50": 694.139,
                        "Y좌표53": 406.248,
                        "Y좌표52": 520.205,
                        "X좌표29": 601.0,
                        "X좌표28": 606.0,
                        "X좌표27": 684.0,
                        "X좌표26": 711.0,
                        "X좌표25": 711.0,
                        "X좌표24": 708.0,
                        "X좌표23": 708.0,
                        "X좌표22": 700.0,
                        "X좌표21": 691.0,
                        "X좌표20": 680.0
                    }
                ],
                "하의": [
                    {}
                ],
                "원피스": [
                    {}
                ],
                "상의": [
                    {}
                ]
            },
            "라벨링": {
                "스타일": [
                    {
                        "스타일": "밀리터리"
                    }
                ],
                "아우터": [
                    {
                        "기장": "하프",
                        "색상": "카키",
                        "카테고리": "재킷",
                        "디테일": [
                            "포켓",
                            "셔링"
                        ],
                        "소매기장": "긴팔",
                        "소재": [
                            "우븐"
                        ],
                        "프린트": [
                            "무지"
                        ],
                        "넥라인": "후드",
                        "핏": "루즈"
                    }
                ],
                "하의": [
                    {}
                ],
                "원피스": [
                    {}
                ],
                "상의": [
                    {}
                ]
            }
        },
        "파일 번호": 353924,
        "파일 이름": "u_154892233694411000_400400624.jpg"
    }
}
```

</p>
</details>


## II. 코드 설명 / 예시

### 사용 방법

[K-Fashion 공개 데이터 사이트](https://aihub.or.kr/aidata/7988)에서 승인을 받고 디렉토리에 압축을 푼다.

```
k-coord
│   README.md
│   train_attributercnn.py
└───data/
│   │   Training/
│   │   Validation/
└───models
    │   attributercnn.py
...
```

`dataset.py`를 활용해 데이터를 로딩하면 `torch.utils.data.DataLoader`객채가 변환된다.

```python
from dataset import load_data
    
dataloader = load_data(train=False, batch_size=2, num_workers=0, data_root="./data")
for batch in dataloader:
    images, targets, idxs = batch

    for idx, (image, target) in enumerate(zip(images, targets)):
        # print("box", target["boxes"][0])
        saveimg_bbox(image, f"examples/{idx}_bboxed.jpg", target["boxes"][0])
    print(batch)
```

위 코드의 출력값은 아래와 같다:

```python
(tensor([[[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.9176, 0.9451, 0.9255,  ..., 0.5255, 0.5373, 0.5608],
          [0.9216, 0.9569, 0.9412,  ..., 0.5294, 0.5412, 0.5686],
          [0.9412, 0.9765, 0.9686,  ..., 0.5333, 0.5451, 0.5686]]]]),           
[{'boxes': tensor([[309.5000, 220.5000, 595.5000, 922.5000]]), 'labels': [13], 'attributes': {'material': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]), 'fit': tensor([2]), 'collar': tensor([0]), 'neckline': tensor([0]), 'shirt_sleeve': tensor([6]), 'sleeve': tensor([8])}}, {'boxes': tensor([[279.5000, 358.5000, 508.5000, 701.5000],
        [315.5000, 603.5000, 459.5000, 955.5000]]), 'labels': [14, 7], 'attributes': {'material': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), 'fit': tensor([2, 1]), 'collar': tensor([0, 0]), 'neckline': tensor([0, 0]), 'shirt_sleeve': tensor([6, 0]), 'sleeve': tensor([6, 7])}}], [416417, 940910])
```

각 속성마다 one-hot-encoding 라벨로 변환되는 것이다. 

### 설명

K-Fashion 이미지 데이터는 결국에는 이런 구조로 되어 있는데:
```
# one image
one_image_data = {
    "clothing_item_1": { 렉트 좌표, 세부 속성 },
    "clothing_item_2": { 렉트 좌표, 세부 속성 },
}
```

`소매기장` 같은 세부 속성은 

```SHIRT_SLEEVES = ["없음", "민소매", "반팔", "캡", "7부소매", "긴팔"]```

아이템이 상의일때만, 이 중 하나에만 해당 할 수 있지만,

`소재` 같은 세부속성은

```
MATERIAL_CATEGORIES = ["패딩", "퍼", "무스탕", "스웨이드", "앙고라", "코듀로이", "시퀸/글리터", "데님", "저지", "트위드", "벨벳", "비닐/PVC", "울/캐시미어", "합성섬유", "헤어 니트", "니트", "레이스", "린넨", "메시", "플리스", "네오프렌", "실크", "스판덱스", "자카드", "가죽", "면", "시폰", "우븐"]
```

아이템이 무슨 종류의 옷이든 제한이 없고, 이 중 다수에 해당 할 수 있다.

그래서 이런 세부속성을 추론하는 모델은 single-label classification과 동시에 multi-label classification도 가능해야 한다. 손실 함수에도 이를 반영해야 한다.

K-COORD팀은 어느 세부속성을 추론 할지 사전에 골라서 학습하였지만 어떤 세부속성을 골라도 이 코드를 활용 할 수 있게끔 코드를 작성하였다:

```python
# train_attributercnn.py > train(args)

model.load_saved_matchrcnn(sd, new_num_classes=len(CLOTHING_CATEGORIES), attribute_dict={
    "material": ("multi", len(MATERIAL_CATEGORIES) + 1),
    "sleeve": ("single", len(SLEEVE_CATEGORIES) + 1),
    "shirt_sleeve": ("single", len(SHIRT_SLEEVES) + 1),
    "neckline": ("single", len(NECKLINE_CATEGORIES) + 1),
    "collar": ("single", len(COLLAR_CATEGORIES) + 1),
    "fit": ("single", len(FIT_CATEGORIES) + 1)
})

...

# attributercnn.py > fastrcnn_loss_with_attr(...)

attribute_loss_collect = []
for key in attributes_dict.keys():
    gt_attr = attributes_dict[key]
    # print("gt_attr", gt_attr.size())
    if len(gt_attr.size()) == 2:
        # multi-label loss
        sig = F.sigmoid(attributes_logits[key])
        one_attr_loss = F.binary_cross_entropy(sig, gt_attr)
        attribute_loss_collect.append(one_attr_loss)
    elif len(gt_attr.size()) == 1:
        # single-label loss
        one_attr_loss = F.cross_entropy(attributes_logits[key], gt_attr)
        attribute_loss_collect.append(one_attr_loss)

attribute_loss = torch.mean(torch.stack(attribute_loss_collect))
```

