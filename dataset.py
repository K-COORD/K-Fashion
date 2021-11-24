import glob
import json
from itertools import chain

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from torchvision import transforms


from PIL import Image, ImageDraw,ImageFile
# from PIL import 
ImageFile.LOAD_TRUNCATED_IMAGES = True


def saveimg_bbox(image, name, box):
    if not isinstance(box, list):
        box = box.detach().cpu().numpy()
    np_img = np.uint8(image.cpu().numpy()*255)
    np_img = np_img.transpose(1,2,0)
    # np_img = np.flip(np_img)
    img = Image.fromarray(np_img, mode="RGB")
    
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(box, outline="red")
    img.save(name)

pil_to_tensor = transforms.ToTensor()


# "카테코리" single label but make into multilabel (explained below)
CLOTHING_CATEGORIES = {
    "상의": ["탑", "블라우스", "티셔츠", "니트웨어", "셔츠", "브라탑", "후드티"],
    "하의": ["청바지", "팬츠", "스커트", "래깅스", "조거팬츠"],
    "아우터": ["코트", "재킷", "점퍼", "패딩", "베스트", "가디건", "짚업"],
    "원피스": ["드레스", "점프수트"]
}
# including the keys because some data doesnt have the specific clothing item tag
# MULTILABEL
CLOTHING_CATEGORIES = list(chain(*list(CLOTHING_CATEGORIES.values()))) + list(CLOTHING_CATEGORIES.keys()) 
# print("CLOTHING_CATEGORIES", CLOTHING_CATEGORIES)
# MULTILABEL
MATERIAL_CATEGORIES = ["패딩", "퍼", "무스탕", "스웨이드", "앙고라", "코듀로이", "시퀸/글리터", "데님", "저지", "트위드", "벨벳", "비닐/PVC", "울/캐시미어", "합성섬유", "헤어 니트", "니트", "레이스", "린넨", "메시", "플리스", "네오프렌", "실크", "스판덱스", "자카드", "가죽", "면", "시폰", "우븐"]

STYLE_CATEGORIES = {
    "클래식": ["클래식", "프레피"],
    "매니시": ["매니시", "톰보이"],
    "엘레강스": ["엘레강스", "소피스케이티드", "글래머러스"],
    "에스닉": ["에스닉", "히피", "오리엔탈"],
    "모던": ["모던", "미니멀"],
    "내추럴": ["내추럴", "컨트리", "리조트"],
    "로맨틱": ["로맨틱", "섹시"],
    "스포티": ["스포티", "애슬레져", "밀리터리"],
    "문화": ["뉴트로", "힙합", "키티/키덜트", "맥시멈", "펑크/로커"],
    "캐주얼": ["캐주얼", "놈코어"]
}
# "기장" - single label
SLEEVE_CATEGORIES = {
    "상의": ["크롭", "노멀", "롱"],
    "하의": ["미니", "니렝스", "미디", "발목", "맥시"],
    "아우터": ["크롭", "노멀", "하프", "롱", "맥시"],
    "원피스": ["미니", "니렝스", "미디", "발목", "맥시"],
}
SLEEVE_CATEGORIES = list(set(chain(*list(SLEEVE_CATEGORIES.values()))))

# "소매기장" - single label
SHIRT_SLEEVES = ["없음", "민소매", "반팔", "캡", "7부소매", "긴팔"]


# "넥라인" - single label
NECKLINE_CATEGORIES = ["라운드넥", "유넥", "브이넥", "홀토넥", "오프숄더", "원 숄더", "스퀘어넥", "노카라", "후드", "터틀넥", "보트넥", "스위트하트"]

# "옷깃" - single label
COLLAR_CATEGORIES = ["셔츠칼라", "보우칼라", "세일러칼라", "숄칼라", "폴로칼라", "피터팬칼라", "너치드칼라", "차이나칼라", "테일러칼라", "밴드칼라"]

# "핏" - single label
FIT_CATEGORIES = ["노멀", "루즈", "오버사이즈", "스키니", "와이드", "타이트", "벨보텀"]


def box_area(box):
    width = box[2] - box[0]
    height = box[3] - box[1]

    return width * height

def get_transform(train):
    selected_transforms = []
    selected_transforms.append(transforms.ToTensor())
    if train:
        selected_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(selected_transforms)


def extract_data(one_labels):
    # print(json.dumps(one_labels, ensure_ascii=False,indent=4))

    # print("one_labels", json.dumps(one_labels, ensure_ascii=False,indent=4, ))
    ID = one_labels["이미지 정보"]["이미지 식별자"]
    height = one_labels["이미지 정보"]["이미지 높이"]
    width = one_labels["이미지 정보"]["이미지 너비"]

    all_rects = one_labels["데이터셋 정보"]["데이터셋 상세설명"]["렉트좌표"]
    styles = one_labels["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]
    overall_style = list([one["스타일"] for one in styles["스타일"] if len(one.values()) > 0])
    if len(overall_style) == 0:
        overall_style = ["기타"]

    all_clothing_categories = list(all_rects.keys())
    
    rects = []
    clothing_categories = []
    for rect_idx, rect in enumerate(all_rects.values()):
        for one_rect in rect:
            # for some reason the rects are given as array per clothing
            # and some of these are broken (has 0 area)
            if len(one_rect.keys()) > 0:
                assert list(one_rect.keys()) == ["X좌표", "Y좌표", "가로", "세로"]
                box = list(one_rect.values())
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]

                area = box_area(box)

                if area > 1:        
                    rects.append(box)
                    clothing_categories.append(all_clothing_categories[rect_idx])
                break

    boxes = [] 
    # clothing_labels = []
    # style_labels = []
    
    
    all_attributes = {
        "material": [], # "소재"
        "fit": [], # "핏"
        "collar": [],  # "칼라" 
        "neckline": [], # "넥라인"
        "shirt_sleeve": [], # "소매기장"
        "sleeve": [], # "기장"
        "clothing_categories": [] # "카테고리"
    }

    if len(clothing_categories) == 0:
        return None

    for clothing_idx, cat in enumerate(clothing_categories):
        rect = rects[clothing_idx]
        boxes.append(rect) 

        # print("styles[cat]", styles[cat][])
        one_clothing_attributes = styles[cat][0]
        # clothing_labels.append(cat)
        for attribute_name, attribute_value in one_clothing_attributes.items():
            if attribute_name == "카테고리":
                all_attributes["clothing_categories"].append(attribute_value)
            elif attribute_name == "기장":
                all_attributes["sleeve"].append(attribute_value)
            elif attribute_name == "소매기장":
                all_attributes["shirt_sleeve"].append(attribute_value)
            elif attribute_name == "핏":
                all_attributes["fit"].append(attribute_value)
            elif attribute_name == "소재":
                all_attributes["material"].append(attribute_value)
            elif attribute_name == "칼라":
                all_attributes["collar"].append(attribute_value)
    
        for key, value in all_attributes.items():
            if len(value) <= clothing_idx:
                # print("append")
                if key == "카테고리":
                    all_attributes[key].append(clothing_categories[clothing_idx])
                else:
                    all_attributes[key].append(0)
    obj = {
        "ID": ID,
        "overall_style": overall_style,
        "height": height,
        "width": width,
        "boxes": boxes,
        "labels": clothing_categories,
        "attributes": all_attributes
    }
    return obj


def extract_to_onehot(extracted_obj):
    attr = extracted_obj["attributes"]

    attribute_dict = {}

    multi_label_keys = {
        "material": MATERIAL_CATEGORIES,
    }
    single_label_keys = {
        "fit" : FIT_CATEGORIES,
        "collar": COLLAR_CATEGORIES, 
        "neckline": NECKLINE_CATEGORIES, 
        "shirt_sleeve": SHIRT_SLEEVES, 
        "sleeve": SLEEVE_CATEGORIES,
    }

    # need to specially handle "clothing_categories"
    clothing = attr["clothing_categories"]
    for idx, label in enumerate(clothing):
        clothing_super = extracted_obj["labels"][idx]
        if label == 0:
            clothing[idx] = clothing_super
        else:
            # clothing[idx] = [label, clothing_super]
            clothing[idx] = label
        
    clothing = [CLOTHING_CATEGORIES.index(one) for one in clothing]
    extracted_obj["labels"] = clothing

    for multi_label_key, corres_lookup in multi_label_keys.items(): 
        multi_attr = attr[multi_label_key]
        for idx, one_box_attr in enumerate(multi_attr):
            new_arr = [0] * (len(corres_lookup) + 1) 
            if one_box_attr == 0:
                new_arr[0] = 1
            else:
                for one_hot_idx in one_box_attr:
                    if one_hot_idx != 0:
                        try:
                            one_hot_idx = corres_lookup.index(one_hot_idx) + 1 # starts at 1, since 0 is no match
                        except:
                            print(one_hot_idx, multi_label_key, corres_lookup)
                            raise Exception()
                    new_arr[one_hot_idx] = 1
                    # one_hot_array = [0] * (len(corres_lookup) + 1)
                    # one_hot_array[one_hot_idx] = 1
                    # # print("one_hot_array", one_hot_array)
                    # new_arr.append(one_hot_array)
            
            multi_attr[idx] = new_arr
        # print("multi_attr", multi_attr)
       
        attribute_dict[multi_label_key] = torch.tensor(multi_attr, dtype=torch.float32)
    
    for single_label_key, corres_lookup in single_label_keys.items():
        single_attr = attr[single_label_key]
        for idx, one_box_attr in enumerate(single_attr):
            if one_box_attr != 0:
                if one_box_attr == "노말":
                    one_box_attr = "노멀"
                try:
                    one_box_attr = corres_lookup.index(one_box_attr) + 1
                except:
                    print(one_box_attr, single_label_key, corres_lookup)
                    raise Exception()
            single_attr[idx] = one_box_attr
        attribute_dict[single_label_key] = torch.tensor(single_attr, dtype=torch.int64)
    
    extracted_obj["attributes"] = attribute_dict

    return extracted_obj


class KFashionDataset(Dataset):
    def __init__(self, train, data_root, selected_transforms=None):
        
        data_root = data_root + "/" if not data_root.endswith("/") else data_root
        if train:
            json_files = list(glob.glob(data_root + "Training/*/*.json"))
        else:
            json_files = list(glob.glob(data_root + "Validation/*/*.json"))

        images = []
        dataset = []
        for jfile in json_files:
            jopen = open(jfile, "r")
            full_data = json.load(jopen)
            jopen.close()

            extracted = extract_data(full_data)
            if extracted is not None:
                processed = extract_to_onehot(extracted)

                ID = extracted["ID"]
                if train:
                    image_path = list(glob.glob(data_root + f"Training/*/{ID}.jpg"))
                else:
                    image_path = list(glob.glob(data_root + f"Validation/*/{ID}.jpg"))
                assert len(image_path) == 1
                image_path = image_path[0]

                images.append(image_path)
                dataset.append(processed)
                # try:

                #     one_image = pil_to_tensor(Image.open(image_path))
                #     images.append(one_image)
                #     one_image.close()
                    
                # except:
                #     print("BROKEN IMAGE:", image_path)
                #     pass
                

        self.images = images
        self.dataset = dataset
        if selected_transforms is None:
            self.transforms = get_transform(train)
        else: 
            self.transforms = selected_transforms
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):        
        image = self.images[idx]
        pil_image = Image.open(image)
        image = self.transforms(pil_image)
        pil_image.close()
        return (image, self.dataset[idx])

def collate_fn(batch):
    images = [one[0] for one in batch]
    targets = [one[1] for one in batch]
    boxes = [one["boxes"] for one in targets] 
    indices = []


    height = max([one.size(1) for one in images])
    width = max([one.size(2) for one in images])

    new_images = []
    for image_idx, image in enumerate(images):
        one_boxes = boxes[image_idx]

        c, cur_height, cur_width = image.size()
        # if cur_width > cur_height:
        #     image = image.transpose(1, 2)
        #     for i in range(len(one_boxes)):
        #         one_box = one_boxes[i]
        #         one_boxes[i] = [one_box[1], cur_width-one_box[2], one_box[3], cur_height-one_box[0]]
        #     print("rot image", image.size())
        #     print("rot one_boxes", one_boxes)
        #     print()
        #     c, cur_height, cur_width = image.size()

        one_boxes = torch.tensor(one_boxes)
        
        height_pad = height - cur_height
        width_pad = width - cur_width

        width_diff = 0 if (width_pad / 2) % 1 == 0 else 1
        height_diff = 0 if (height_pad / 2) % 1 == 0 else 1

        padding = (width_pad // 2 + width_diff, height_pad // 2 + height_diff, width_pad // 2, height_pad // 2)
        image = TF.pad(image, padding=padding, padding_mode="constant", fill=0)

        box_padding = (width_pad // 2 + width_diff, height_pad // 2 + height_diff, width_pad // 2 + width_diff, height_pad // 2 + height_diff)
        one_boxes = torch.add(one_boxes, torch.tensor(box_padding))
        
        targets[image_idx]["boxes"] = one_boxes

        indices.append(targets[image_idx]["ID"])
        new_images.append(image)

        del targets[image_idx]["ID"]
        del targets[image_idx]["overall_style"]
        del targets[image_idx]["height"]
        del targets[image_idx]["width"]
    images = torch.stack(new_images, dim=0)
    return (images, targets, indices)

def load_data(train, batch_size=16, num_workers=0, data_root="./"):
    dataset = KFashionDataset(train, data_root=data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

# dataloader = load_data(train=False, batch_size=2, num_workers=0, data_root="./")
# for batch in dataloader:
#     images, targets, idxs = batch

#     for idx, (image, target) in enumerate(zip(images, targets)):
#         # print("box", target["boxes"][0])
#         saveimg_bbox(image, f"examples/{idx}_bboxed.jpg", target["boxes"][0])
#     print(batch)

#     # print(json.dumps(targets, ensure_ascii=False,indent=4, ))
#     # print(batch)
#     break



            
