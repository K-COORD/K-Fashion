import json
from PIL import Image, ImageDraw


def extract_data(one_labels):
    print("one_labels", json.dumps(one_labels, ensure_ascii=False,
            indent=4, ))
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
        if len(rect[0].keys()) > 0:
            rects.append(rect)
            clothing_categories.append(all_clothing_categories[rect_idx])
    boxes = [] 
    # clothing_labels = []
    # style_labels = []
    
    
    all_attributes = {
        "material_categories": [], # "소재"
        "fit_categories": [], # "핏"
        "collar_categories": [],  # "칼라" 
        "neckline_categories": [], # "넥라인"
        "shirt_sleeve_categories": [], # "소매기장"
        "sleeve_categories": [], # "기장"
        "clothing_categories": [] # "카테고리"
    }
    for clothing_idx, cat in enumerate(clothing_categories):
        rect = rects[clothing_idx][0]
        print("rect", rect)
        assert list(rect.keys()) == ["X좌표", "Y좌표", "가로", "세로"]
        box = list(rect.values())
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        boxes.append(box) 

        # print("styles[cat]", styles[cat][])
        one_clothing_attributes = styles[cat][0]
        # clothing_labels.append(cat)
        for attribute_name, attribute_value in one_clothing_attributes.items():
            print("attribute_name", attribute_name)
            if attribute_name == "카테고리":
                all_attributes["clothing_categories"].append(attribute_value)
            elif attribute_name == "기장":
                all_attributes["sleeve_categories"].append(attribute_value)
            elif attribute_name == "소매기장":
                all_attributes["shirt_sleeve_categories"].append(attribute_value)
            elif attribute_name == "핏":
                all_attributes["fit_categories"].append(attribute_value)
            elif attribute_name == "소재":
                all_attributes["material_categories"].append(attribute_value)
            elif attribute_name == "옷깃":
                all_attributes["collar_categories"].append(attribute_value)
    
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


def saveimg_bbox(image, name, boxes):
    img = Image.open(image)
    img1 = ImageDraw.Draw(img)  
    for box in boxes:
        img1.rectangle(box, outline="red")
    img.save(name)


if __name__ == "__main__":
    JSON_FILES = ["examples/100026.json", "examples/1070263.json", "examples/1092253.json"]
    for jfile in JSON_FILES:
        full_data = json.load(open(jfile, "r"))
        extracted = extract_data(full_data)
        print(extracted)

        img_file = f'examples/{extracted["ID"]}.jpg'
        saveimg_bbox(img_file, f'examples/{extracted["ID"]}_box.jpg', extracted["boxes"])

        json.dump(
            extracted, 
            open(f'examples/{extracted["ID"]}_extracted.json', "w", encoding='utf8'), 
            ensure_ascii=False,
            indent=4, 
        )
        print()

