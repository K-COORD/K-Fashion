import json
from PIL import Image, ImageDraw


def extract_data(one_labels):
    ID = one_labels["이미지 정보"]["이미지 식별자"]
    height = one_labels["이미지 정보"]["이미지 높이"]
    width = one_labels["이미지 정보"]["이미지 너비"]

    rects = one_labels["데이터셋 정보"]["데이터셋 상세설명"]["렉트좌표"]
    styles = one_labels["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]
    overall_style = list([one["스타일"] for one in styles["스타일"] if len(one.values()) > 0])
    if len(overall_style) == 0:
        overall_style = ["기타"]

    clothing_categories = list(rects.keys())
    assert clothing_categories == list(styles.keys())[1:]

    boxes = []
    clothing_labels = []
    style_labels = []
    for cat in clothing_categories:
        for idx, rect in enumerate(rects[cat]):
            if len(rect.values()) == 0:
                continue
            assert list(rect.keys()) == ["X좌표", "Y좌표", "가로", "세로"]
            box = list(rect.values())
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            boxes.append(box) 
            clothing_labels.append(cat)
            style_labels.append(styles[cat][idx])

    obj = {
        "ID": ID,
        "overall_style": overall_style,
        "height": height,
        "width": width,
        "boxes": boxes,
        "clothing_labels": clothing_labels,
        "style_labels": style_labels
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

        json.dump(extracted, open(f'examples/{extracted["ID"]}_extracted.json', "w"))
        print()

