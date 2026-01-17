import os
import random
import shutil
from glob import glob
import yaml

def split_valid(base: str, ratio: float = 0.2, seed: int = 0):
    train_img = os.path.join(base, "train/images")
    train_lbl = os.path.join(base, "train/labels")
    valid_img = os.path.join(base, "valid/images")
    valid_lbl = os.path.join(base, "valid/labels")

    os.makedirs(valid_img, exist_ok=True)
    os.makedirs(valid_lbl, exist_ok=True)

    imgs = glob(train_img + "/*.jpg") + glob(train_img + "/*.png") + glob(train_img + "/*.jpeg")
    print("총 train 이미지 수:", len(imgs))

    random.seed(seed)
    random.shuffle(imgs)

    n_valid = int(len(imgs) * ratio)
    valid_imgs = imgs[:n_valid]
    print("valid로 이동할 이미지 수:", n_valid)

    for img_path in valid_imgs:
        fn = os.path.basename(img_path)
        stem = os.path.splitext(fn)[0]
        lbl_path = os.path.join(train_lbl, stem + ".txt")

        shutil.move(img_path, os.path.join(valid_img, fn))
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(valid_lbl, stem + ".txt"))

    print("✅ valid 분리 완료")

def update_yaml(yaml_path: str, base: str):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = base
    data["train"] = "train/images"
    data["val"] = "valid/images"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("✅ data.yaml 업데이트 완료:", yaml_path)
    print(data)

def main():
    base = os.getenv("DATASET_DIR", "/content/Hard-Hat-Workers-1")
    ratio = float(os.getenv("VALID_RATIO", "0.2"))

    split_valid(base, ratio=ratio)
    update_yaml(os.path.join(base, "data.yaml"), base)

if __name__ == "__main__":
    main()
