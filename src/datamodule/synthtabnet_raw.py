import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any, Literal, Union


from utils import load_json_annotations, bbox_augmentation_resize, convert_html_to_otsl, extract_structure_tokens

# invalid data pairs: image_000000_1634629424.098128.png has 4 channels
INVALID_DATA = [
    {
        "dataset": "fintabnet",
        "split": "train",
        "image": "image_009379_1634631303.201671.png",
    },
    {
        "dataset": "marketing",
        "split": "train",
        "image": "image_000000_1634629424.098128.png",
    },
]


CENTER_MODE_WATERMARK_TEXTS = [
    "CONFIDENTIAL", "TOP SECRET", "CLASSIFIED", "AUTHORIZED USE ONLY",
    "PROPRIETARY INFORMATION", "STRICTLY CONFIDENTIAL", "INTERNAL USE ONLY",
    "RESTRICTED ACCESS", "FOR OFFICIAL USE ONLY", "EYES ONLY",
    "SENSITIVE MATERIAL", "DO NOT COPY", "PRIVATE PROPERTY", "COPYRIGHT © 2025",
    "TRADE SECRET", "LIMITED DISTRIBUTION", "CORPORATE CONFIDENTIAL"
]

GRID_MODE_WATERMARK_TEXTS = [
    "CONFIDENTIAL", "PRIVATE", "INTERNAL", "SECRET", "RESTRICTED",
    "OFFICIAL", "DO NOT COPY", "COPYRIGHT", "SENSITIVE", "PROTECTED"
]

class Synthtabnet(Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image", "html", "cell", "bbox", "mix"],
        split: Literal["train", "val", "test"],
        transform: transforms = None,
        json_html: Union[Path, str] = None,
        cell_limit: int = 100,
        otsl_mode: bool = False,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir) / "images"
        self.split = split
        self.label_type = label_type
        self.transform = transform
        self.cell_limit = cell_limit
        self.font_root = "/table/table_structure/unitable_IX/src/datamodule/fonts_list/*.ttf"
        self.otsl_mode = otsl_mode

        # SSP only needs image
        self.img_list = os.listdir(self.root_dir / self.split)
        if label_type != "image":
            self.image_label_pair = load_json_annotations(
                json_file_dir=Path(root_dir) / json_html, split=split
            )

    def __len__(self):
        return len(self.img_list)



    # ====== augmentation methods ======
    def apply_watermark(
        self,
        image: np.ndarray,
        opacity_range=(50, 150),
        rotation_range=(-45, 45),
        font_size_ratio=5,
        color_scheme='gray'
    ) -> np.ndarray:
        """50% 확률로 'center'/'grid' 워터마크를 OpenCV BGR 이미지에 적용."""
        mode = random.choice(['center', 'grid'])
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))

        # 폰트 경로 중 하나 랜덤 선택
        font_paths = glob(self.font_root)
        if not font_paths:
            raise FileNotFoundError(f"No fonts found at: {self.font_root}")
        font_path = random.choice(font_paths)
        font_size = min(pil_image.size) // font_size_ratio
        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(overlay)

        # 그레이 계열 or RGB 랜덤 색상
        if color_scheme == 'gray':
            color = (
                random.randint(128, 200),
                random.randint(128, 200),
                random.randint(128, 200),
                random.randint(*opacity_range)
            )
        else:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(*opacity_range)
            )

        # ===== center mode =====
        if mode == 'center':
            text = random.choice(CENTER_MODE_WATERMARK_TEXTS)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (pil_image.width - text_width) // 2
            text_y = (pil_image.height - text_height) // 2

            text_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_image)
            text_draw.text((text_x, text_y), text, font=font, fill=color)

            angle = random.uniform(*rotation_range)
            rotated_text = text_image.rotate(angle, resample=Image.BICUBIC, expand=True)

            paste_x = (overlay.width - rotated_text.width) // 2
            paste_y = (overlay.height - rotated_text.height) // 2
            overlay.paste(rotated_text, (paste_x, paste_y), rotated_text)

        # ===== grid mode =====
        elif mode == 'grid':
            text = random.choice(GRID_MODE_WATERMARK_TEXTS)
            spacing = font_size * 2
            angle = random.uniform(*rotation_range)
            for x in range(0, pil_image.width, spacing):
                for y in range(0, pil_image.height, spacing):
                    text_image = Image.new('RGBA', (spacing, spacing), (0, 0, 0, 0))
                    text_draw = ImageDraw.Draw(text_image)
                    text_draw.text((0, 0), text, font=font, fill=color)
                    rotated_text = text_image.rotate(angle, resample=Image.BICUBIC, expand=True)
                    overlay.paste(rotated_text, (x, y), rotated_text)

        # 두 이미지 합성
        watermarked = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
        return cv2.cvtColor(np.array(watermarked), cv2.COLOR_RGBA2BGR)

    def apply_black_noise(self, image: np.ndarray, amount=0.0001) -> np.ndarray:
        """일정 확률로 0(검정) 픽셀을 찍어 '노이즈'를 추가한다."""
        height, width = image.shape[:2]
        num_pepper = int(height * width * amount)
        ys = np.random.randint(0, height, num_pepper)
        xs = np.random.randint(0, width, num_pepper)
        image[ys, xs] = [0, 0, 0]
        return image

    def apply_gaussian_blur(self, image: np.ndarray, ksize=(5, 5), sigma_max=1.0) -> np.ndarray:
        """가우시안 블러로 약간 흐리게 만들기."""
        sigma = random.uniform(0.5, sigma_max)
        return cv2.GaussianBlur(image, ksize, sigma)

    def aug(self, image: np.ndarray) -> np.ndarray:
        """
        self.prob 확률로 워터마크 / 검정노이즈 / 블러 등을 적용.
        image: OpenCV(BGR) 이미지 array
        """
        aug_prob = 0.0
        if random.random() < aug_prob:
            image = self.apply_watermark(image)
        if random.random() < aug_prob:
            image = self.apply_black_noise(image)
        if random.random() < aug_prob:
            image = self.apply_gaussian_blur(image)
        return image


    def __getitem__(self, index: int) -> Any:
        if self.label_type == "image":
            img = Image.open(self.root_dir / self.split / self.img_list[index])
            if self.transform:
                sample = self.transform(img)
            return sample
        else:
            # ====== label_type != "image" ======
            obj = self.image_label_pair[index]
            img_path = self.root_dir / self.split / obj[0]
            img = Image.open(img_path)
            # if pil_img.mode == 'RGBA':
            #     pil_img = pil_img.convert('RGB')

            # # 증강 적용
            # cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # cv2_img = self.aug(cv2_img)
            # img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

            if self.label_type == "html":
                if self.transform:
                    img = self.transform(img)
                sample = dict(
                    filename=obj[0], image=img, html=obj[1]["structure"]["tokens"]
                )
                return sample
            
            elif self.label_type == "box":
                img_size = img.size
                if self.transform:
                    print(img_size, self.root_dir / self.split / obj[0])
                    img = self.transform(img)
                tgt_size = img.shape[-1]
                sample = dict(filename=obj[0], image=img)

                bboxes = [
                    entry["bbox"]
                    for entry in obj[1]["cells"]
                    if "bbox" in entry and  len(entry['tokens']) > 0
                    and entry["bbox"][0] < entry["bbox"][2]
                    and entry["bbox"][1] < entry["bbox"][3]
                ]

                bboxes[:] = [
                    i
                    for entry in bboxes
                    for i in bbox_augmentation_resize(entry, img_size, tgt_size)
                ]

                sample["bbox"] = bboxes

                return sample

            elif self.label_type == "mix":
                img_size = img.size
                if self.transform:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img = self.transform(img)
                tgt_size = img.shape[-1]
                bboxes = [
                    entry["bbox"]
                    for entry in obj[1]["cells"]
                    if "bbox" in entry and len(entry['tokens']) > 0 and ''.join(entry['tokens']).strip() != ''
                    and entry["bbox"][0] < entry["bbox"][2]
                    and entry["bbox"][1] < entry["bbox"][3]
                ]

                bboxes[:] = [
                    i
                    for entry in bboxes
                    for i in bbox_augmentation_resize(entry, img_size, tgt_size)
                ]

                ori_html = obj[1]["structure"]["tokens"]
                ori_html = extract_structure_tokens(ori_html)
                
                if '<thead>' in ori_html:
                    ori_html.remove('<thead>')
                if '</thead>' in ori_html:
                    ori_html.remove('</thead>')
                if '<tbody>' in ori_html:
                    ori_html.remove('<tbody>')
                if '</tbody>' in ori_html:
                    ori_html.remove('</tbody>')

                if self.otsl_mode:
                    try:
                        otsl, otsl_with_tags = convert_html_to_otsl("".join(ori_html))
                        sample = dict(filename=obj[0], image=img, html=otsl)
                    except Exception as e:
                        print(img_path)
                        print(f"Error converting HTML to OTSL: {e}")
                        print("".join(ori_html))
                        raise e
                    # print(obj[0])
                    # print("otsl", otsl)
                    # print("ori_html", "".join(ori_html), '\n\n')
                else:
                    sample = dict(filename=obj[0], image=img, html=ori_html)

                # sample = dict(filename=obj[0], image=img, html=ori_html)
                sample['bbox'] = bboxes

                return sample