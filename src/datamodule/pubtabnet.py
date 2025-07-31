import os
import random
import cv2
import numpy as np
import lmdb
import pickle
import json
from glob import glob
from pathlib import Path
from typing import Any, Literal, Union
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


from utils import (
    load_json_annotations,
    bbox_augmentation_resize,
    convert_html_to_otsl,
    extract_structure_tokens,
)

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


class PubTabNet(Dataset):
    """Load PubTabNet for different training purposes with optional augmentation."""

    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image", "html", "cell", "bbox", "mix"],
        split: Literal["train", "val"],
        transform: transforms = None,
        json_html: Union[Path, str] = None,  # e.g. "annotations.jsonl"
        cell_limit: int = 150,
        otsl_mode: bool = False,
        aug: bool = False,
    ) -> None:
        super().__init__()
        root_dir = Path(root_dir)
        self.root_dir   = root_dir / split
        self.split      = split
        self.label_type = label_type
        self.transform  = transform
        self.cell_limit = cell_limit
        self.otsl_mode  = otsl_mode
        self.font_root  = "/table/table_structure/unitable_mvp3.0/src/datamodule/fonts_list/*.ttf"
        self.font_paths= glob(self.font_root)
        self._font_cache = {
            size: ImageFont.truetype(random.choice(self.font_paths), size)
            for size in [20, 32, 48]  # 자주 쓰일 크기만
        }
        # 2) 워터마크 템플릿 캐시
        self._wm_cache = {}  # 키: (text, font_size, color, angle), 값: RGBA ndarray

        self.aug_ = aug
        # image-only 모드면 이미지 파일 리스트만 로드
        if self.label_type == "image":
            self.img_list = sorted(self.root_dir.iterdir())
            return

        # unified LMDB 파일 경로: json_html의 stem 뒤 .lmdb
        lmdb_path = root_dir / Path(json_html).with_suffix(".lmdb").name
        print(f"LBDB : {lmdb_path.exists()}")
        if lmdb_path.exists():
            
            # --- LMDB 모드 ---
            self.use_lmdb = True
            self.env      = lmdb.open(
                str(lmdb_path),
                subdir=lmdb_path.is_dir(),
                readonly=True,
                lock=False,
                readahead=False,
                map_size=0,
            )
            self.txn      = self.env.begin()
            # split 분할용 키 목록 수집
            self.keys = []
            with self.txn.cursor() as cursor:
                for key, val in cursor:
                    rec = pickle.loads(val)
                    if rec.get("split") == split:
                        self.keys.append(key)
            self.n_entries = len(self.keys)
        else:
            # --- JSONL 메모리 로드 모드 ---
            self.use_lmdb = False
            jsonl_path = root_dir / json_html
            self.image_label_pair = load_json_annotations(
                json_file_dir=jsonl_path, split=split
            )

    def __len__(self):
        if self.label_type == "image":
            return len(self.img_list)
        return self.n_entries if self.use_lmdb else len(self.image_label_pair)

    def _load_record(self, index: int):
        """index → (filename, annotation_dict)"""
        if self.use_lmdb:
            key  = self.keys[index]
            data = self.txn.get(key)
            rec  = pickle.loads(data)
            filename   = rec["filename"]
            annotation = rec["html"]
        else:
            filename, annotation = self.image_label_pair[index]
        return filename, annotation

    # ====== augmentation methods (원본 그대로) ======

    def _get_watermark(self, text, font_size, color, angle):
        key = (text, font_size, color, int(angle))
        if key in self._wm_cache:
            return self._wm_cache[key]

        # 폰트 객체 꺼내거나 새로 로드
        font = self._font_cache.get(font_size)
        if font is None:
            font = ImageFont.truetype(random.choice(self.font_paths), font_size)
            self._font_cache[font_size] = font

        # 텍스트 크기 계산
        # mask 이미지 크기를 텍스트 bbox 기반으로 적절히 잡습니다
        w_text, h_text = font.getsize(text)
        mask = Image.new('RGBA', (w_text, h_text), (0,0,0,0))
        draw = ImageDraw.Draw(mask)
        draw.text((0,0), text, font=font, fill=color)
        # 회전 후 expand
        mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)
        arr = np.array(mask)  # RGBA ndarray
        self._wm_cache[key] = arr
        return arr

    def apply_watermark(self,
                        image: np.ndarray,
                        opacity_range=(50, 150),
                        rotation_range=(-45, 45),
                        font_size_ratio=5,
                        color_scheme='gray') -> np.ndarray:
        """
        NumPy 기반 워터마킹: center/grid 모드 지원
        """
        h, w = image.shape[:2]
        mode = random.choice(['center', 'grid'])

        # 1) 텍스트·폰트·컬러·불투명도·회전 결정
        text = random.choice(
            CENTER_MODE_WATERMARK_TEXTS if mode=='center' else GRID_MODE_WATERMARK_TEXTS
        )
        font_size = max(8, min(h, w) // font_size_ratio)
        angle = random.uniform(*rotation_range)

        if color_scheme=='gray':
            c = random.randint(128,200)
            alpha = random.randint(*opacity_range)
            color = (c, c, c, alpha)
        else:
            color = (
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255),
                random.randint(*opacity_range)
            )

        wm = self._get_watermark(text, font_size, color, angle)
        mh, mw = wm.shape[:2]
        wm_rgb = wm[...,:3].astype(np.float32)
        wm_a   = (wm[...,3:] / 255.0).astype(np.float32)  # shape (mh,mw,1)

        # 2) 위치 계산 및 합성
        if mode=='center':
            y0, x0 = (h-mh)//2, (w-mw)//2
            positions = [(y0, x0)]
        else:
            # grid: 텍스트 간격 font_size*2
            spacing = font_size * 2
            positions = [
                (yi, xi)
                for yi in range(0, h, spacing)
                for xi in range(0, w, spacing)
            ]

        out = image.astype(np.float32)
        for y0, x0 in positions:
            y1, x1 = y0 + mh, x0 + mw
            if y1<=0 or x1<=0 or y0>=h or x0>=w:
                continue
            # ROI 영역 자르기
            y0c, y1c = max(0,y0), min(h,y1)
            x0c, x1c = max(0,x0), min(w,x1)
            wy0, wx0 = y0c-y0, x0c-x0
            wy1, wx1 = wy0 + (y1c-y0c), wx0 + (x1c-x0c)

            roi = out[y0c:y1c, x0c:x1c]
            alpha = wm_a[wy0:wy1, wx0:wx1]
            rgb   = wm_rgb[wy0:wy1, wx0:wx1]
            # 합성
            out[y0c:y1c, x0c:x1c] = roi*(1-alpha) + rgb*alpha

            # grid 모드는 조금만 보이도록 할 경우
            if mode=='grid' and random.random()<0.3:
                break  # 일부만 채울 수도

        return out.astype(np.uint8)

    def apply_black_noise(self, image: np.ndarray, amount=1e-4):
        h, w = image.shape[:2]
        n = int(h*w*amount)
        idx = np.random.choice(h*w, n, replace=False)
        ys = idx // w
        xs = idx % w
        image[ys, xs] = 0
        return image

    def apply_gaussian_blur(self, image: np.ndarray, ksize=(5, 5), sigma_max=1.0) -> np.ndarray:
        sigma = random.uniform(0.5, sigma_max)
        return cv2.GaussianBlur(image, ksize, sigma)

    def aug(self, image: np.ndarray) -> np.ndarray:
        # 원본 로직 그대로
        aug_prob = 0.1
        if random.random() < aug_prob:
            image = self.apply_watermark(image)
        if random.random() < aug_prob:
            image = self.apply_black_noise(image)
        if random.random() < aug_prob:
            image = self.apply_gaussian_blur(image)
        return image

    # =================getitem =================
    def __getitem__(self, index: int) -> Any:
        # —— image-only ——
        if self.label_type == "image":
            path = self.img_list[index]
            img = Image.open(path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            return self.transform(img) if self.transform else img

        # —— html / cell / bbox / mix ——
        filename, ann = self._load_record(index)
        img_path = self.root_dir / filename
        # print(f'✅✅✅ : {str(img_path)}')
        # img_path = str(self.img_dir / filename)

        # 1) cv2 로드 (BGR)
        arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # 2) optional augmentation
        if self.aug_:
            arr = self.aug(arr)

        # 3) orig_size 계산 (width, height)
        orig_size = (arr.shape[1], arr.shape[0])

        # 4) PIL 변환 + transform
        pil = to_pil_image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        img = self.transform(pil) if self.transform else pil

        # 3) bbox 리사이즈용 타깃 크기 (정사각형 가정)
        if hasattr(img, "shape"):
            tgt_size = img.shape[-1]
        else:
            tgt_size = img.size[0]

        # —— html ——
        if self.label_type == "html":
            tokens = ann["structure"]["tokens"]
            return dict(filename=filename, image=img, html=tokens)


        # —— bbox ——
        if self.label_type == "bbox":
            raw_bboxes = [
                c["bbox"]
                for c in ann["cells"]
                if "bbox" in c
                   and c["bbox"][0] < c["bbox"][2]
                   and c["bbox"][1] < c["bbox"][3]
            ]

            new_bboxes = []
            for box in raw_bboxes:
                augmented = bbox_augmentation_resize(box, orig_size, tgt_size)
                # augmented가 [[x0,y0,x1,y1], ...] 이면 extend,
                # [x0,y0,x1,y1] 평탄 리스트 하나면 append
                if (isinstance(augmented, (list, tuple))
                    and augmented
                    and isinstance(augmented[0], (list, tuple))):
                    new_bboxes.extend(augmented)
                else:
                    new_bboxes.append(augmented)

            return dict(filename=filename, image=img, bbox=new_bboxes)

        # —— mix ——
        if self.label_type == "mix":
            # 1) HTML tokens
            tokens = ann["structure"]["tokens"]
            tokens = extract_structure_tokens(tokens)
            
            if self.otsl_mode:
                otsl, _ = convert_html_to_otsl("".join(tokens))
                tokens = otsl
            
            tokens = [t for t in tokens if t not in ("<thead>","</thead>","<tbody>","</tbody>")]
            
            # 2) raw bboxes
            raw_bboxes = [
                c["bbox"]
                for c in ann["cells"]
                if "bbox" in c
                   and "".join(c["tokens"]).strip()
                   and c["bbox"][0] < c["bbox"][2]
                   and c["bbox"][1] < c["bbox"][3]
            ]

            new_bboxes = []
            for box in raw_bboxes:
                augmented = bbox_augmentation_resize(box, orig_size, tgt_size)
                if (isinstance(augmented, (list, tuple))
                    and augmented
                    and isinstance(augmented[0], (list, tuple))):
                    new_bboxes.extend(augmented)
                else:
                    new_bboxes.append(augmented)

            return dict(filename=filename, image=img, html=tokens, bbox=new_bboxes)

        # 지원하지 않는 타입
        raise ValueError(f"Unknown label_type: {self.label_type}")




    def __del__(self):
        # LMDB 닫기
        if getattr(self, "use_lmdb", False):
            try:
                self.env.close()
            except:
                pass
