from typing import Any
from torch.utils.data import DataLoader, Dataset, Sampler
from functools import partial
import tokenizers as tk
import torch
from torch.utils.data import default_collate
from utils.mask_generator import MaskGenerator
import torch
import multiprocessing
from utils import (
    prepare_html_seq,
    prepare_cell_seq,
    prepare_bbox_seq,
)
# num_workers = int(max(8, multiprocessing.cpu_count() // 3)  )
num_workers = 12
print(f'✅ CPU : {multiprocessing.cpu_count()} | num_workers : {num_workers}')


from torch.utils.data import default_collate
from utils import prepare_html_seq, prepare_bbox_seq

class Collator:
    def __init__(
        self,
        vocab_html: tk.Tokenizer,
        vocab_bbox: tk.Tokenizer,
        max_seq_len_html: int,
        max_seq_len_bbox: int,
        label_type: str,
    ) -> None:
        self.vocab_html = vocab_html
        self.vocab_html.enable_truncation(max_seq_len_html)
        self.vocab_bbox = vocab_bbox
        self.vocab_bbox.enable_truncation(max_seq_len_bbox)
        self.vocab_cp = vocab_bbox
        self.vocab_cp.enable_truncation(max_seq_len_bbox//2)
        self.label_type = label_type

    def __call__(self, batch) -> Any:
        return self._collate_batch(batch)

    def _collate_batch(self, batch: list,):
        # 1) image
        if "cell" in self.label_type:
            images = [img for sample in batch for img in sample[0]]
        else:
            images = [sample["image"] for sample in batch]
        images = default_collate(images)

        # 2) filename
        if "cell" in self.label_type:
            filenames = [(item["filename"], item["bbox_id"])
                         for sample in batch for item in sample[1]]
        else:
            filenames = [sample["filename"] for sample in batch]

        label = {"filename": filenames}

        # 3) mix 모드 먼저 처리
        if self.label_type == "mix":

            debug_name = ['PMC4150558_006_00.png', 'PMC4077062_007_00.png', 'PMC6057087_003_00.png', 'PMC3446521_003_00.png', 'PMC4149784_006_00.png', 'PMC1847435_008_00.png', 'PMC5681930_003_00.png', 'PMC5445258_003_00.png']

            # html
            html_tokens = ["".join(prepare_html_seq(s["html"])) for s in batch]
            label["html"] = self.vocab_html.encode_batch(html_tokens)
            # bbox
            bbox_strs = []
            cp_strs = []
            for s in batch:
                coords_bbox = [c for box in s["bbox"] for c in box]
                bbox_tokens = prepare_bbox_seq(coords_bbox)
                bbox_strs.append(" ".join(bbox_tokens))

                # if s["filename"] in debug_name:
                #     print(f'🔥s["filename"] : {s["filename"]}')
                #     print(f"coords_bbox : {len(coords_bbox)} | bbox_tokens : {len(bbox_tokens)}")
                    # print(f"coords_cp : {len(coords_cp)} | cp_tokens : {len(cp_tokens)} | cp_tokens : {len(cp_tokens)}\n\n")


            label["bbox"] = self.vocab_bbox.encode_batch(bbox_strs)

        # 3) mix 모드 먼저 처리
        if self.label_type == "mix_cp":

            # debug_name = ['PMC4150558_006_00.png', 'PMC4077062_007_00.png', 'PMC6057087_003_00.png', 'PMC3446521_003_00.png', 'PMC4149784_006_00.png', 'PMC1847435_008_00.png', 'PMC5681930_003_00.png', 'PMC5445258_003_00.png']


            # html
            html_tokens = ["".join(prepare_html_seq(s["html"])) for s in batch]
            label["html"] = self.vocab_html.encode_batch(html_tokens)
            # bbox
            bbox_strs = []
            cp_strs = []
            for s in batch:
                coords_bbox = [c for box in s["bbox"] for c in box]
                bbox_tokens = prepare_bbox_seq(coords_bbox)
                bbox_strs.append(" ".join(bbox_tokens))

                coords_cp = [c for box in s["cp"] for c in box]
                cp_tokens = prepare_bbox_seq(coords_cp)
                cp_strs.append(" ".join(cp_tokens))

                # if s["filename"] in debug_name:
                #     print(f'🔥s["filename"] : {s["filename"]}')
                #     print(f"coords_bbox : {len(coords_bbox)} | bbox_tokens : {len(bbox_tokens)}")
                #     print(f"coords_cp : {len(coords_cp)} | cp_tokens : {len(cp_tokens)}\n\n")

            label["bbox"] = self.vocab_bbox.encode_batch(bbox_strs)
            label["cp"] = self.vocab_cp.encode_batch(cp_strs)
            
        # 4) html-only
        elif self.label_type == "html":
            html_tokens = ["".join(prepare_html_seq(s["html"])) for s in batch]
            label["html"] = self.vocab_html.encode_batch(html_tokens)

        # 5) bbox-only
        elif self.label_type == "bbox":
            bbox_strs = []
            for s in batch:
                coords = [c for box in s["bbox"] for c in box]
                tokens = prepare_bbox_seq(coords)
                bbox_strs.append(" ".join(tokens))
            label["bbox"] = self.vocab_bbox.encode_batch(bbox_strs)

        # 6) cell 모드는 기존대로
        elif self.label_type == "cell":
            # collate_fn 자체가 cell 모드용으로 다르게 호출되기 때문에
            # 여긴 그냥 pass 해도 됩니다.
            pass
        

        # print(f"label bbox : \n{len(bbox_strs[0])} | {bbox_strs[0]}\n")
        # print(f"label cp : \n{len(cp_strs[0])} | {cp_strs[0]} \n\n\n")
        return images, label


# class Collator:
#     def __init__(
#         self,
#         # vocab: tk.Tokenizer, # ❌
#         vocab_html: tk.Tokenizer,# ✅
#         vocab_bbox: tk.Tokenizer,# ✅
#         # max_seq_len: int,# ❌
#         max_seq_len_html: int,# ✅
#         max_seq_len_bbox: int,# ✅
#         label_type: str,) -> None:

#         self.vocab_html = vocab_html
#         self.vocab_html.enable_truncation(max_seq_len_html)
#         self.vocab_bbox = vocab_bbox
#         self.vocab_bbox.enable_truncation(max_seq_len_bbox)
#         self.label_type = label_type

#     def __call__(self, batch) -> Any:
#         return self._collate_batch(batch, self.vocab_html, self.vocab_bbox, self.label_type)

#     def _collate_batch(
#         self,
#         batch: list[dict],
#         vocab_html: tk.Tokenizer,
#         vocab_bbox: tk.Tokenizer,
#         label_type: str,
#     ):
#         if "cell" in label_type:
#             image_list = [j for i in batch for j in i[0]]
#         else:
#             image_list = [i["image"] for i in batch]
#         image_list = default_collate(image_list)

#         if "cell" in label_type:
#             filename = [(j["filename"], j["bbox_id"]) for i in batch for j in i[1]]
#         else:
#             filename = [i["filename"] for i in batch]
#         label = dict(filename=filename)

#         if "html" in label_type:
#             html_list = ["".join(prepare_html_seq(i["html"])) for i in batch]
#             label["html"] = vocab_html.encode_batch(html_list)

#         if "bbox" in label_type:
#             bbox_list = [" ".join(prepare_bbox_seq(i["bbox"])) for i in batch]
#             label["bbox"] = vocab_bbox.encode_batch(bbox_list)

#         if 'mix' in label_type: # ✅    
#             html_list = ["".join(prepare_html_seq(i["html"])) for i in batch] # ✅
#             label["html"] = vocab_html.encode_batch(html_list) # ✅

#             bbox_list = [" ".join(prepare_bbox_seq(i["bbox"])) for i in batch] # ✅
#             label["bbox"] = vocab_bbox.encode_batch(bbox_list) # ✅

#         return image_list, label


def generate_mask_for_batch_samples(
    batch, grid_size: int, num_mask_patches: int, min_num_patches: int
):
    N = len(batch)
    mg = MaskGenerator(
        input_size=grid_size,
        num_mask_patches=num_mask_patches,
        min_num_patches=min_num_patches,
    )
    mask_list = [mg() for _ in range(N)]
    return default_collate(batch), default_collate(mask_list)


def dataloader_vae(
    dataset: Dataset, batch_size: int, sampler: Sampler = None, **kwargs
) -> DataLoader:
    dataloader = DataLoader(
        dataset, batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True
    )

    return dataloader

def dataloader_beit(
    dataset: Dataset,
    grid_size: int,
    num_mask_patches: int,
    min_num_patches: int,
    batch_size: int,
    sampler: Sampler = None,
    **kwargs
):
    dataloader = DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        collate_fn=partial(
            generate_mask_for_batch_samples,
            grid_size=grid_size,
            num_mask_patches=num_mask_patches,
            min_num_patches=min_num_patches,
        ),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def dataloader_html(
    dataset: Dataset,
    batch_size: int,
    vocab_html: tk.Tokenizer,
    vocab_bbox: tk.Tokenizer,
    max_seq_len_html: int,
    max_seq_len_bbox: int,
    label_type: str,
    aug: bool,
    sampler=None,
) -> DataLoader:
    collate_fn = Collator(vocab_html, vocab_bbox, max_seq_len_html, max_seq_len_bbox, label_type)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=1,
        sampler=sampler,
    )

    return dataloader

    # weights_only=True
