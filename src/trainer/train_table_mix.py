from typing import Tuple, List, Union, Dict, Optional
import torch
import wandb
import json
import os
from torch import nn, Tensor, autograd
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
import tokenizers as tk
import torch.nn.functional as F
from time import time
import pandas as pd
import torch.amp as amp
import pynvml
import zipfile  # BadZipFile 예외 처리를 위해
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


# conda activate unitable_gitlab
# pip install nvidia-ml-py3

from trainer.utils import (
    Batch_MIX, # Batch,
    configure_optimizer_weight_decay,
    turn_off_beit_grad,
    turn_on_beit_grad,
    VALID_HTML_TOKEN,
    VALID_BBOX_TOKEN,
)
import random
import numpy as np


def setup_l40s_optimization():
    """L40S 전용 최적화 설정"""
    
    # ✅ Ada Lovelace 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # ✅ L40S 4세대 Tensor Core 활용
    torch.backends.cuda.enable_flash_sdp(True)  # Flash Attention
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # ✅ 메모리 최적화 (44GB 활용)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # ✅ cuDNN 9.x 최적화
    torch.backends.cudnn.deterministic = False  # 성능 우선
    
    print("✅ L40S 최적화 설정 완료!")




def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  # 다중 GPU 사용 시  
    torch.backends.cudnn.deterministic = True  # 🚀 재현성을 위해 설정  

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)  # Flash Attention
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

    print(f'✅ setting cuda backends')
    
set_seed(42)  # ✅ 실행 시 동일한 결과 보장

pynvml.nvmlInit()
# 사용할 GPU 디바이스 인덱스 (0,1,…)
GPU_INDEX = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
_nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)


import gc
import torch
import psutil
import os

def print_memory_state(tag=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1e9  # Resident Set Size (actual RAM usage)
    vms = process.memory_info().vms / 1e9  # Virtual Memory Size
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[{tag}] CPU RSS: {rss:.2f} GB | VMS: {vms:.2f} GB | GPU: {gpu_mem:.2f} GB")

def safe_gc_collect(verbose=False):
    if verbose:
        print_memory_state("Before GC")
    torch.cuda.empty_cache()
    gc.collect(0)
    gc.collect(1)
    # gc.collect(2)
    torch.cuda.empty_cache()
    if verbose:
        print_memory_state("After GC")


def get_gpu_stats():
    """
    현재 GPU의 사용률, 메모리 사용량, 온도를 반환합니다.
    - gpu_util    : 정수 % 단위
    - mem_used    : MiB 단위
    - mem_total   : MiB 단위
    - temperature : °C 단위
    """
    util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
    return {
        "gpu_util": util.gpu,
        "gpu_mem": mem.used // 1024**2,
    }




from utils import (
    printer,
    compute_grad_norm,
    count_total_parameters,
    batch_autoregressive_decode,
    combine_filename_pred_gt,
)


SNAPSHOT_KEYS = set(["EPOCH", "STEP", "OPTIMIZER", "LR_SCHEDULER", "MODEL", "LOSS"])


class TableTrainer_MIX:
    """A trainer for table recognition. The supported tasks are:
    1) table structure extraction
    2) table cell bbox detection
    3) table cell content recognition

    Args:
    ----
        device: gpu id
        vocab: a vocab shared among all tasks
        model: model architecture
        log: logger
        exp_dir: the experiment directory that saves logs, wandb files, model weights, and checkpoints (snapshots)
        snapshot: specify which snapshot to use, only used in training
        model_weights: specify which model weight to use, only used in testing
        beit_pretrained_weights: load SSL pretrained visual encoder
        freeze_beit_epoch: freeze beit weights for the first {freeze_beit_epoch} epochs
    """
    
    def __init__(
        self,
        device: int, 
        vocab_html: tk.Tokenizer, # ✅
        vocab_bbox: tk.Tokenizer, # ✅
        model, # ✅ 통합 모델
        log: logging.Logger, 
        exp_dir: Path,
        snapshot: Path = None,
        model_weights: str = None, 
        beit_pretrained_weights: str = None, 
        freeze_beit_epoch: int = None,
        use_mix_loss: bool = False,
        otsl_mode: bool = False,
        finetune_mode: bool = False,
    ) -> None:
        
        self.use_mix_loss = use_mix_loss
        self.otsl_mode = otsl_mode
        self.finetune_mode = finetune_mode

        self.accumulate_grad_steps = 8  # 통합 모델용

        print(f"✅ use mix loss : {self.use_mix_loss}")
        print(f"✅ use OTSL : {self.otsl_mode}")
        
        self.device = device
        self.log = log
        self.exp_dir = exp_dir
        self.vocab_html = vocab_html
        self.vocab_bbox = vocab_bbox
        self.padding_idx_html = vocab_html.token_to_id("<pad>") # ✅
        self.padding_idx_bbox = vocab_bbox.token_to_id("<pad>") # ✅
        self.freeze_beit_epoch = freeze_beit_epoch

        self.model = model.to(device)

        # loss for training html, bbox
        self.criterion_html = nn.CrossEntropyLoss(ignore_index=self.padding_idx_html) # ✅
        self.criterion_mix = nn.L1Loss(reduction="none").to(self.device) # ✅
        self.scaler = amp.GradScaler()
        
        
        ####### 기존 가중치 불러오는 코드 #######
        if finetune_mode and not (snapshot is not None and snapshot.is_file()):
            # SharedEncoder_DualDecoder는 단일 모델이므로 별도 처리 필요
            # 기존 개별 모델 가중치를 SharedEncoder_DualDecoder로 변환하는 로직 필요
            print("Warning: Finetuning with separate HTML/BBOX weights not yet implemented for SharedEncoder_DualDecoder")

        elif (snapshot is not None) and (snapshot.is_file()):
            pass
            ####### 새롭게 학습하는 코드 #######
        else:
            if beit_pretrained_weights is not None and Path(beit_pretrained_weights).is_file():
                self.load_pretrained_beit(Path(beit_pretrained_weights))
            
            assert (
                snapshot is None or model_weights is None
            ), "Cannot set snapshot and model_weights at the same time!"


        if snapshot is not None and snapshot.is_file():
            print(f'✅ snapshot :  {os.getcwd()} | {Path(snapshot).is_file()}')
            self.snapshot = self.load_snapshot(snapshot)
            self.model.load_state_dict(self.snapshot["MODEL"])
            self.start_epoch = self.snapshot["EPOCH"]
            self.global_step = self.snapshot["STEP"]
            
        elif model_weights is not None and Path(model_weights).is_file():
            self.snapshot = None
            self.start_epoch = 0
            self.global_step = 0
            self.load_model(Path(model_weights))

        else:
            self.snapshot = None
            self.start_epoch = 0
            self.global_step = 0


        # freeze beit weights if needed
        if freeze_beit_epoch and freeze_beit_epoch > 0:
            self._freeze_beit()

        self.model = self.model.to(device)
        self.model = DDP(self.model, device_ids=[device])

        torch.cuda.set_device(device)  # master gpu takes up extra memory
        torch.cuda.empty_cache()
    
    
    def _freeze_beit(self):
        if self.start_epoch < self.freeze_beit_epoch:
            turn_off_beit_grad(self.model)
            self.log.info(
                printer(
                    self.device,
                    f"Lock SSL params for {self.freeze_beit_epoch} epochs (params: {count_total_parameters(self.model) / 1e6:.2f}M) - Current epoch {self.start_epoch + 1}"
                )
            )
        else:
            turn_on_beit_grad(self.model)
            self.log.info(
                printer(
                    self.device,
                    f"Unlock all weights (params: {count_total_parameters(self.model) / 1e6:.2f}M) - Current epoch {self.start_epoch + 1}",
                )
            )



    def train_epoch(
        self,
        epoch: int,
        target: str,
        loss_weights: List[float],
        grad_clip: float = None,
    ):

        # ✅ Gradient Accumulation용 카운터
        accumulation_count = 0

        avg_loss = 0.0
        st_1epoch = time()
        time_inf_1epoch = 0
        time_backward_1epoch = 0

        gpu_util_sum = 0.0
        gpu_mem_sum = 0.0
        loss_cp_sum = 0.0
        loss_html_sum = 0.0
        loss_bbox_sum = 0.0
        loss_mix_sum = 0.0

        tact_10it_inf = 0
        tact_10it_back = 0

        for i, obj in enumerate(self.train_dataloader):
            batch = Batch_MIX(
                device=self.device,
                target=target,
                vocab_html=self.vocab_html,
                vocab_bbox=self.vocab_bbox,
                obj=obj,
                use_mix_loss=self.use_mix_loss,
                otsl_mode=self.otsl_mode,
            )
            
            # ✅ 수정: 통합 모델 - accumulation 시작 시에만 zero_grad 호출
            if accumulation_count == 0:
                self.optimizer.zero_grad()

            st = time()
            with amp.autocast(device_type='cuda'):
                if 'cp' in target:
                    loss, _, time_dict = batch.inference_shared_cp(
                        self.model,
                        criterion_cp=self.criterion_cp,
                        criterion_html=self.criterion_html,
                        criterion_bbox=self.criterion_bbox,
                        criterion_mix=self.criterion_mix,
                        loss_weights=loss_weights,
                    )
                else:
                    loss, _, time_dict = batch.inference_shared(
                        self.model,
                        criterion_html=self.criterion_html,
                        criterion_bbox=self.criterion_bbox,
                        criterion_mix=self.criterion_mix,
                        loss_weights=loss_weights,
                    )


                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

                tact_10it_inf += time() - st
                total_loss = loss["total"]
                
                # ✅ 수정: Gradient accumulation을 위해 loss를 accumulation step으로 나누기
                total_loss = total_loss / self.accumulate_grad_steps

                time_inf_1epoch += time_dict["time_inf"]

            st = time()
            self.scaler.scale(total_loss).backward()
            tact_10it_back += time() - st
            time_backward_1epoch += time() - st

            # ✅ 추가: 카운터 증가 (중요!)
            accumulation_count += 1

            # ✅ 통합 모델: Accumulation step마다 업데이트
            if accumulation_count >= self.accumulate_grad_steps:
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                
                self.scaler.step(self.optimizer)
                self.lr_scheduler.step()
                self.scaler.update()
                accumulation_count = 0  # 카운터 리셋

            # ✅ 수정: 원본 loss 값을 사용하여 로깅 (scaling 이전 값)
            total_loss_scalar = (total_loss * self.accumulate_grad_steps).item()  # 원본 loss 복원
            avg_loss += total_loss_scalar


            peak_mem = torch.cuda.max_memory_allocated() / 1e9

            gpu_util_sum += util
            gpu_mem_sum += peak_mem
            loss_cp_sum += loss["cp"].item()
            loss_html_sum += loss["html"].item()
            loss_bbox_sum += loss["bbox"].item()
            loss_mix_sum += loss["mix_loss"].item()

            self.global_step += 1

            if i % 10 == 0:
                loss_info = f"Loss {total_loss_scalar:.3f} ({avg_loss / (i + 1):.3f})"
                loss_info += f" CP {loss['cp'].item():.3f}"
                loss_info += f" HTML {loss['html'].item():.3f}"
                loss_info += f" BBOX {loss['bbox'].item():.3f}"
                loss_info += f" Mix-LOSS {loss['mix_loss'].item():.3f}"
                
                # Learning rate monitoring for shared model
                current_lr = self.lr_scheduler.get_last_lr()[0]
                loss_info += f" LR {current_lr:.2e}"

                self.log.info(
                    printer(
                        self.device,
                        f"Epoch {epoch} Step {i + 1}/{len(self.train_dataloader)} | {loss_info}",
                    )
                )
                tact_10it_inf = 0
                tact_10it_back = 0

            if i % 10000 == 0 and i != 0:
                safe_gc_collect(verbose=True)
        
        del loss, batch
        # Epoch-level 평균 계산

        time_dict_1epoch = dict()
        if self.device == 0:
            n_batches = len(self.train_dataloader)
            time_total_1epoch = time() - st_1epoch
            loss_total_avg_per_1epoch = avg_loss / n_batches

            time_dict_1epoch = {
                "epoch": epoch,
                "time_total_1epoch": time_total_1epoch,
                # "time_inf_1epoch": time_inf_1epoch,
                # "time_backward_1epoch": time_backward_1epoch,
                "gpu_util_avg_per_1epoch": f"{gpu_util_sum / n_batches:.2f}%",
                "gpu_mem_avg_per_1epoch": f"{gpu_mem_sum / n_batches:.2f}GB",
                "train_loss_total": loss_total_avg_per_1epoch,
                "train_loss_cp": loss_cp_sum / n_batches,
                "train_loss_html": loss_html_sum / n_batches,
                "train_loss_bbox": loss_bbox_sum / n_batches,
                "train_loss_mix": loss_mix_sum / n_batches,

            }
            
        return time_dict_1epoch



                
        
    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        train_cfg: DictConfig,
        valid_cfg: DictConfig,):
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # ensure correct weight decay: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        optim_params = configure_optimizer_weight_decay(
            self.model.module, weight_decay=train_cfg.optimizer.weight_decay)
        self.optimizer = instantiate(train_cfg.optimizer, optim_params)
        # print(f'🔥optimizer : {self.optimizer} ')

        
        self.criterion_bbox = None
        if ("bbox" in train_cfg.target) or ("mix" in train_cfg.target): # ✅
            tmp = [
                self.vocab_bbox.token_to_id(i)
                for i in VALID_BBOX_TOKEN[
                    : train_cfg.img_size[0] + 2
                ]  # +1 for <eos> +1 for bbox == img_size
            ]
            tmp = [1.0 if i in tmp else 0.0 for i in range(self.vocab_bbox.get_vocab_size())]
            self.criterion_bbox = nn.CrossEntropyLoss(
                weight=torch.tensor(tmp, device=self.device),
                ignore_index=self.padding_idx_bbox,
            )
        
        self.criterion_cp = None
        if ("cp" in train_cfg.target):
            tmp = [
                self.vocab_bbox.token_to_id(i)
                for i in VALID_BBOX_TOKEN[
                    : train_cfg.img_size[0] + 2
                ]  # +1 for <eos> +1 for bbox == img_size
            ]
            tmp = [1.0 if i in tmp else 0.0 for i in range(self.vocab_bbox.get_vocab_size())]
            self.criterion_cp = nn.CrossEntropyLoss(
                weight=torch.tensor(tmp, device=self.device),
                ignore_index=self.padding_idx_bbox,
            )
        

        best_loss = float("inf")
        self.model.train()

        if (self.freeze_beit_epoch) and (self.start_epoch < self.freeze_beit_epoch):
            max_epoch = self.freeze_beit_epoch
        else:
            max_epoch = train_cfg.epochs

        if self.finetune_mode:
            self.freeze_beit_epoch = max_epoch

        self.max_epoch = max_epoch

        # ================= LR Scheduler (Warmup 포함) =================
        # 웜업(3 에포크) + s1(29 에포크) + s2(16 에포크) = 총 48 에포크
        # SharedEncoder_DualDecoder용 스케줄러 설정

        # ⚠️ 중요: 배치 누적 횟수로 나눠줌
        steps_per_epoch = len(self.train_dataloader) // self.accumulate_grad_steps

        warmup_steps = steps_per_epoch * 3   # 3 epochs
        s1_steps = steps_per_epoch * (self.max_epoch // 3) * 2
        s2_steps = steps_per_epoch * (self.max_epoch // 3)

        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-5, total_iters=warmup_steps)
        s1_scheduler = CosineAnnealingLR(self.optimizer, T_max=s1_steps, eta_min=1e-6)
        s2_scheduler = CosineAnnealingLR(self.optimizer, T_max=s2_steps, eta_min=1e-6)
        milestones = [warmup_steps, warmup_steps + s1_steps] # 각 스케줄러가 끝나는 지점
        self.lr_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, s1_scheduler, s2_scheduler], milestones=milestones)


        if self.snapshot is not None:
            self.optimizer.load_state_dict(self.snapshot["OPTIMIZER"])
            self.lr_scheduler.load_state_dict(self.snapshot["LR_SCHEDULER"])



        for epoch in range(self.start_epoch, max_epoch):
            print(f'👉 max_epoch : {max_epoch} | current epoch : {epoch}')
            train_dataloader.sampler.set_epoch(epoch)
            time_dict_1epoch = self.train_epoch( 
                                                epoch,
                                                grad_clip=train_cfg.grad_clip,
                                                target=train_cfg.target,
                                                loss_weights=train_cfg.loss_weights,
                                            )
            
            torch.cuda.empty_cache()

            valid_loss_dict = self.valid(valid_cfg)

            valid_total_loss = valid_loss_dict['total']
            valid_cp_loss = valid_loss_dict['cp']
            valid_html_loss = valid_loss_dict['html']
            valid_bbox_loss = valid_loss_dict['bbox']
            valid_mix_loss = valid_loss_dict['mix']

            time_dict_1epoch["val_loss_total"] = float(valid_total_loss)
            time_dict_1epoch["val_loss_cp"] = float(valid_cp_loss)
            time_dict_1epoch["val_loss_html"] = float(valid_html_loss)
            time_dict_1epoch["val_loss_bbox"] = float(valid_bbox_loss)
            time_dict_1epoch["val_loss_mix"] = float(valid_mix_loss)
                

            if self.device == 0:
                wandb.log(
                    {"valid loss (epoch)": valid_total_loss},
                    step=self.global_step,
                )

                if epoch % train_cfg.save_every == 0:
                    self.save_snapshot(epoch, best_loss)
                
                if valid_total_loss < best_loss:
                    self.save_model(epoch)
                    best_loss = valid_total_loss


                # 1 에포크가 끝나고 기록을 남길 때
                df_time = pd.DataFrame({k: [v] for k, v in time_dict_1epoch.items()})
                filename = Path(self.exp_dir) / "train_log" / f"log_{os.path.basename(self.exp_dir)}.xlsx"
                os.makedirs(filename.parent, exist_ok=True)

                try:
                    if epoch == 0 or not filename.exists():
                        # 첫 기록 혹은 파일이 없으면 새로 생성
                        df_time.to_excel(filename, index=False)
                    else:
                        # 이어쓰기 모드
                        with pd.ExcelWriter(
                            filename,
                            mode="a",
                            engine="openpyxl",
                            if_sheet_exists="overlay"
                        ) as writer:
                            # 기존 시트의 마지막 행을 구해서 그 다음 줄에 쓴다
                            sheet = writer.sheets["Sheet1"]
                            startrow = sheet.max_row
                            df_time.to_excel(writer, startrow=startrow, index=False, header=False)

                except zipfile.BadZipFile as e:
                    # CRC 에러 등 엑셀 파일이 깨졌을 경우
                    print(f"[경고] 로그 파일에 기록하는 중 에러 발생 ({e!r}) → 이번 에포크 기록은 건너뛰고 계속 진행합니다.")
                    # 원한다면, CSV 로 백업:
                    csv_backup = filename.with_suffix(".csv")
                    df_time.to_csv(csv_backup, mode="a", header=not csv_backup.exists(), index=False)

                except Exception as e:
                    # 다른 쓰기 에러는 그대로 알려주되, 트레이닝을 중단시키진 않음
                    print(f"[경고] 로그 파일 기록 중 알 수 없는 에러 ({e!r})")



    def valid(self, cfg: DictConfig):
        total_loss = 0.0
        cp_loss = 0.0
        html_loss = 0.0
        bbox_loss = 0.0
        mix_loss = 0.0

        avg_total_loss = 0.0
        avg_cp_loss = 0.0
        avg_html_loss = 0.0
        avg_bbox_loss = 0.0
        avg_mix_loss = 0.0

        total_samples = 0

        self.model.eval()

        # print(f'🔥 self.valid_dataloader : {len(self.valid_dataloader)}')
        for i, obj in enumerate(self.valid_dataloader):
            batch = Batch_MIX(device=self.device, 
                              target=cfg.target, 
                              vocab_html=self.vocab_html, 
                              vocab_bbox=self.vocab_bbox, 
                              obj=obj,
                              use_mix_loss = self.use_mix_loss,
                              otsl_mode = self.otsl_mode)

            with torch.no_grad():

                if 'cp' in cfg.target:
                    loss, _, time_dict = batch.inference_shared_cp(
                        self.model,
                        criterion_cp=self.criterion_cp,
                        criterion_html=self.criterion_html,
                        criterion_bbox=self.criterion_bbox,
                        criterion_mix=self.criterion_mix,
                        loss_weights=cfg.loss_weights,
                    )
                else:
                    loss, _, time_dict = batch.inference_shared(
                        self.model,
                        criterion_html=self.criterion_html,
                        criterion_bbox=self.criterion_bbox,
                        criterion_mix=self.criterion_mix,
                        loss_weights=cfg.loss_weights,
                    )



            total_loss = loss["total"].detach().cpu().data
            cp_loss = loss["cp"].detach().cpu().data
            html_loss = loss["html"].detach().cpu().data
            bbox_loss = loss["bbox"].detach().cpu().data
            mix_loss = loss["mix_loss"].detach().cpu().data

            avg_total_loss += total_loss * batch.image.shape[0]
            avg_cp_loss += cp_loss * batch.image.shape[0]
            avg_html_loss += html_loss * batch.image.shape[0]
            avg_bbox_loss += bbox_loss * batch.image.shape[0]
            avg_mix_loss += mix_loss * batch.image.shape[0]

            total_samples += batch.image.shape[0]

            if i % 10 == 0:
                loss_info = f"Loss {total_loss:.3f} ({avg_total_loss / total_samples:.3f})"
                if not isinstance(loss["cp"], int):
                    loss_info += f" CP {loss['cp'].detach().cpu().data:.3f}"
                if not isinstance(loss["html"], int):
                    loss_info += f" Html {loss['html'].detach().cpu().data:.3f}"
                if not isinstance(loss["cell"], int):
                    loss_info += f" Cell {loss['cell'].detach().cpu().data:.3f}"
                if not isinstance(loss["bbox"], int):
                    loss_info += f" Bbox {loss['bbox'].detach().cpu().data:.3f}"
                if not isinstance(loss["mix_loss"], int) and loss['mix_loss'] != 0:
                    loss_info += f" Mix-Combine {loss['mix_loss'].detach().cpu().data:.3f}"
                    
                self.log.info(
                    printer(
                        self.device,
                        f"Valid: Step {i + 1}/{len(self.valid_dataloader)} | {loss_info}",
                    )
                )

        loss_dict = {
            'total' : avg_total_loss / total_samples,
            'cp' : avg_cp_loss / total_samples,
            'html' : avg_html_loss / total_samples,
            'bbox' : avg_bbox_loss / total_samples,
            'mix' : avg_mix_loss / total_samples,
            }
        return loss_dict

    def test(self, test_dataloader: DataLoader, cfg: DictConfig, save_to: str):
        total_result = dict()
        for i, obj in enumerate(test_dataloader):
            batch = Batch_MIX(device=self.device, 
                              target=cfg.target, 
                              vocab_html=self.vocab_html, 
                              vocab_bbox=self.vocab_bbox, 
                              obj=obj,
                              use_mix_loss= self.use_mix_loss,
                              otsl_mode = self.otsl_mode)

            if cfg.target == "html":
                prefix = [self.vocab_html.token_to_id("[html]")]
                valid_token_whitelist = [
                    self.vocab.token_to_id(i) for i in VALID_HTML_TOKEN
                ]
                valid_token_blacklist = None

            elif cfg.target == "bbox":
                prefix = [self.vocab_bbox.token_to_id("[bbox]")]
                valid_token_whitelist = [
                    self.vocab.token_to_id(i)
                    for i in VALID_BBOX_TOKEN[: cfg.img_size[0]]
                ]
                valid_token_blacklist = None
            else:
                raise NotImplementedError

            pred_id = batch_autoregressive_decode(
                device=self.device,
                model=self.model,
                batch_data=batch,
                prefix=prefix,
                max_decode_len=cfg.max_seq_len,
                eos_id=self.vocab.token_to_id("<eos>"),
                valid_token_whitelist=valid_token_whitelist,
                valid_token_blacklist=valid_token_blacklist,
                sampling=cfg.sampling,
            )

            if cfg.target == "html":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.html_tgt,
                    vocab=self.vocab,
                    type="html",
                )
            elif cfg.target == "cell":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.cell_tgt,
                    vocab=self.vocab,
                    type="cell",
                )
            elif cfg.target == "bbox":
                result = combine_filename_pred_gt(
                    filename=batch.name,
                    pred_id=pred_id,
                    gt_id=batch.bbox_tgt,
                    vocab=self.vocab,
                    type="bbox",
                )
            else:
                raise NotImplementedError

            total_result.update(result)

            if i % 10 == 0:
                self.log.info(
                    printer(
                        self.device,
                        f"Test: Step {i + 1}/{len(test_dataloader)}",
                    )
                )

        self.log.info(
            printer(
                self.device,
                f"Converting {len(total_result)} samples to html tables ...",
            )
        )

        with open(
            os.path.join(save_to, cfg.save_to_prefix + f"_{self.device}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(total_result, f, indent=4)

        return total_result

    def save_model(self, epoch: int):
        filename = Path(self.exp_dir) / "model" / f"epoch{epoch}_model.pt"
        torch.save(self.model.module.state_dict(), filename)
        self.log.info(printer(self.device, f"Saving model to {filename}"))
        filename = Path(self.exp_dir) / "model" / "best.pt"
        torch.save(self.model.module.state_dict(), filename)

    def save_model_mix(self, epoch: int):
        filename_html = Path(self.exp_dir) / "model" / f"epoch{epoch}_model_html.pt"
        torch.save(self.model_html.module.state_dict(), filename_html)
        self.log.info(printer(self.device, f"Saving model to {filename_html}"))
        filename_html = Path(self.exp_dir) / "model" / "best_html.pt"
        torch.save(self.model_html.module.state_dict(), filename_html)

        filename_bbox = Path(self.exp_dir) / "model" / f"epoch{epoch}_model_bbox.pt"
        torch.save(self.model_bbox.module.state_dict(), filename_bbox)
        self.log.info(printer(self.device, f"Saving model to {filename_bbox}"))
        filename_bbox = Path(self.exp_dir) / "model" / "best_bbox.pt"
        torch.save(self.model_bbox.module.state_dict(), filename_bbox)



    def load_model(self, path: Union[str, Path]):
        self.model.load_state_dict(torch.load(path, map_location=f"cuda:{self.device}", weights_only=True))
        self.log.info(printer(self.device, f"Loading model from {path}"))

    def load_model_mix(self, model, path: Union[str, Path]):
        model.load_state_dict(torch.load(path, map_location=f"cuda:{self.device}", weights_only=True))
        self.log.info(printer(self.device, f"Loading model from {path}"))

    def save_snapshot(self, epoch: int, best_loss: float):
        state_info = {
            "EPOCH": epoch + 1,
            "STEP": self.global_step,
            "OPTIMIZER": self.optimizer.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler.state_dict(),
            "MODEL": self.model.module.state_dict(),
            "LOSS": best_loss,
        }

        snapshot_path = Path(self.exp_dir) / "snapshot" / f"epoch{epoch}_snapshot.pt"
        torch.save(state_info, snapshot_path)

        self.log.info(printer(self.device, f"Saving snapshot to {snapshot_path}"))

    def save_snapshot_mix(self, epoch: int, best_loss: float):
        state_info_html = {
            "EPOCH": epoch + 1,
            "STEP": self.global_step,
            "OPTIMIZER": self.optimizer_html.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler_html.state_dict(),
            "MODEL": self.model_html.module.state_dict(),
            "LOSS": best_loss,
        }

        # HTML
        snapshot_path_html = Path(self.exp_dir) / "snapshot" / f"epoch{epoch}_snapshot_html.pt"
        torch.save(state_info_html, snapshot_path_html)
        self.log.info(printer(self.device, f"[HTML] Saving snapshot to {snapshot_path_html}"))


        # BBOX
        state_info_bbox = {
            "EPOCH": epoch + 1,
            "STEP": self.global_step,
            "OPTIMIZER": self.optimizer_bbox.state_dict(),
            "LR_SCHEDULER": self.lr_scheduler_bbox.state_dict(),
            "MODEL": self.model_bbox.module.state_dict(),
            "LOSS": best_loss,
        }
            
        snapshot_path_bbox = Path(self.exp_dir) / "snapshot" / f"epoch{epoch}_snapshot_bbox.pt"
        torch.save(state_info_bbox, snapshot_path_bbox)
        self.log.info(printer(self.device, f"[BBOX] Saving snapshot to {snapshot_path_bbox}"))
        

    # SNAPSHOT_KEYS = set(["EPOCH", "STEP", "OPTIMIZER", "LR_SCHEDULER", "MODEL", "LOSS"])
    def load_snapshot(self, path: Path):
        self.log.info(printer(self.device, f"Loading snapshot from {path}"))
        snapshot = torch.load(path, map_location=f"cuda:{self.device}", weights_only=True)
        assert SNAPSHOT_KEYS.issubset(snapshot.keys())
        return snapshot

    def load_snapshot_mix(self, path_html: Path, path_bbox: Path):
        self.log.info(printer(self.device, f"Loading snapshot from {path_html}"))
        self.log.info(printer(self.device, f"Loading snapshot from {path_bbox}"))
        snapshot_html = torch.load(path_html, map_location=f"cuda:{self.device}")
        assert SNAPSHOT_KEYS.issubset(snapshot_html.keys())
        print(f'✅ snapshot_html.keys() : {snapshot_html.keys()}')


    
        snapshot_bbox = torch.load(path_bbox, map_location=f"cuda:{self.device}")
        assert SNAPSHOT_KEYS.issubset(snapshot_bbox.keys())
        print(f'✅ snapshot_bbox.keys() : {snapshot_bbox.keys()}')
    
        return snapshot_html, snapshot_bbox

    def load_pretrained_beit(self, path: Path):
        self.log.info(printer(self.device, f"Loading pretrained BEiT from {path}"))
        beit = torch.load(path, map_location=f"cuda:{self.device}", weights_only=True)
        redundant_keys_in_beit = [
            "cls_token",
            "mask_token",
            "generator.weight",
            "generator.bias",
        ]
        for key in redundant_keys_in_beit:
            if key in beit:
                del beit[key]

        # max_seq_len in finetuning may go beyond the length in pretraining
        if (
            self.model.pos_embed.embedding.weight.shape[0]
            != beit["pos_embed.embedding.weight"].shape[0]
        ):
            emb_shape = self.model.pos_embed.embedding.weight.shape
            ckpt_emb = beit["pos_embed.embedding.weight"].clone()
            assert emb_shape[1] == ckpt_emb.shape[1]

            ckpt_emb = ckpt_emb.unsqueeze(0).permute(0, 2, 1)
            ckpt_emb = F.interpolate(ckpt_emb, emb_shape[0], mode="nearest")
            beit["pos_embed.embedding.weight"] = ckpt_emb.permute(0, 2, 1).squeeze()

        out = self.model.load_state_dict(beit, strict=False)

        # ensure missing keys are task-specific components (token_embed, decoder, generator)
        missing_keys_prefix = ("token_embed_html", "token_embed_bbox", "decoder_html", "decoder_bbox", "generator_html", "generator_bbox")
        for key in out[0]:
            assert key.startswith(
                missing_keys_prefix
            ), f"Key {key} should be loaded from BEiT, but missing in current state dict."
        assert len(out[1]) == 0, f"Unexpected keys from BEiT: {out[1]}"


    def load_pretrained_beit_mix(self, path: Path):
        self.log.info(printer(self.device, f"Loading pretrained BEiT from {path}"))
        beit = torch.load(path, map_location=f"cuda:{self.device}", weights_only=True)
        redundant_keys_in_beit = [
            "cls_token",
            "mask_token",
            "generator.weight",
            "generator.bias",
        ]
        for key in redundant_keys_in_beit:
            if key in beit:
                del beit[key]

        # max_seq_len in finetuning may go beyond the length in pretraining
        if (self.model_html.pos_embed.embedding.weight.shape[0] != beit["pos_embed.embedding.weight"].shape[0]):
            emb_shape = self.model_html.pos_embed.embedding.weight.shape
            ckpt_emb = beit["pos_embed.embedding.weight"].clone()
            assert emb_shape[1] == ckpt_emb.shape[1]

            ckpt_emb = ckpt_emb.unsqueeze(0).permute(0, 2, 1)
            ckpt_emb = F.interpolate(ckpt_emb, emb_shape[0], mode="nearest")
            beit["pos_embed.embedding.weight"] = ckpt_emb.permute(0, 2, 1).squeeze()

        out_html = self.model_html.load_state_dict(beit, strict=False)

        # ensure missing keys are just token_embed, decoder, and generator
        missing_keys_prefix = ("token_embed", "decoder", "generator")
        for key in out_html[0]:
            assert key.startswith(
                missing_keys_prefix
            ), f"Key {key} should be loaded from BEiT, but missing in current state dict."
        assert len(out_html[1]) == 0, f"Unexpected keys from BEiT: {out_html[1]}"


        # max_seq_len in finetuning may go beyond the length in pretraining
        if (self.model_bbox.pos_embed.embedding.weight.shape[0] != beit["pos_embed.embedding.weight"].shape[0]):
            emb_shape = self.model_bbox.pos_embed.embedding.weight.shape
            ckpt_emb = beit["pos_embed.embedding.weight"].clone()
            assert emb_shape[1] == ckpt_emb.shape[1]

            ckpt_emb = ckpt_emb.unsqueeze(0).permute(0, 2, 1)
            ckpt_emb = F.interpolate(ckpt_emb, emb_shape[0], mode="nearest")
            beit["pos_embed.embedding.weight"] = ckpt_emb.permute(0, 2, 1).squeeze()

        out_bbox = self.model_bbox.load_state_dict(beit, strict=False)

        # ensure missing keys are just token_embed, decoder, and generator
        missing_keys_prefix = ("token_embed", "decoder", "generator")
        for key in out_bbox[0]:
            assert key.startswith(
                missing_keys_prefix
            ), f"Key {key} should be loaded from BEiT, but missing in current state dict."
        assert len(out_bbox[1]) == 0, f"Unexpected keys from BEiT: {out_bbox[1]}"



