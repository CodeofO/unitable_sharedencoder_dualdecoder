import torch
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint


def compute_coco_map(file):
    coco_pred = list()
    coco_gt = list()
    for _, obj in file.items():
        tmp_pred = {
            "boxes": torch.tensor(obj["pred"], device=0),
            "labels": torch.tensor([0] * len(obj["pred"]), device=0),
            "scores": torch.tensor([0.999] * len(obj["pred"]), device=0),
        }

        tmp_gt = {
            "boxes": torch.tensor(obj["gt"], device=0),
            "labels": torch.tensor([0] * len(obj["gt"]), device=0),
        }

        coco_pred.append(tmp_pred)
        coco_gt.append(tmp_gt)

    metric = MeanAveragePrecision(
        iou_type="bbox",
        max_detection_thresholds=[1, 10, 1000],
        backend="faster_coco_eval",
    )
    metric.update(coco_pred, coco_gt)
    pprint(metric.compute())


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="mAP Computation")

    parser.add_argument("-f", "--file", help="path to html table results in json file")
    args = parser.parse_args()


    results_file = args.file
    with open(results_file, "r") as f:
        results_json = json.load(f)

    compute_coco_map(results_json)



def convert_to_coco_format(pred_bbox, gt_bbox):
    # COCO 포맷을 맞추기 위해 리스트를 변환
    coco_pred = []
    coco_gt = []

    # 예측 바운딩 박스에 필요한 정보: boxes, labels, scores
    pred = {
        "boxes": torch.tensor(pred_bbox, dtype=torch.float32),
        "labels": torch.tensor([0] * len(pred_bbox), dtype=torch.int64),  # 라벨 0으로 설정
        "scores": torch.tensor([0.999] * len(pred_bbox), dtype=torch.float32)  # 점수는 임의로 0.999로 설정
    }
    coco_pred.append(pred)

    # 실제 바운딩 박스에 필요한 정보: boxes, labels
    gt = {
        "boxes": torch.tensor(gt_bbox, dtype=torch.float32),
        "labels": torch.tensor([0] * len(gt_bbox), dtype=torch.int64)  # 라벨 0으로 설정
    }
    coco_gt.append(gt)

    return coco_pred, coco_gt


def compute_map_eval(pred_bbox, gt_bbox):    
    # 바운딩 박스를 COCO 포맷으로 변환
    coco_pred, coco_gt = convert_to_coco_format(pred_bbox, gt_bbox)

    # MeanAveragePrecision 객체 생성 (IoU 임계값 0.5와 0.75 설정)
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75])

    # 예측된 bbox와 실제 bbox를 업데이트
    metric.update(coco_pred, coco_gt)

    # mAP, AP50, AP75 계산
    results = metric.compute()

    # AP50, mAP, AP75 출력
    ap50 = results['map_50'].item()  # tensor -> float
    ap75 = results['map_75'].item()  # tensor -> float
    map_value = results['map'].item()  # tensor -> float

    # 결과 반환
    return {
        "AP50": ap50,
        "AP75": ap75,
        "mAP": map_value
    }




def convert_to_coco_format_ddp(pred_bbox, gt_bbox):
    """
    pred_bbox, gt_bbox: list of [x1, y1, x2, y2]
    
    Returns:
        preds: [{"boxes": Tensor[N, 4], "scores": Tensor[N], "labels": Tensor[N]}]
        targets: [{"boxes": Tensor[M, 4], "labels": Tensor[M]}]
    """
    # 예측 바운딩 박스에 필요한 정보: boxes, labels, scores
    preds = [{
        "boxes": torch.tensor(pred_bbox, dtype=torch.float32, device="cuda"),
        "labels": torch.zeros(len(pred_bbox), dtype=torch.int64, device="cuda"),  # 단일 클래스라면 0으로 설정
        "scores": torch.ones(len(pred_bbox), dtype=torch.float32, device="cuda")  # 실제 점수가 있다면 해당 값을 사용
    }]
    
    # 실제 바운딩 박스에 필요한 정보: boxes, labels
    targets = [{
        "boxes": torch.tensor(gt_bbox, dtype=torch.float32, device="cuda"),
        "labels": torch.zeros(len(gt_bbox), dtype=torch.int64, device="cuda")  # 단일 클래스라면 0으로 설정
    }]
    
    return preds, targets



def compute_map_eval_ddp(pred_bbox, gt_bbox):    
    """
    pred_bbox: list of [x1, y1, x2, y2]
    gt_bbox: list of [x1, y1, x2, y2]
    
    Returns:
        dict: {"AP50": float, "AP75": float, "mAP": float}
    """
    # 바운딩 박스를 COCO 포맷으로 변환
    preds, targets = convert_to_coco_format_ddp(pred_bbox, gt_bbox)
    
    # MeanAveragePrecision 객체 생성 및 GPU로 이동
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5, 0.75]).to('cuda')
    
    # 예측값 및 타겟값 업데이트
    metric.update(preds, targets)
    
    # mAP, AP50, AP75 계산
    results = metric.compute()
    
    # AP50, mAP, AP75 반환
    return {
        "AP50": results['map_50'].item(),  # tensor -> float
        "AP75": results['map_75'].item(),  # tensor -> float
        "mAP": results['map'].item(),      # tensor -> float
    }
