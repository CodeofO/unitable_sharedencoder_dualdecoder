import os
import json
from vocab.constant import CELL_SPECIAL
from PIL import Image, ImageDraw
import os
import GPUtil
from utils import html_table_template, build_table_from_html_and_cell, build_table_from_html
import shutil
import base64


def check_gpu():
    gpus = GPUtil.getGPUs()
    gpu_list = []
    for gpu in gpus:
        gpu_list.append((
            gpu.id,
            gpu.name,
            gpu.memoryTotal,
            gpu.memoryUsed,
            gpu.memoryUtil,
            gpu.load
        ))
    return gpu_list


def change_to_unitable():
    # 현재 작업 디렉토리 경로를 가져옵니다.
    current_path = os.getcwd()
    
    # 현재 디렉토리의 마지막 폴더명을 가져옵니다.
    last_dir = os.path.basename(current_path)
    
    # 마지막 폴더명이 'unitable'이 아닌 경우
    if last_dir != 'unitable':
        # 'unitable' 폴더의 경로를 생성합니다.
        target_path = os.path.join(current_path, 'unitable')
        
        # 'unitable' 폴더가 존재하고 디렉토리인지 확인합니다.
        if os.path.exists(target_path) and os.path.isdir(target_path):
            os.chdir(target_path)
            print(f"✅ Path : 디렉토리를 '{target_path}' 로 변경했습니다.")
        else:
            print(f"✅ Path : 'unitable' 폴더가 현재 경로에 존재하지 않습니다: {current_path}")
    else:
        print("✅ Path : 현재 디렉토리가 'unitable' 이므로 변경하지 않습니다.")


def build_anno(gt_html, gt_cell):
    anno_html = []
    idx = 0
    while idx < len(gt_html):
        if "[" in gt_html[idx]:
            assert idx + 1 < len(gt_html)
            assert gt_html[idx + 1] == "]</td>"
            anno_html.append(gt_html[idx] + "]</td>")
            idx = idx + 2
        else:
            anno_html.append(gt_html[idx])
            idx = idx + 1

    anno_cell = []
    for txt in gt_cell:
        for black in CELL_SPECIAL:
            txt = txt.replace(black, "")
        anno_cell.append(txt)

    gt_code_html = "".join(build_table_from_html_and_cell(anno_html, anno_cell))
    return gt_code_html


def visualize_bboxes(image: Image.Image, 
                     cfg,
                     obj,
                     pred_bboxes: list[tuple[int, int, int, int]], 
                     gt_bboxes: list[tuple[int, int, int, int]], 
                     ) -> None:
    
    # 이미지 복사본 생성
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # 예측 BBox를 초록색으로 그리기
    for bbox in pred_bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    
    # 정답 BBox를 파란색으로 그리기
    if cfg['mode'] == 'evaluation':
        for bbox in gt_bboxes:
            draw.rectangle(bbox, outline="green", width=2)
    
    # 이미지 저장
    save_path = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'bbox_image', obj['filename'])
    img.save(save_path)

def visualize_bboxes_batch(
                     cfg,
                     image_path,
                     pred_bboxes: list[tuple[int, int, int, int]], 
                     gt_bboxes: list[tuple[int, int, int, int]], 
                     ) -> None:
    
    # 원본 이미지 로드
    image = Image.open(image_path).convert("RGB")
    
    # GT용 이미지와 Pred용 이미지 각각 복사
    gt_img = image.copy()
    pred_img = image.copy()
    
    gt_draw = ImageDraw.Draw(gt_img)
    pred_draw = ImageDraw.Draw(pred_img)
    
    # GT 이미지에는 정답 BBox만 그리기
    if cfg['mode'] == 'evaluation':
        for bbox in gt_bboxes:
            gt_draw.rectangle(bbox, outline="green", width=2)
    
    # Pred 이미지에는 예측 BBox만 그리기
    for bbox in pred_bboxes:
        pred_draw.rectangle(bbox, outline="red", width=2)
    
    # 두 이미지를 가로로 연결할 새 이미지 생성
    total_width = gt_img.width * 2
    max_height = gt_img.height
    
    combined_img = Image.new('RGB', (total_width, max_height))
    
    # 왼쪽에 GT 이미지, 오른쪽에 Pred 이미지 붙이기
    combined_img.paste(gt_img, (0, 0))
    combined_img.paste(pred_img, (gt_img.width, 0))
    
    # 이미지 저장
    save_path = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'bbox_image', os.path.basename(image_path))
    combined_img.save(save_path)



# def visualize_bboxes(image: Image.Image, 
#                      cfg,
#                      obj,
#                      pred_bboxes: list[tuple[int, int, int, int]], 
#                      gt_bboxes: list[tuple[int, int, int, int]], 
#                      ) -> None:
    
#     # 이미지 복사본 생성
#     img = image.copy()
#     draw = ImageDraw.Draw(img)
    
#     # 예측 BBox를 초록색으로 그리기
#     for bbox in pred_bboxes:
#         draw.rectangle(bbox, outline="red", width=2)
    
#     # 정답 BBox를 파란색으로 그리기
#     if cfg['mode'] == 'evaluation':
#         for bbox in gt_bboxes:
#             draw.rectangle(bbox, outline="green", width=2)
    
#     # 이미지 저장
#     save_path = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'bbox_image', obj['filename'])
#     img.save(save_path)




def save_pred_anno(cfg, obj, pred_html, pred_cell, pred_bbox, pred_code_html):

    try:
        labelme_format = dict()
        labelme_format['version'] = "5.5.0"
        labelme_format['flags'] = {}
        labelme_format['imagePath'] = os.path.join(cfg['data_dir'], 'test', obj['filename'])
        labelme_format['shapes'] = []
        labelme_format['imageData'] = None
        labelme_format['imageHeight'] = 1
        labelme_format['imageWidth'] = 1

        with open(labelme_format['imagePath'], 'rb') as image_file:
            labelme_format['imageData'] = base64.b64encode(image_file.read()).decode('utf-8')

        for cell_id in range(len(pred_bbox)):

            only_one_cell_dict = dict()

            x1, y1, x2, y2 = pred_bbox[cell_id]
            cell = pred_cell[cell_id]

            only_one_cell_dict['label'] = f"{cell}"
            only_one_cell_dict['points'] = [[x1, y1], [x2, y2]]
            only_one_cell_dict['group_id'] = None
            only_one_cell_dict['description'] = ""
            only_one_cell_dict['shape_type'] = "rectangle"
            only_one_cell_dict['mask'] = None

            labelme_format['shapes'].append(only_one_cell_dict)
    except:
        pass


    save_dir = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'cell_labellme_format')
    path_json = os.path.join(save_dir, obj['filename'][:-4] + '.json')

    save_dir = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure')
    path_html = os.path.join(save_dir, obj['filename'][:-4] + '.html')

    save_dir = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure_filled')
    path_html_filled = os.path.join(save_dir, obj['filename'][:-4] + '.html')

    save_dir = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure_pred_html')
    path_pred_html = os.path.join(save_dir, obj['filename'][:-4] + '.txt')
    
    
    table_structure_html = build_table_from_html(pred_html)
    pred_code_html = html_table_template("".join(table_structure_html))

    try:
        table_structure_html_full = build_table_from_html_and_cell(pred_html, pred_cell)     
        pred_code_html_filled = html_table_template("".join(table_structure_html_full))
    except:
        pass

    try:
        with open(path_json, 'w') as f:
            json.dump(labelme_format, f, indent=4)
    except:
        pass

    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(pred_code_html)
    try:
        with open(path_html_filled, 'w', encoding='utf-8') as f:
            f.write(pred_code_html_filled)
    except:
        pass
    with open(path_pred_html, 'w') as f:
            for item in pred_html:
                f.write(f"{item}\n")




def save_pred_anno_batch(cfg, filename, pred_html):


    save_dir = os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure_pred_html')
    path_pred_html = os.path.join(save_dir, os.path.basename(filename)[:-4] + '.html')
    
    with open(path_pred_html, 'w', encoding='utf-8') as f:
        f.write(pred_html)  # 직접 문자열 저장









def extract_number(file_path):
    """
    파일 경로에서 숫자 부분을 추출하여 정수로 반환하는 함수
    """
    # 파일 이름 추출
    base_name = os.path.basename(file_path)
    # 확장자 제거
    name_without_ext = os.path.splitext(base_name)[0]
    # 정수로 변환
    try:
        return int(name_without_ext)
    except ValueError:
        # 숫자로 변환할 수 없는 경우, 큰 수를 반환하여 정렬 시 뒤로 밀기
        return float('inf')


def set_folder(cfg):
    if os.path.exists(os.path.join(cfg['save_dir'], cfg['dataset_name'])):
        shutil.rmtree(os.path.join(cfg['save_dir'], cfg['dataset_name']))
    
    os.makedirs(os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'cell_labellme_format'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure_filled'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_dir'], cfg['dataset_name'], 'pred_html', 'structure_pred_html'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_dir'], cfg['dataset_name'], 'bbox_image'), exist_ok=True)
   
    print(f'✅ make save directory : {os.path.join(cfg["save_dir"], cfg["dataset_name"])}')


def count_cells_from_list(html_list):
    cell_count = 0
    i = 0
    while i < len(html_list):
        # <td> 태그를 찾기
        if '<td' in html_list[i]:
            colspan = 1
            rowspan = 1
            
            # 다음 항목에서 colspan, rowspan이 있는지 확인하기 전에 리스트의 범위를 체크
            if i + 1 < len(html_list):
                if 'colspan' in html_list[i + 1]:
                    colspan_str = html_list[i + 1].split('=')[1].replace('"', '')
                    colspan = int(colspan_str)
                if 'rowspan' in html_list[i + 1]:
                    rowspan_str = html_list[i + 1].split('=')[1].replace('"', '')
                    rowspan = int(rowspan_str)
            
            # 셀 개수는 rowspan * colspan
            cell_count += rowspan * colspan
        
        i += 1  # 다음 요소로 이동

    return cell_count
