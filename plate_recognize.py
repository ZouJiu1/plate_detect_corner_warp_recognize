import torch
import random
import os
import time
import numpy as np
from PIL import Image
import cv2
import shutil
from pathlib import Path
from utils.datasets import LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from models.experimental import attempt_load
from models.corner_network import cornernet
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
import torchvision.transforms as transforms
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pwd = os.path.abspath('./')
test_data_transforms = transforms.Compose([
    transforms.Resize([60, 190]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def warp(height, width, star_points, src_img):
    # 目标尺寸
    # height = 35
    # width = 110

    pts1 = np.float32([star_points[0], star_points[1], star_points[2], star_points[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src_img, M, (width, height))
    return dst

plate_detect_model = os.path.join(pwd, 'runs', 'exp9', 'weights', 'best.pt')
corner_detect_model = os.path.join(pwd, 'models', 'corner_32', 'corner_epoch_97_valoss_0.000223.pt')
img_dir = os.path.join(pwd, 'data', 'stage1', 'test')
out = os.path.join(pwd, 'inference', 'output')

if os.path.exists(out):
    shutil.rmtree(out)  # delete output folder

corner_model = cornernet()
corner_model.eval()  # 验证模式
os.makedirs(out)  # make new output folder
device = select_device('0') #'cuda device, i.e. 0 or 0,1,2,3 or cpu' type: str
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
plate_model = attempt_load(plate_detect_model, map_location = device)  # load FP32 model
corner_model_state = torch.load(corner_detect_model, map_location = device)
corner_model.load_state_dict(corner_model_state['model_state_dict'])
print('loaded the plate detected model：', plate_detect_model)
print('loaded the corner detected model：', corner_detect_model)
if half:
    plate_model.half()  # to FP16
dataset = LoadImages(img_dir, img_size=640)
# Get names and colors
names = plate_model.module.names if hasattr(plate_model, 'module') else plate_model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# Run inference
t0 = time.time()
img = torch.zeros((1, 3, 640, 640), device=device)  # init img
_ = plate_model(img.half() if half else img) if device.type != 'cpu' else None  # run once
count = 0
if torch.cuda.is_available():
    plate_model = plate_model.cuda()
    corner_model = corner_model.cuda()
plate_model.eval()
corner_model.eval()
mean_width = 190 # np.mean(widths)
mean_height = 60 # np.mean(height)

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = plate_model(img, augment = False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=False, agnostic=False)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s

        save_path = str(Path(out) / Path(p).name)
        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            coordes = []
            det_count = 1
            for *xyxy, conf, cls in reversed(det):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                imgple = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                imgple_ori = deepcopy(imgple)
                imgple = cv2.cvtColor(imgple, cv2.COLOR_BGR2RGB)
                h_ple, w_ple, _ = imgple.shape
                imgple = Image.fromarray(imgple).convert('RGB')
                imgple = test_data_transforms(imgple)
                if torch.cuda.is_available():
                    imgple = imgple.unsqueeze(0).cuda()
                else:
                    imgple = imgple.unsqueeze(0)
                output = corner_model(imgple)
                luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y = tuple(output.detach().cpu().numpy()[0])
                luc_xo, luc_yo, ldc_xo, ldc_yo, rdc_xo, rdc_yo, ruc_xo, ruc_yo = int(luc_x * w_ple), int(luc_y * h_ple),\
                                                                                 int(ldc_x * w_ple), int(ldc_y * h_ple), \
                                                                                 int(rdc_x * w_ple), int(rdc_y * h_ple), \
                                                                                 int(ruc_x * w_ple), int(ruc_y * h_ple)
                luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y = int(luc_x * w_ple + c1[0]), int(luc_y * h_ple+c1[1]), \
                                                                         int(ldc_x * w_ple+ c1[0]), int(ldc_y * h_ple+c1[1]), \
                                                                         int(rdc_x * w_ple+ c1[0]), int(rdc_y * h_ple+c1[1]), \
                                                                         int(ruc_x * w_ple+ c1[0]), int(ruc_y * h_ple+c1[1])
                cv2.circle(im0, (rdc_x, rdc_y), 3, [255, 0, 0], 1)
                cv2.circle(im0, (ldc_x, ldc_y), 3, [255, 0, 0], 1)
                cv2.circle(im0, (luc_x, luc_y), 3, [255, 0, 0], 1)
                cv2.circle(im0, (ruc_x, ruc_y), 3, [255, 0, 0], 1)

                plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                start_points = [[luc_xo, luc_yo], [ruc_xo, ruc_yo], [ldc_xo, ldc_yo], [rdc_xo, rdc_yo]]
                warp_img = warp(mean_height, mean_width, start_points, imgple_ori)
                warp_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
                
                warp_img = Image.fromarray(warp_img).convert('RGB')
                warp_img.save(save_path.replace('.jpg', '_warp%d.jpg'%det_count).replace('.png','_warp%d.jpg'%det_count))
                det_count += 1
        print('%sDone. (%.3fs)' % (s, t2 - t1))
        cv2.imwrite(save_path, im0)
print('Done. (%.3fs)' % (time.time() - t0))
