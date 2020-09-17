# 路径置顶
import sys
import os
import cv2
import copy
from PIL import Image
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.getcwd())
# 导入包
import torch
import torchvision.transforms as transforms
from models.corner_network import cornernet
test_data_transforms = transforms.Compose([
    transforms.Resize([60, 190]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# 导入文件
# from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu
pwd = os.path.abspath('./')
model = cornernet()

def warp(height, width, star_points, src_img):
    # 目标尺寸
    # height = 35
    # width = 110

    pts1 = np.float32([star_points[0], star_points[1], star_points[2], star_points[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src_img, M, (width, height))
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst

model_path = os.path.join(pwd, 'models', 'corner_epoch_227_valoss_0.000117.pt')
if torch.cuda.is_available():
    model_state = torch.load(model_path)
else:
    model_state = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_state['model_state_dict'])
if torch.cuda.is_available():
    model = model.cuda()
print('loaded %s' % model_path)
model.eval()  # 模型验证模式
imgpath = os.path.join(pwd, 'data', 'corner', 'test')
output_path = os.path.join(pwd, 'inference', 'corner')
mean_width = 190
mean_height = 60
if not os.path.exists(output_path):
    os.mkdir(output_path)
for i in os.listdir(imgpath):
    if 'txt' in i:
        continue
    imgpth = os.path.join(imgpath, i)
    img = cv2.imread(imgpth)
    img_ori = copy.deepcopy(img)
    h, w, c = img.shape
    txtfile = imgpth.replace('.jpg', '.txt').replace('.png', '.txt')
    with open(txtfile, 'r') as obj:
        coords = obj.readline().strip()
        coords = [float(a) for a in coords.split(' ')]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).convert('RGB')
    img = test_data_transforms(img)
    if torch.cuda.is_available():
        img = img.unsqueeze(0).cuda()
    else:
        img = img.unsqueeze(0)
    output = model(img)
    luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y = tuple(output.detach().cpu().numpy()[0])
    luc_xo, luc_yo, ldc_xo, ldc_yo, rdc_xo, rdc_yo, ruc_xo, ruc_yo = coords
    luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y = int(luc_x*w), int(luc_y*h),int(ldc_x*w),\
        int(ldc_y*h), int(rdc_x*w), int(rdc_y*h), int(ruc_x*w), int(ruc_y*h)
    luc_xo, luc_yo, ldc_xo, ldc_yo, rdc_xo, rdc_yo, ruc_xo, ruc_yo = int(luc_xo * w), int(luc_yo * h), int(ldc_xo * w), \
        int(ldc_yo * h), int(rdc_xo * w), int(rdc_yo * h), int(ruc_xo * w), int(ruc_yo * h)
    cv2.circle(img_ori, (rdc_x, rdc_y), 3, [255, 0, 255], 1)
    cv2.circle(img_ori, (ldc_x, ldc_y), 3, [255, 0, 255], 1)
    cv2.circle(img_ori, (luc_x, luc_y), 3, [255, 0, 255], 1)
    cv2.circle(img_ori, (ruc_x, ruc_y), 3, [255, 0, 255], 1)

    cv2.circle(img_ori, (rdc_xo, rdc_yo), 3, [255, 0, 0], 1)
    cv2.circle(img_ori, (ldc_xo, ldc_yo), 3, [255, 0, 0], 1)
    cv2.circle(img_ori, (luc_xo, luc_yo), 3, [255, 0, 0], 1)
    cv2.circle(img_ori, (ruc_xo, ruc_yo), 3, [255, 0, 0], 1)
    print('predict: ', luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y)
    print('label: ', luc_xo, luc_yo, ldc_xo, ldc_yo, rdc_xo, rdc_yo, ruc_xo, ruc_yo)
    start_points = [[luc_x, luc_y], [ruc_x, ruc_y], [ldc_x, ldc_y], [rdc_x, rdc_y]]
    warp_img = warp(mean_height, mean_width, start_points, img_ori)
    cv2.imwrite(os.path.join(output_path, i), img_ori)
    cv2.imwrite(os.path.join(output_path, i.replace('.jpg','_warp.jpg')), warp_img)
    # cv2.imshow('i', img_ori)
    # cv2.waitKey(0)
    # cv2.destroyWindow('i')

