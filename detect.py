import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from copy import deepcopy
import cv2
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.corner_network import cornernet
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
# from crnn import utils
# import crnn.models.crnn as crnn
import CRNN_Chinese_Characters_Rec.lib.utils.utils as utils
import CRNN_Chinese_Characters_Rec.lib.models.crnn as crnn
import CRNN_Chinese_Characters_Rec.lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict

pwd = os.path.abspath('./')

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

data_transforms = transforms.Compose([
    transforms.Resize([32, 190]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
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

def recognition(warp_img, crnn_model, converter):
    img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (190, 32))
    img = np.reshape(img, (32, 190, 1))
    img = img.astype(np.float32)
    img = (img / 255. - 0.588) / 0.193
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    if torch.cuda.is_available():
        img = img.cuda()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    preds = crnn_model(img)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

def corner_detection(corner_model, imgple, im0, xyxy, colors, cls, c1, c2):
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
    luc_xo, luc_yo, ldc_xo, ldc_yo, rdc_xo, rdc_yo, ruc_xo, ruc_yo = int(luc_x * w_ple), int(luc_y * h_ple), \
                                                                     int(ldc_x * w_ple), int(ldc_y * h_ple), \
                                                                     int(rdc_x * w_ple), int(rdc_y * h_ple), \
                                                                     int(ruc_x * w_ple), int(ruc_y * h_ple)
    luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y = int(luc_x * w_ple + c1[0]), int(luc_y * h_ple + c1[1]), \
                                                             int(ldc_x * w_ple + c1[0]), int(ldc_y * h_ple + c1[1]), \
                                                             int(rdc_x * w_ple + c1[0]), int(rdc_y * h_ple + c1[1]), \
                                                             int(ruc_x * w_ple + c1[0]), int(ruc_y * h_ple + c1[1])
    cv2.circle(im0, (rdc_x, rdc_y), 3, [255, 0, 0], 1)
    cv2.circle(im0, (ldc_x, ldc_y), 3, [255, 0, 0], 1)
    cv2.circle(im0, (luc_x, luc_y), 3, [255, 0, 0], 1)
    cv2.circle(im0, (ruc_x, ruc_y), 3, [255, 0, 0], 1)
    mean_width = 190  # np.mean(widths)
    mean_height = 60  # np.mean(height)
    plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
    start_points = [[luc_xo, luc_yo], [ruc_xo, ruc_yo], [ldc_xo, ldc_yo], [rdc_xo, rdc_yo]]
    warp_img = warp(mean_height, mean_width, start_points, imgple_ori)
    return warp_img, im0

chinesechar = {'皖': 0, '沪': 1, '津': 2, '渝': 3, '冀': 4, '晋': 5, '蒙': 6, '辽': 7, '吉': 8, '黑': 9, '苏': 10, '浙': 11, \
               '京': 12, '闽': 13, '赣': 14, '鲁': 15, '豫': 16, '鄂': 17, '湘': 18, '粤': 19, '桂': 20, '琼': 21, '川': 22, \
               '贵': 23, '云': 24, '西': 25, '陕': 26, '甘': 27, '青': 28, '宁': 29, '新': 30}
chinesechar_reverse = {0: '皖', 1: '沪', 2: '津', 3: '渝', 4: '冀', 5: '晋', 6: '蒙', 7: '辽', 8: '吉', 9: '黑', 10: '苏', \
                      11: '浙', 12: '京', 13: '闽', 14: '赣', 15: '鲁', 16: '豫', 17: '鄂', 18: '湘', 19: '粤', 20: '桂', \
                      21: '琼', 22: '川', 23: '贵', 24: '云', 25: '西', 26: '陕', 27: '甘', 28: '青', 29: '宁', 30: '新'}

alphaenglish = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'j': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, \
               'p': 13, 'q': 14, 'r': 15, 's': 16, 't': 17, 'u': 18, 'v': 19, 'w': 20, 'x': 21, 'y': 22, 'z': 23, '0': 24, \
               '1': 25, '2': 26, '3': 27, '4': 28, '5': 29, '6': 30, '7': 31, '8': 32, '9': 33}
alphaenglish_reverse = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'j', 9: 'k', 10: 'l', 11: 'm', \
                        12: 'n', 13: 'p', 14: 'q', 15: 'r', 16: 's', 17: 't', 18: 'u', 19: 'v', 20: 'w', 21: 'x', 22: 'y', \
                        23: 'z', 24: '0', 25: '1', 26: '2', 27: '3', 28: '4', 29: '5', 30: '6', 31: '7', 32: '8', 33: '9'}

classes = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', \
           '粤', '桂', '琼', '川', '贵', '云', '西', '陕', '甘', '青', '宁', '新', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', \
           'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', \
           '6', '7', '8', '9']

def detect(save_img=False):

    plate_detect_model = os.path.join(pwd, 'models', 'best.pt')
    corner_detect_model = os.path.join(pwd, 'models', 'corner_epoch_227_valoss_0.000117.pt')

    corner_model = cornernet()
    corner_model.eval()  # 验证模式

    out, weights, view_img, save_txt, imgsz = \
        opt.output, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    generate_crnn_trainset = False
    if generate_crnn_trainset:
        recognition_crnn_model = os.path.join(pwd, 'models', 'checkpoint_123_acc_0.9940.pth')
        source = os.path.join(pwd, 'CCPD2019', 'ccpd_base')
        save_path_train = os.path.join(pwd,'data','crnn','warpimg')
        save_path_test = os.path.join(pwd,'data','crnn','test')
        ftrain = open(os.path.join(pwd, 'CRNN_Chinese_Characters_Rec', 'lib', 'train_own.txt'), 'w')
        ftest = open(os.path.join(pwd, 'CRNN_Chinese_Characters_Rec', 'lib', 'test_own.txt'), 'w')
        if not os.path.exists(save_path_train):
            os.makedirs(save_path_train)
        if not os.path.exists(save_path_test):
            os.makedirs(save_path_test)
        all_length = len(os.listdir(source))
    else:
        source = os.path.join(pwd, 'data', 'stage1', 'test')
        # source = r'/data/tmp'
    print('image\' length is: ', len(os.listdir(source)))
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(plate_detect_model, map_location = device)  # load FP32 model
    corner_model_state = torch.load(corner_detect_model, map_location=device)
    corner_model.load_state_dict(corner_model_state['model_state_dict'])
    print('loaded the plate detected model：', plate_detect_model)
    print('loaded the corner detected model：', corner_detect_model)
    crnn_train_data = r''

    alphabet ='0123456789abcdefghjklmnpqrstuvwxyz云京冀吉宁川新晋桂沪津浙渝湘琼甘皖粤苏蒙西豫贵赣辽鄂闽陕青鲁黑'
    converter = utils.strLabelConverter(alphabet)
    nclass = len(alphabet) + 1
    # crnn_model = crnn.CRNN(32, 1, nclass, 190)
    if not generate_crnn_trainset:
        crnn_model = crnn.CRNN(32, 1, nclass, 256)
        checkpoint = torch.load(recognition_crnn_model)
        print('loaded the corner recognition model：', recognition_crnn_model)
        if 'state_dict' in checkpoint.keys():
            crnn_model.load_state_dict(checkpoint['state_dict'])
        else:
            crnn_model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
        corner_model = corner_model.cuda()
        if not generate_crnn_trainset:
            crnn_model = crnn_model.cuda()
            crnn_model.eval()

    corner_model.eval()
    model.eval()

    imgsz = check_img_size(imgsz, s = model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    count = 0
    n_correct = 0
    false_img = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        count += 1
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    imgple = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label='', color=colors[int(cls)], line_thickness=3)
                    warp_img, im0 = corner_detection(corner_model, imgple, im0, xyxy, colors, cls, c1, c2)
                    if generate_crnn_trainset:
                        name = save_path.split(os.sep)[-1]
                        string = name.replace('.jpg', '').replace('.png', '').replace('&','_').replace('-', '_').split('_')
                        platestring = chinesechar_reverse[int(string[15])]
                        for ij in string[16:-2]:
                            platestring += alphaenglish_reverse[int(ij)]
                        platestring = platestring.upper()
                        if count<3800:
                            ftest.write(os.path.join(save_path_test, name) + ' ' + platestring+'\n')
                            cv2.imwrite(os.path.join(save_path_test, name), warp_img)
                        else:
                            ftrain.write(os.path.join(save_path_train, name) + ' ' + platestring+'\n')
                            cv2.imwrite(os.path.join(save_path_train, name), warp_img)
                    else:
                        try:
                            name = save_path.split(os.sep)[-1].replace('.jpg', '').replace('.png', '').replace('&', '_').replace('-', '_').split('_')
                            platestring = chinesechar_reverse[int(name[15])]
                            for ij in name[16:-2]:
                                platestring += alphaenglish_reverse[int(ij)]
                            platestring = platestring.upper()
                        except:
                            pass
                        #
                        sim_pred = recognition(warp_img, crnn_model, converter)
                        sim_pred = sim_pred.upper()
                        #
                        try:
                            if sim_pred==platestring:
                                print()
                                n_correct+=1
                            else:
                                false_img.append(platestring)
                                cv2.imwrite(save_path, warp_img)
                            # print('gt: ', platestring, '，predict: ', sim_pred, ' ', sim_pred == platestring, ' ', n_correct,
                            #       ' ', count)
                        except:
                            pass
                        #
                        sim_pred = sim_pred[:2]+' '+sim_pred[2:]
                        print('predict: ', sim_pred)
                        img_PIL = Image.fromarray(im0).convert('RGB')
                        font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 50)#('simsun.ttc', 50)
                        # 字体颜色
                        fillColor = (255, 0, 0)
                        draw = ImageDraw.Draw(img_PIL)
                        draw.text((c1[0], c1[1] - 39,), sim_pred, font=font, fill=fillColor)
                        im0 = np.asarray(img_PIL)
            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img and (not generate_crnn_trainset):
                if dataset.mode == 'images':
                    # pass
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    if generate_crnn_trainset:
        print('生成的CRNN训练集数量：{}，测试集数量是：{}'.format(len(os.listdir(save_path_train)), len(os.listdir(save_path_test))))
    else:
        try:
            print('n_correct: {}，count: {}，test accuray: '.format(n_correct, count, n_correct/count))
            print('false_image', false_img)
        except:
            pass
    if generate_crnn_trainset:
        ftrain.close()
        ftest.close()
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=os.path.join(pwd, 'models', 'best.pt'), help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
