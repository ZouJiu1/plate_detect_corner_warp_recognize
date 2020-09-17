import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import shutil
import copy
import time
seq = os.sep
pwd = os.path.abspath(r'.%s'%seq)

def write_train_val(file, stage, name):
    with open(os.path.join(pwd, 'data', stage, '%s.txt'%name), 'w') as obj:
        for i in file:
            obj.write(i)
            obj.write('\n')

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

print('len(classes): ', classes.__len__())

def warp(height, width, star_points, src_img):
    # 目标尺寸
    # height = 35
    # width = 110

    pts1 = np.float32([star_points[0], star_points[1], star_points[2], star_points[3]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(src_img, M, (width, height))
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
    # cv2.destroyWindow('img')
    return dst

def cal_txt(shape, xmin, ymin, xmax, ymax, name):
    h, w, c = shape
    center_x = (xmin+xmax)/2
    center_y = (ymin+ymax)/2
    center_x = center_x/w
    center_y = center_y/h
    item_h = (ymax-ymin)/h
    item_w = (xmax-xmin)/w
    cor = np.array([center_x, center_y, item_w, item_h])
    if (np.any(cor > 1)) or (np.any(cor < 0)):
        return -1, -1, -1, -1
    return center_x,center_y,item_w,item_h

def cvshow(img, xmin, ymin ,xmax, ymax, rdc_x,rdc_y, ldc_x, ldc_y, ruc_x, ruc_y, luc_x, luc_y, platestring):
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [255, 0, 0], 1)
    cv2.circle(img, (rdc_x, rdc_y), 3, [255, 0, 255], 1)
    cv2.circle(img, (ldc_x, ldc_y), 3, [255, 0, 255], 1)
    cv2.circle(img, (luc_x, luc_y), 3, [255, 0, 255], 1)
    cv2.circle(img, (ruc_x, ruc_y), 3, [255, 0, 255], 1)
    img_PIL = Image.fromarray(img).convert('RGB')
    font = ImageFont.truetype('simsun.ttc', 50)
    # 字体颜色
    fillColor = (255, 0, 0)
    draw = ImageDraw.Draw(img_PIL)
    draw.text((xmin, ymin-39 ,), platestring, font=font, fill=fillColor)
    img = np.asarray(img_PIL)
    cv2.imshow('i', img)
    cv2.waitKey(0)
    cv2.destroyWindow('i')

def rectangle_save(img, xmin, ymin, xmax, ymax, rdc_x, rdc_y, ldc_x, ldc_y, ruc_x, \
                   ruc_y, luc_x, luc_y, rectangle, ori_i,rectangle_label):
    rectangle_img = img[ymin:ymax + 1, xmin:xmax + 1, :]
    h, w, c = rectangle_img.shape
    luc_x, luc_y = (luc_x-xmin)/w, (luc_y-ymin)/h
    ldc_x, ldc_y = (ldc_x-xmin)/w, (ldc_y-ymin)/h
    rdc_x, rdc_y = (rdc_x-xmin)/w, (rdc_y-ymin)/h
    ruc_x, ruc_y = (ruc_x-xmin)/w, (ruc_y-ymin)/h
    #
    # cv2.circle(rectangle_img, (int(rdc_x*w), int(rdc_y*h)), 3, [255, 0, 255], 1)
    # cv2.circle(rectangle_img, (int(ldc_x*w), int(ldc_y*h)), 3, [255, 0, 255], 1)
    # cv2.imshow('i', rectangle_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('i')

    corner = [luc_x, luc_y, ldc_x, ldc_y, rdc_x, rdc_y, ruc_x, ruc_y]
    cor = np.array(corner)
    if (np.any(cor>1)) or (np.any(cor<0)):
        return -1
    # rectangle_img = img[ymin:ymax + 1, xmin:xmax + 1, :]
    cv2.imwrite(os.path.join(rectangle, ori_i), rectangle_img)
    with open(os.path.join(rectangle_label, ori_i.replace('.jpg','.txt')), 'w') as obj:
        obj.write(' '.join([str(a) for a in corner]))

pwd = os.path.abspath('./')

path = os.path.join(pwd, 'data', 'CCPD_part')
widths = []
heights = []
# for root_dir, dir, file in os.walk(path):
#     if len(file)==0:
#         continue
#     if 'base' not in root_dir:
#         continue
#     count=0
#     for i in file:
#         count+=1
#         if count%999==0:
#             print(count)
#         # img = cv2.imread(os.path.join(root_dir, i))
#         i = i.replace('.jpg', '').replace('.png', '').replace('&', '_').replace('-', '_').split('_')
#         angle = [int(i[1]), int(i[2])]
#         xmin, ymin, xmax, ymax = int(i[3]), int(i[4]), int(i[5]), int(i[6])
#         rdc_x, rdc_y, ldc_x, ldc_y, luc_x, luc_y, ruc_x, ruc_y = int(i[7]), int(i[8]), int(i[9]), int(i[10]), \
#                                                                  int(i[11]), int(i[12]), int(i[13]), int(i[14])
#
#         # shape = img.shape
#         # start_points = [[luc_x, luc_y], [ruc_x, ruc_y], [ldc_x, ldc_y], [rdc_x, rdc_y]]
#         # warp(70, 160, start_points, img)
#
#         width = (np.sqrt(np.square(rdc_x-ldc_x)+np.square(rdc_y-ldc_y))+\
#                  np.sqrt(np.square(ruc_x-luc_x)+np.square(ruc_y-luc_y)))/2
#         height = (np.sqrt(np.square(rdc_x-ruc_x)+np.square(rdc_y-ruc_y))+\
#                  np.sqrt(np.square(ldc_x-luc_x)+np.square(ldc_y-luc_y)))/2
#         if (abs(np.mean(widths)- width) > 10) or (abs(np.mean(heights) - height) > 10):
#             continue
#         # if (width > 300) or (height > 119):
#         #     print(root_dir, i)
#         #     # cvshow(img, xmin, ymin, xmax, ymax, rdc_x, rdc_y, ldc_x, ldc_y, ruc_x, ruc_y, luc_x, luc_y, '111111')
#         #     continue
#         widths.append(width)
#         heights.append(height)

mean_width = 190 # np.mean(widths)
mean_height = 60 # np.mean(height)
print('车牌的平均宽度为：', mean_width)
print('车牌的平均高度为：', mean_height)

stage1_images = os.path.join(pwd, 'data', 'stage1', 'images')
stage1_labels = os.path.join(pwd, 'data', 'stage1', 'labels')
stage1_test = os.path.join(pwd, 'data', 'stage1', 'test')

rectangle = os.path.join(pwd, 'data', 'corner', 'rectangle')
rectangle_label = os.path.join(pwd, 'data', 'corner', 'label')
rectangle_test = os.path.join(pwd, 'data', 'corner', 'test')

# labeltxt = os.path.join(pwd, 'data', 'crnn', 'labeltxt')
# warpimg = os.path.join(pwd, 'data', 'crnn', 'warpimg')
# warpimg_test = os.path.join(pwd, 'data', 'crnn', 'test')

dirs = [stage1_images, stage1_labels, rectangle, rectangle_label, stage1_test, rectangle_test]

delete = input('是否删除之前的文件夹？Y or N')
for i in dirs:
    if delete.upper()=='Y':
        try:
            shutil.rmtree(i)
        except:
            pass
    os.makedirs(i)

print('begin proprecess...')
start = time.time()
count = 0
for root_dir, dir, file in os.walk(path):
    if len(file)==0:
        continue
    if ('&' not in file[0]) or ('-' not in file[0]):
        continue
    np.random.shuffle(file)
    # file = file[:30000]
    for i in file:
        # if count > 1000:
        #     break
        count+=1
        ori_i = copy.deepcopy(i)
        name = copy.deepcopy(i)
        img = cv2.imread(os.path.join(root_dir, i))
        shape = img.shape
        i = i.replace('.jpg', '').replace('.png', '').replace('&', '_').replace('-', '_').split('_')
        angle = [int(i[1]), int(i[2])]
        xmin, ymin, xmax, ymax = int(i[3]), int(i[4]), int(i[5]), int(i[6])

        #stage1
        convert = cal_txt(shape, xmin, ymin, xmax, ymax, i)
        if convert[0] < 0:
            continue
        name = name.replace('.jpg', '.txt').replace('.png', '.txt')
        if count<30000:
            shutil.copyfile(os.path.join(root_dir, ori_i), os.path.join(pwd, 'data', 'stage1', 'images', ori_i))
            with open(os.path.join(pwd, 'data', 'stage1', 'labels', name.replace('.jpg', '.txt')), 'w') as obj:
                obj.write("0 " + " ".join([str(a) for a in convert]) + '\n')

        rdc_x, rdc_y, ldc_x, ldc_y, luc_x, luc_y, ruc_x, ruc_y = int(i[7]), int(i[8]), int(i[9]), int(i[10]), \
                                                                 int(i[11]), int(i[12]), int(i[13]), int(i[14])
        xmax = np.max([rdc_x, ruc_x, xmax])
        xmin = np.min([luc_x, ldc_x, xmin])
        ymin = np.min([luc_y, ruc_y, ymin])
        ymax = np.max([ldc_y, rdc_y, ymax])
        platestring = chinesechar_reverse[int(i[15])]
        for j in i[16:-2]:
            platestring += alphaenglish_reverse[int(j)]
        platestring = platestring.upper()
        # print(platestring.upper())
        # cvshow(img, xmin, ymin, xmax, ymax, rdc_x, rdc_y, ldc_x, ldc_y, ruc_x, ruc_y, luc_x, luc_y, platestring)
        start_points = [[luc_x, luc_y], [ruc_x, ruc_y], [ldc_x, ldc_y], [rdc_x, rdc_y]]
        # warp_img = warp(mean_height, mean_width, start_points, img)
        rectangle_save(img, xmin, ymin, xmax, ymax, rdc_x, rdc_y, ldc_x, ldc_y, ruc_x, \
                       ruc_y, luc_x, luc_y, rectangle, ori_i, rectangle_label)
        # warp_img = Image.fromarray(warp_img)
        # warp_img.save(os.path.join(warpimg, platestring+'.jpg'))
        # with open(os.path.join(labeltxt, ori_i.replace('.jpg', '.txt').replace('.png', '.txt')), 'w') as obj:
        #     obj.write(platestring)

#自动划分训练集和验证集
namelist = [os.path.join(stage1_images, i) for i in os.listdir(stage1_images)]
trainchoose = list(np.random.choice(namelist, len(namelist)*9//10, replace=False))
valchoose = list(set(namelist) - set(trainchoose))
testchoose = list(np.random.choice(valchoose, len(valchoose)*9//10, replace=False))
valchoose = list(set(valchoose) - set(testchoose))

for i in testchoose:
    shutil.move(i, i.replace('images', 'test'))
    shutil.move(i.replace('images', 'labels').replace('.jpg', '.txt'), \
                    i.replace('.jpg', '.txt').replace('images', 'test'))

write_train_val(testchoose, 'stage1', 'test')
print('训练集图片个数：{}，验证集图片个数：{}'.format(len(trainchoose), len(valchoose)))
write_train_val(trainchoose, 'stage1', 'train')
write_train_val(valchoose, 'stage1', 'valid')

namelist = [os.path.join(rectangle, i) for i in os.listdir(rectangle)]
trainchoose = list(np.random.choice(namelist, len(namelist)*9//10, replace=False))
valchoose = list(set(namelist) - set(trainchoose))
testchoose = list(np.random.choice(valchoose, len(valchoose)*9//10, replace=False))
valchoose = list(set(valchoose) - set(testchoose))
for i in testchoose:
    shutil.move(i, i.replace('rectangle', 'test'))
    shutil.move(i.replace('rectangle', 'label').replace('.jpg', '.txt'), \
                    i.replace('.jpg', '.txt').replace('rectangle', 'test'))

# namelist = [os.path.join(warpimg, i) for i in os.listdir(warpimg)]
# trainchoose = list(np.random.choice(namelist, len(namelist)*9//10, replace=False))
# valchoose = list(set(namelist) - set(trainchoose))
# testchoose = list(np.random.choice(valchoose, len(valchoose)*9//10, replace=False))
# valchoose = list(set(valchoose) - set(testchoose))
# for i in testchoose:
#     shutil.move(i, i.replace('warpimg', 'test'))
    # shutil.move(i.replace('warpimg', 'labeltxt').replace('.jpg', '.txt'), \
    #                 i.replace('.jpg', '.txt').replace('warpimg', 'test'))

print('use: ', round((time.time() - start)/60, 3), 'min')