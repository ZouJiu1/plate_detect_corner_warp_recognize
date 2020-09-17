import os
train = r"/data/yolov5/data/crnn/warpimg"
test = r"/data/yolov5/data/crnn/test"
outtrain = '/data/yolov5/CRNN_Chinese_Characters_Rec/lib/train_own.txt'
outtest = '/data/yolov5/CRNN_Chinese_Characters_Rec/lib/test_own.txt'

tr = [i for i in os.listdir(train) if '.jpg' in i]
te = [i for i in os.listdir(test) if '.jpg' in i]
print(len(tr))
print(len(te))
print('皖AH765B.jpg' in tr)

with open(outtrain, 'w') as obj:
    for i in tr:
        if os.path.exists(os.path.join(train, i)):
            obj.write(i + ' ' + i.replace('.jpg','').replace('.png','')+'\n')
with open(outtest, 'w') as obj:
    for i in te:
        if os.path.exists(os.path.join(test, i)):
            obj.write(i + ' ' + i.replace('.jpg','').replace('.png','')+'\n')