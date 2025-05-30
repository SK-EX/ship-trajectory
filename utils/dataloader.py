from random import shuffle

import numpy as np
from PIL import Image
import cv2


#用于加载数据
class SSDDataloader:
    def __init__(self,annotation_lines, input_shape, anchors, batch_size, num_classes, train, overlap_threshold = 0.5):
        self.annotations_lines  = annotation_lines
        self.length = len(self.annotations_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.is_train = train
        self.overlap_threshold = overlap_threshold


    def len(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        imgae, box = self.get_random_data()

    def rand(self, a1, a2):
        return np.random.rand() * (a2 - a1) + a1
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):

        #将annotation字符按照空格分隔
        line  = annotation_line.split()
        image = Image.open(line[0])

        #BGR颜色空间转换为RGB颜色空间
        image = cv2.cvtColor(image)
        #获取原始图像的宽度与高度
        iw , ih = image.size
        #获取期望输入形状的宽度高度
        h, w = input_shape
        #将边界信息框从字符串转为整数数组
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            #计算缩放比例
            scale = min( w / iw, h / ih)
            #计算新的宽度高度
            nw = int(iw * scale )
            nh = int(ih * scale)
            #计算水平、垂直线上的偏移量
            dx = (w-nw)//2
            dy = (h-nh)//2

            #将多余的图像加上灰条
            #将图片缩放到新的尺寸
            image       = image.resize((nw,nh),Image.BICUBIC)
            #创建新的图像，颜色为灰色（128，128，128）
            new_image   = Image.new('RGB',(w,h),(128,128,128))
            #将缩放后的图像粘贴到新的图像中心
            new_image.paste(image, (dx,dy))
            image_data  = np.array(new_image, np.float32)


            #检查边界框是否为空
            if len(box) > 0:
            #如果random参数为true,随机打乱边界框的顺序
                np.random.shuffle(box)
                #调整box的x坐标（左边界跟右边界，使其适应新的图像尺寸,并加上水平偏移量）
                box[: , [0,2]] = box[:, [0,2]] * nw / iw + dx
                box[: , [1,3]] = box[:, [1,3]] * nh / ih + dy
                #将
                box[:,0:2][box[:,0:2] < 0 ] = 0
                box[:,2][box[:, 2] > w] = w
                box[:,3][box[:, 3] > h] = h
                box_w = box[:,2] - box[:,0]
                box_h = box[:,3] - box[:,1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                return image_data, box

            #将图片缩小到一定比例
            new_ar = iw / ih * self.rand(1 - jitter , 1 + jitter) / self.rand(1 - jitter , 1 + jitter )
            scale = self.rand(.25, 2)
            if new_ar < 1:
                # iw < ih,w小h大
                nh = int(scale * h)
                nw = int(new_ar * nh)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image.resize((nw, nh), Image.BICUBIC)

            # ------------------------------------------#
            #   将图像多余的部分加上灰条
            # ------------------------------------------#

            dx = self.rand(0, w- nw)
            dy = self.rand(0 , h - nh)

            new_image = Image.new('RGB', (nw, nh), (128, 128, 128))
            # 将缩放后的图像粘贴到新的图像中心
            new_image.paste(image, (dx, dy))
            image = new_image

            #将图像转化为hsv图像
            #设置随机种子， 随机种子小于0.5进行随机数据增强
            flip = self.rand(0 ,1) < 0.5
            if flip:
                r = np.random.uniform(-1 , 1 ,3) *[hue, sat, val]
                #split拆分hue,sat, val三元组
                hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
                dtype = image_data.dtype

                x = np.arange(0,255,dtype = r.dtype)
                lut_hue =(x * r[0] % 180).astype(dtype)
                #将sat,val限制在0,255，使用np.clip函数
                lut_sat = np.clip(x * r[1] , 0,  255).astype(dtype)
                lut_val = np.clip(x * r[2] , 0 ,  255).astype(dtype)

                #整合
                image_data  = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val,lut_val)))
                image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

            #对真实框进行调整
            if len(box) > 0:
                # 如果random参数为true,随机打乱边界框的顺序
                np.random.shuffle(box)
                # 调整box的x坐标（左边界跟右边界，使其适应新的图像尺寸,并加上水平偏移量）
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 将
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                return image_data, box

    def iou(self, box):
        #计算广义交并比
        #真实框面积
        true_box = (box[2] - box[0]) * (box[3] - box[1])
        #先验框面积
        anchor_box = (self.anchors[: ,2]  - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])


        inter_upleft = np.maximum(self.anchors[: ,2], box[2])
        inter_boright = np.minimum(self.anchors[:, 2:4], box[2])

        inter_wh = inter_upleft - inter_boright
        inter_wh = np.maximum(inter_wh, 0 )

        inter = inter_wh[: , 0] * inter_wh[: ,1]
        union = true_box + anchor_box - inter
        iou = inter / union
        return iou

    #蒋boundingbox转换为ssd能识别的格式
    def encode_box(self, box, return_iou=True, variances = [0.1, 0.1, 0.2, 0.2]):

        iou = self.iou(box)
        #新定义一个encode_box矩阵, n * 4 、、 n * 5(center_x, center_y , w, h , iou)格式
        encode_box = np.zeros((self.num_classes , return_iou + 4 ))

        #返回bool数组
        assign_mask = iou > self.overlap_threshold


        #找到数组中最大的下标
        #如果都灭有大于阈值的iou，找到一个最大的iou ， 把他设置成true
        if not np.any(assign_mask):
            assign_mask[np.argmax(iou)] = True

        #用这个iou去匹配真实框
        #将iou赋值给encode_box的最后一哈哈， 其他行就是（center——x，center——y， w ， h）
        if return_iou:
            encode_box[:, -1][assign_mask] = iou[assign_mask]


        assign_anchors = self.anchors[assign_mask]
        #box格式[[
        # ]
        # ]
        box_center = (box[:2] + box[2:]) * 0.5
        box_wh = box[2:] - box[:2]

        #
        assign_anchors_center = (assign_anchors[:, 2:4] + assign_anchors[:, 0:2] )*0.5
        assign_anchors_wh     = assign_anchors[:, 2:4] - assign_anchors[:, 0 : 4]

        #逆向编码
        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encode_box[:, :2][assign_mask] = box_center - assign_anchors_center
        encode_box[:, :2][assign_mask] /= assign_anchors_wh
        encode_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encode_box[:, 2:4][assign_mask] = np.log(box_wh / assign_anchors_wh)
        encode_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]

        return encode_box.ravel()


