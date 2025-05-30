import numpy as np


# anchor box 是很多候选框，然后通过算法求出最终的bounding box
class AnchorsBox:
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape
        self.min_size = min_size
        self.max_size = max_size
        #aspect_ratios 是宽高比缩放
        self.aspect_ratios = []

        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0/ar)

    def call(self,layer_shape, mask = None):
        # --------------------------------- #
        #   获取输入进来的特征层的宽和高
        #   比如38x38
        # --------------------------------- #
        layer_height = layer_shape[0]
        layer_width  = layer_shape[1]

        # --------------------------------- #
        #   获取输入进来的图片的宽和高
        #   比如300x300
        # --------------------------------- #
        img_height = self.input_shape[0]
        img_width =  self.input_shape[1]

        box_height = []
        box_width  = []


        # --------------------------------- #
        #   self.aspect_ratios一般有两个值
        #   [1, 1, 2, 1/2]
        #   [1, 1, 2, 1/2, 3, 1/3]
        # --------------------------------- #
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_width) == 0:
                box_width.append(self.min_size)
                box_height.append(self.min_size)

            elif ar == 1 and len(box_width) > 1:
                box_width. append(np.sqrt(self.min_size * self.max_size))
                box_height.append(np.sqrt(self.min_size * self.max_size))

            elif ar != 1:
                box_width. append(self.min_size * np.sqrt(ar))
                box_height.append(self.min_size / np.sqrt(ar))


        #获得所有先验框的1/2
        box_width = np.array(box_width) * 0.5
        box_height = np.array(box_height) * 0.5

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        # --------------------------------- #
        #   生成网格中心
        # --------------------------------- #
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y , layer_height)

        center_x, center_y = np.meshgrid(linx, liny)

        #reshape成一维列向量
        center_x = np.reshape(-1,1)
        center_y = np.reshape(-1,1)

        num_anchors = len(self.aspect_ratios)
        #数组拼接np.concatenate(axis = 1 表示列向量方向)
        anchors_box = np.concatenate((center_x, center_y), 1 )
        #anchors_box 在x方向上复制2倍
        anchors_box = np.tile(anchors_box,(1, 2 * num_anchors))

        #获得先验框的左下角跟右上角
        anchors_box[:, ::4]  -= box_width
        anchors_box[:,1::4]  -= box_height
        anchors_box[:,2::4]  += box_width
        anchors_box[:,3::4]  += box_height

        # --------------------------------- #
        #   将先验框变成小数的形式
        #   归一化
        # --------------------------------- #
        anchors_box[:,::2] /= img_width
        anchors_box[:,1::2] /= img_height
        anchors_box = np.reshape(anchors_box, 4)
        anchors_box = np.min(np.max(anchors_box,0), 1)
        return anchors_box



def get_vgg_output_length(height, width):
    #计算特征层大小
    filter_size  = [3,3,3,3,3,3,3,3]
    padding      = [1,1,1,1,1,1,0,0]
    stride       = [2,2,2,2,2,2,1,1]
    feature_height = []
    feature_width  = []
    for i in range(len(filter_size)):
        height = (height + 2*  padding[i] - filter_size[i])
        width  = (width + 2 * padding[i] - filter_size[i])
        feature_width.append(width)
        feature_height.append(height)

    return np.array(feature_height[-6:]), np.array(feature_width[-6:])


def get_anchors(input_shape = [300,300], anchors_size = [30, 60, 111, 162, 213, 264, 315], backbone = 'vgg'):
    feature_height, feature_width = get_vgg_output_length(input_shape[0], input_shape[1])
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_height)):
        anchors_boxes = AnchorsBox(input_shape, anchors_size[i], max_size= anchors_size[i + 1],aspect_ratios=aspect_ratios[i]).call([feature_height, feature_height])
        anchors.append(anchors_boxes)

    #列方向上复制一遍
    anchors = np.concatenate(anchors, 0)
    return anchors



