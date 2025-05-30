from statistics import variance
from uu import decode

import numpy as np
import torch
from torch.ao.nn.quantized import Softmax
import torch.nn as nn
from torchvision.ops import nms


class bboxUtil(object):
        def __init__(self, num_class):
            self.num_class = num_class

        # https://blog.csdn.net/weixin_44791964/article/details/104981486   SSD(Single Shout MultiBox Detector) 写的不错的一个博客
        def ssd_correct_box(self, box_xy,box_wh, input_shape, image_shape, letterbox_image):
                box_yx = box_xy[...,::1]
                box_hw = box_wh[...,::1]
                input_shape = np.ndarray(input_shape)
                image_shape = np.ndarray(image_shape)

                if letterbox_image :
                    #ssd的letterbox,np.round()函数是四舍五入
                    new_shape = np.round(image_shape * np.min(input_shape/image_shape))
                    #offect 偏移情况
                    offset = (input_shape - new_shape ) /2./input_shape
                    scale = input_shape/new_shape

                    box_yx = (box_yx - offset )* scale
                    box_hw = scale * box_hw
                #获得左下角box_min ，获得右上角box_max
                box_min = box_yx - (box_hw/2.)
                box_max = box_yx + (box_hw/2.)
                boxes = np.concatenate([box_min[..., 0:1], box_min[..., 1:2], box_max[..., 0:1], box_max[..., 1:2]], axis=-1)
                boxes *= np.concatenate([image_shape, image_shape], axis = -1)

                return  boxes


        def decode_boxes(self, mbox_loc, anchors, variance):
            #获得秒顶框的宽高
            anchor_width = anchors[:, 2] - anchors[:,0]
            anchor_height = anchors[:,3] - anchors[:,1]

            #获取先验框中心点
            anchor_center_x = 0.5*(anchors[:,2] - anchors[:,0])
            anchor_center_y = 0.5*(anchors[:,3] - anchors[:,1])

            #真实框对比先验框的计算偏移
            decode_bbox_center_x = mbox_loc[:,0] * anchor_width * variance[0]
            decode_bbox_center_x += anchor_center_x
            decode_bbox_center_y = mbox_loc[:,1] * anchor_height * variance[0]
            decode_bbox_center_y += anchor_center_y

            #获取真实框的宽高
            decode_bbox_width = torch.exp(mbox_loc[:, 2] * anchor_width * variance[1])
            decode_bbox_width *= anchor_width
            decode_bbox_height = torch.exp(mbox_loc[:, 3] * anchor_height * variance[1])
            decode_bbox_height *= anchor_height

            #获取真实框的几个角落坐标
            decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
            decode_bbox_ymin = decode_bbox_center_y - 0.5* decode_bbox_height
            decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
            decode_bbox_ymax = decode_bbox_center_y + 0.5 *decode_bbox_height

            boxes = torch.cat(
                (decode_bbox_xmin[:,None],
                decode_bbox_ymin[:,None],
                decode_bbox_xmax[:,None],
                decode_bbox_ymax[:,None]), dim = 1
            )

            boxes = torch.min(torch.max(boxes, torch.zeros_like(boxes)), torch.ones_like(boxes))
            return boxes


        def decode_box(self, prediction, anchors, image_shape, input_shape, letterbox_image, variances=None, nms_iou = 0.3, confidence  = 0.5):
            if variances is None:
                variances = [0.1, 0.2]
            mbox_loc = prediction[0]
            mbox_conf = nn.Softmax(-1)(prediction[1])
            result = []

            for i in range(len(mbox_loc)):
                result.append([])
                decode_bbox = self.decode_boxes(mbox_loc[i], anchors, variances)

                for c in range(1, self.num_class):
                    c_confs = mbox_conf[i,:,c]
                    c_conf_m = c_confs > confidence

                    if len(c_confs[c_conf_m]) > 0:
                        boxes_to_process = decode_bbox[c_confs]
                        confs_to_process = decode_bbox[c_conf_m]

                        keep = nms(
                            boxes_to_process,
                            confs_to_process,
                           nms_iou
                        )
                        # -----------------------------------------#
                        #   取出在非极大抑制中效果较好的内容
                        # -----------------------------------------#
                        good_boxes = boxes_to_process[keep]
                        confs = confs_to_process[keep][:, None]
                        labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else torch.ones(
                            (len(keep), 1))
            if len(result[-1]) > 0:
                result[-1] = np.array(result[-1])
                box_xy, box_wh = (result[-1][:, 0:2] + result[-1][:, 2:4]) / 2, result[-1][:, 2:4] - result[-1][:, 0:2]
                result[-1][:, :4] = self.ssd_correct_box(box_xy, box_wh, input_shape, image_shape, letterbox_image)

            return result

