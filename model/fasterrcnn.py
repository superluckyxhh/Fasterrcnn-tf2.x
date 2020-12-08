"""
*---------------------------------- Revision History --------------------------------------
<name>                             <date>          <version>       <desc>
nanxiaohu18@mails.ucas.ac.cn       2020.11.29      1.0             create this moules

*------------------------------------------------------------------------------------------
"""
from __future__ import print_function
from dataset.preprocess import PreProcess

import tensorflow as tf
import numpy as np
from absl import app
from copy import deepcopy
from utils.rpn_tools import AnchorGenerator
from utils.rpn_tools import BoundingBox

class RoIPooling(tf.keras.Model):
    def __init__(self,
                pooling_size=7,
                num_roi=32):
        super(RoIPooling, self).__init__()
        self.pooling_size = pooling_size
        self.num_roi = num_roi

    def call(self, feature_map, proposals):
        """
        :Param 
        proposal [None, num_proposals, 4], specific meaning [xmin, ymin, width, height]
        feature_map [None, h, w, c_f] 
        :Return
        roi_pooling [None, pooling_size, pooling_size, c_r] 
        """
        print('fa proposal shape:', proposals.shape)
        print('fa fm shape:', feature_map.shape)
        channel = feature_map.shape[3]
        output = []
        for id in range(proposals.shape[0]):
            x = proposals[id, 0]
            y = proposals[id, 1]
            w = proposals[id, 2]
            h = proposals[id, 3]
            # print(x, y, w, h)
            x = tf.cast(x, dtype=tf.int32)
            y = tf.cast(y, dtype=tf.int32)
            w = tf.cast(w, dtype=tf.int32)
            h = tf.cast(h, dtype=tf.int32)
            print(x, y, w, h)
            roi = feature_map[:, y:y+h, x:x+w, :]
            # print('roi:', roi)
            resized_roi = tf.image.resize(roi, [self.pooling_size, self.pooling_size])
            print('resized_roi:', resized_roi.shape)

            output.append(resized_roi)
        
        output = tf.concat(output, axis=0)
        output = tf.reshape(output, [1, self.pooling_size, self.pooling_size, channel])

        return output
        

class PostProcess:
    def __init__(self):
        pass
    def postprocess(self):
        pass     

class classifier(tf.keras.Model):
    def __init__(self):
        super(classifier, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc_cls = tf.keras.layers.Dense(units=21, activation='relu')
        self.fc_reg = tf.keras.layers.Dense(units=128, activation='relu')
    def call(self, roi_pooling):
        pass

class RPNHead(tf.keras.Model):
    def __init__(self,
                num_anchors = 9):
        super(RPNHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 256 ,
            kernel_size=3,
            padding='same',
            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            filters = self.num_anchors * 4,
            strides=1,
            kernel_size=1)
        self.conv3 = tf.keras.layers.Conv2D(
            filters = self.num_anchors,
            strides=1,
            kernel_size=1, activation='sigmoid')

    def call(self, inputs, training = False):
        intermediate = self.conv1(inputs)
        cls_prelogits = self.conv3(intermediate)
        reg_box = self.conv2(intermediate)
        cls_shape = cls_prelogits.shape
        reg_shape = reg_box.shape
        cls_prelogits = tf.reshape(cls_prelogits, [cls_shape[0], -1, 1])
        reg_box = tf.reshape(reg_box, [reg_shape[0], -1, 4])

        return cls_prelogits, reg_box

class RPNModule(tf.keras.Model):
    def __init__(self):
        super(RPNModule, self).__init__()
        self.anchorgenerator = AnchorGenerator()
        self.boundingbox = BoundingBox(num_classes=1)
        self.rpnhead = RPNHead()
    def call(self, feature_map, origin_w, origin_h):
        feature_shape = feature_map.shape
        print('feature shape:', feature_shape)
        cls, reg = self.rpnhead.call(feature_map)
        print('cls shape:', cls.shape)
        print('reg shape:', reg.shape)
        anchors = self.anchorgenerator.get_anchor(feature_shape, origin_w, origin_h)
        print('anchors shape:', anchors.shape)
        
        proposals = self.boundingbox.get_proposal(reg, cls, anchors)
        proposals = self.boundingbox.TransProposals(proposals, origin_w, origin_h)
    
        return proposals, cls, reg, anchors
        

class FasterRcnn(tf.keras.Model):
    def __init__(self):
        super(FasterRcnn, self).__init__()
        # 1.Extractor(backbone:VGG16 with preweights and no dense layers. stop block5_conv3)
        self.backbone_base = tf.keras.applications.VGG16(
            weights = 'imagenet')
        self.backbone = tf.keras.Model(
            inputs = self.backbone_base.input,
            outputs = self.backbone_base.get_layer('block5_conv3').output)
        # 2.RPN Module
        self.rpnmodule = RPNModule()
        # 3.
    
    def call(self, inputs, origin_w, origin_h, training = False):
        conv_feature_map = self.backbone(inputs)
        proposals, _, _, _ = self.rpnmodule.call(conv_feature_map, origin_w, origin_h)

        return conv_feature_map, proposals



def main():
    # input_tensor = tf.random.uniform(shape = [1, 300, 300, 3])
    path = '/home/user/code/FasterRcnn/dataset/test_img/car.png'
    input = tf.io.read_file(path)
    img_tensor = tf.image.decode_png(input) 
    img_tensor_shape = img_tensor.shape
    # img_tensor = tf.reshape(img_tensor, [-1, img_tensor_shape[0], img_tensor_shape[1], img_tensor_shape[2]])
    img = np.expand_dims(np.array(img_tensor), 0)

    img_shape = np.shape(img)
    origin_width = img_shape[1]
    origin_height = img_shape[2]
    origin_img = deepcopy(img_tensor)

    # print(img_shape)
    # print(origin_height)
    # print(origin_width)
    # print(origin_img)
    
    fasterrcnn_model = FasterRcnn()
    
    res1, res2 = fasterrcnn_model.call(img, origin_width, origin_height)
  
    # print('feature shape:', res1.shape)
    # print('proposals:', res2)
    x = RoIPooling()
    x.call(res1, res2)
    

if __name__ == '__main__':
    main()
