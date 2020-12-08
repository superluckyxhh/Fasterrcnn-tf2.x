import tensorflow as tf
import numpy as np


class AnchorGenerator:
    def __init__(self,
                scales = [128, 256, 512],
                ratios = [[1, 1], [1, 2], [2, 1]],
                rpn_scale = 16):
        self.scales = scales
        self.ratios = ratios
        self.rpn_scale = rpn_scale

    def anchor_generator(self):
        anchor_num = len(self.scales) * len(self.ratios)
        anchors = np.zeros(shape=[anchor_num, 4])
        anchors[:, 2:] = np.tile(self.scales, (2, len(self.ratios))).T
        
        for i in range(len(self.ratios)):
            anchors[i*3:i*3+3, 2] = anchors[i*3:i*3+3, 2] * self.ratios[i][0]
            anchors[i*3:i*3+3, 3] = anchors[i*3:i*3+3, 3] * self.ratios[i][1]

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        # One grid anchors[9,4]
        return anchors

    def anchor_shift(self,
                    feature_shape,
                    anchors):
        # Feature map shape [None, w, h, c]
        feature_height, feature_width = feature_shape[1:3]

        feature_center_x = np.arange(0, feature_height, 1) + 0.5
        feature_center_y = np.arange(0, feature_width, 1) + 0.5

        origin_center_x = feature_center_x * self.rpn_scale
        origin_center_y = feature_center_y * self.rpn_scale

        origin_center_x, origin_center_y = np.meshgrid(origin_center_x, origin_center_y)

        origin_center_x = origin_center_x.flatten()
        origin_center_y = origin_center_y.flatten()
    
        origin_center = np.stack([origin_center_x, 
                                origin_center_y, 
                                origin_center_x,
                                origin_center_y], 
                                axis=0)

        origin_center = np.transpose(origin_center)

        num_anchors = anchors.shape[0]
        k = origin_center.shape[0]
        
        shifted_anchors = np.reshape(anchors, [1, num_anchors, 4]) + np.array(
                        np.reshape(origin_center, [k, 1, 4]), tf.keras.backend.floatx())
        # [leftcorner, rightcorner]
        shifted_anchors = np.reshape(shifted_anchors, [k * num_anchors, 4])
        # One img all anchors:[2916, 4]

        return shifted_anchors
    
    def get_anchor(self, feature_shape, height, width):
        """
        Give a feature map and return anchors, 
        shape[feature_map_h * feature_map_w * k, 4],
        represent the [xmin, xmax, ymin, ymax] 
        """
        anchors = self.anchor_generator()
        shifted_anchors = self.anchor_shift(feature_shape, anchors)
        # Trans to [0, 1] similiar to cut section that out of img
        shifted_anchors[:, 0] = shifted_anchors[:, 0] / width 
        shifted_anchors[:, 1] = shifted_anchors[:, 1] / height 
        shifted_anchors[:, 2] = shifted_anchors[:, 2] / width 
        shifted_anchors[:, 3] = shifted_anchors[:, 3] / height
        shifted_anchors = np.clip(shifted_anchors, 0, 1)
        # [2916, 4]
        return shifted_anchors

class BoundingBox:
    def __init__(self,
                num_classes=1,
                pos_threshold=0.5,
                neg_threshold=0.3,
                confidence_threshold = 0.5,
                iou_threshold=0.7,
                top_K=300,
                rpn_scale=16):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.top_K = top_K
        self.rpn_scale = rpn_scale
    
    def NMS(self, boxes, scores):
        selected_index = tf.image.non_max_suppression(
                            boxes=boxes, 
                            scores=scores,
                            max_output_size=self.top_K,
                            iou_threshold=self.iou_threshold)

        return selected_index

    def IoU(self, box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        width1 = np.abs(xmax1 - xmin1)
        height1 = np.abs(ymax1 - ymin1)
        width2 = np.abs(xmax2 - xmin2)
        height2 = np.abs(ymax2 - ymin2)
        
        intersect_width = width2 + width1 - np.abs(xmax2 - xmin1)
        intersect_height = height2 + height1 - np.abs(ymax2 - ymin1)
        
        intersect_area = intersect_height * intersect_width
        union_area = (width1 * height1) + (width2 * height2) - intersect_area

        iou = intersect_area / union_area
        
        return iou

    # One img :reg(m*n*9,4) shape =  anchors(m*n*9,4) shape
    def decode_bbox(self, reg, anchors):
        """
        Give a reg param and anchors of a feature map
        Adopt the parameterizations of the 4 coordinates and get predict box
        shape is [fm_h * fm_w * k, 4]

        *Param: reg [fm_h * fm_w * k, 4]
                cls [fm_h * fm_w * k, 4]
        *Return: decode_box [fm_h * fm_w * k, 4]
        """
        anchor_height = anchors[:, 3] - anchors[:, 1]
        anchor_width = anchors[:, 2] - anchors[:, 0]
   
        anchor_center_x = (anchors[:, 0] + anchors[:, 2]) * 0.5 
        anchor_center_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
        
        # Computer the predict value by t_i* and anchors 
        decode_x = anchor_width * reg[:, 0] + anchor_center_x
        decode_y = anchor_height * reg[:, 1] + anchor_center_y
        decode_w = np.exp(reg[:, 2]) * anchor_width
        decode_h = np.exp(reg[:, 3]) *  anchor_height
        # Trans (center_x, center_y, h, w) to (left_x, left_y, right_x, right_y)
        xmax = decode_x + decode_w * 0.5
        xmin = decode_x - decode_w * 0.5
        ymax = decode_y + decode_h * 0.5
        ymin = decode_y - decode_h * 0.5

        decode_box = np.concatenate((xmin[:, None],
                                    ymin[:, None],
                                    xmax[:, None],
                                    ymax[:, None]), axis=-1)
        decode_box = np.clip(decode_box, 0, 1)
        # decode_box equal predict box[2916, 4]
        return decode_box  

    def get_proposal(self, reg, cls, anchors):
        """
        Using the rpnHead and anchors to compute the predict box
        1. Get the all predict boxes;
        2. Select the boxes that conf is greater than conf threshold;
        3. Do NMS to delete some boxes;
        4. Sort the boxes with confidence.

        *param: reg [num_of_anchors, 4]
                cls [num_of_anchors, 1]
                anchors [num_of_anchors, 4]
        *Return: results list [array(label, conf, xmin, ymin, xmax, ymax)]
        """
        # One batch imgs
        results = []
        for i in range(len(reg)):
            results.append([])
            # Get all predict boxes in one img
            decode_boxes = self.decode_bbox(reg[i], anchors)
            for c in range(self.num_classes):
                conf = cls[i,:,c]
                # selected_conf is bool type
                selected_bool = conf > self.confidence_threshold
                # conf[selecetd] shape:[1481, 4], [1491, 4]...
                if len(conf[selected_bool]) > 0:
                    # selected_box:[selected_nums, 4]
                    # selected_conf:]selected_nums, 1]
                    selected_boxes = decode_boxes[selected_bool]
                    selected_confs = conf[selected_bool]
    
                    # selected_box_index shape [num_of_nms_selected, ]
                    selected_box_index = self.NMS(selected_boxes, selected_confs)
           
                    after_nms_boxes = tf.gather(selected_boxes, selected_box_index)
                    after_nms_confs = tf.reshape(tf.gather(selected_confs, selected_box_index),[-1, 1])
 
                    # Concat the label, confidence and box
                    # labels shape is [num_of_nms_selected, 1]
                    labels = c * np.ones((len(selected_box_index), 1))
                    # Predict is [label, conf, xmin, ymin, xmax, ymax](One deature map)
                    predict = np.concatenate([labels, after_nms_confs, after_nms_boxes],
                                            axis=1)
                    results[-1].extend(predict)

            if len(results[-1]) > 0:
                # Sort the results 
                results[-1] = np.array(results[-1])
                sort_results = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][sort_results]

        # TODO: if u want to watch results shape, do this:
        # results = np.array(results)

        return results
    
    def TransProposals(self, proposals, origin_width, origin_height):
        """
        Scale the proposals onto feature map, only care the coordination and delete some box (h or w < 1)

        """
        R = proposals[0][:, 2:]

        R[:, 0] = np.array(np.round(R[:, 0] * origin_width / self.rpn_scale),dtype=np.int32)
        R[:, 1] = np.array(np.round(R[:, 1] * origin_height / self.rpn_scale),dtype=np.int32)
        R[:, 2] = np.array(np.round(R[:, 2] * origin_width / self.rpn_scale),dtype=np.int32)
        R[:, 3] = np.array(np.round(R[:, 3] * origin_height / self.rpn_scale),dtype=np.int32)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        R = np.array(R)
        print('R shape:', R.shape)
        delete_boxes = []
        for id, roi in enumerate(R):
            if roi[2] < 1 or roi[3] < 1:
                delete_boxes.append(id)
        R = np.delete(R, delete_boxes, axis=0)
        print('proposals:', R)    
        print('proposals: shape', R.shape)

        for jk in range(R.shape[0] // 129):
            RoIs = np.expand_dims(R[128 * jk : 128 * (jk + 1), :], axis = 0)
        print('RoI shape:', RoIs.shape)
        print(RoIs)                    
        return RoIs
                        
                        



    
   

        
        


if __name__ == '__main__':
    x = tf.random.uniform(shape = [1, 2, 2, 256], minval=0, maxval=10)
    feature = tf.random.uniform(
                shape=[18, 18, 256],
                minval=-1,
                maxval=1)
    feature_shape = feature.shape
    a = AnchorGenerator()
    anchors = a.get_anchor(feature_shape, 300, 300)
    print('anchors shape:', anchors.shape)
    
    reg = tf.random.uniform([10, 2916, 4])
    cls = tf.random.uniform([10, 2916, 1])
    b = BoundingBox()
    b.decode_bbox(reg[1], anchors)
    b.get_proposal(reg, cls,anchors)
    # box1 = [1, 2, 2, 0]
    # box2 = [1, 1, 3, 0]
  
