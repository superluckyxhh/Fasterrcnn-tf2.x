import tensorflow as tf
import numpy as np
from copy import deepcopy


class PreProcess(tf.keras.Model):
    def __init__(self, 
                min_size,
                max_size,
                img_mean,
                img_std,
                obj_gt = None):
        super(PreProcess, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.obj_gt = obj_gt
    
    # Suppose img is a tensor
    def normalize(self, image):
        """
        :Param image [None, w, h, c]
        """
        image = tf.cast(image, tf.float32)
        normalized_img = tf.math.divide(tf.subtract(image, self.img_mean), self.img_std)
        
        return normalized_img

    def resize_bbox(self, box, origin_imgshape, zoom_imgshape):
        zoom_f = []
        for i, j in zip(origin_imgshape, zoom_imgshape):
            zoom_f.append(j / i) 

        zoom_h_factor, zoom_w_factor = zoom_f[:] # TODO
        xmin, ymin, xmax, ymax = tf.slice(box, 1, -1)
        # print(xmin)
        # print(ymin)
        # print(xmax)
        # print(ymax)
        xmin = xmin * zoom_w_factor
        xmax = xmax * zoom_w_factor
        ymin = ymin * zoom_h_factor
        ymax = ymax * zoom_h_factor
        resized_box = tf.stack((xmin, ymin, xmax, ymax), axis = 0)
        
        return resized_box
        

    def resize(self, image):
        """
        :Param image [None, w, h, c]
        """
        height, width = image.shape[1:3]
        min_imgsize = tf.cast(tf.minimum(height, width), tf.float64)
        max_imgsize = tf.cast(tf.maximum(height, width), tf.float64)
        zoom_factor = tf.math.divide(self.min_size, min_imgsize)
   
        if zoom_factor * max_imgsize > self.max_size:
            zoom_factor = tf.math.divide(self.max_size, max_imgsize)

        fix_height = zoom_factor * height
        fix_width = zoom_factor * width
        resized_img = tf.image.resize(image,
                                    size = [fix_height, fix_width],
                                    method='bilinear')

        if self.obj_gt is None:
            return resized_img, self.obj_gt

        if self.obj_gt is not None:
            bbox = self.obj_gt['bbox']
            bbox = self.resize_bbox(bbox, [height, width], [fix_height, fix_width])
            self.obj_gt['bbox'] =bbox
            return resized_img, self.obj_gt

if __name__ == '__main__':
    # gt = {'bbox': tf.random.uniform([2, 4])}
    # print(gt)
    # p = PreProcess(400, 600, 10, 2, gt)
    p = PreProcess(600, 600, 10, 2)
    path = '/home/user/code/FasterRcnn/dataset/test_img/car.png'
    input = tf.io.read_file(path)
    img_tensor = tf.image.decode_png(input) 
    img_tensor_shape = img_tensor.shape
    img = np.expand_dims(np.array(img_tensor), 0)

    img_shape = np.shape(img)
    origin_width = img_shape[1]
    origin_height = img_shape[2]
    origin_img = deepcopy(img_tensor)

    print(img_shape)
    print(origin_height)
    print(origin_width)
    h1 = p.normalize(img)
    # print(h1.shape)
    h2, h3 = p.resize(h1)
    print(h2.shape)
    print(h3)
    
    