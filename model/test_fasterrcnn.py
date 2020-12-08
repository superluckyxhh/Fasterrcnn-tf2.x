import numpy as np
import tensorflow as tf
from absl import app
import fasterrcnn

def main():
    input_tensor = tf.random.uniform(shape = [1, 300, 300, 3])

    fasterrcnn_model = fasterrcnn.FasterRcnn()
    rpn_model = fasterrcnn.RPNHead()
    rpn = fasterrcnn.RPNModule()
    res1, res2, res3 = fasterrcnn_model.call(input_tensor)
  
    print('feature shape:', res1.shape)
    print('rpn scores:', res2.shape)
    print('rpn coordinate:', res3.shape)

if __name__ == '__main__':
    main()