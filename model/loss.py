import tensorflow as tf

class Loss(tf.keras.losses):
    def __init__(self,
                proposals,
                groundTruth):
        super(Loss, self).__init__()
        self.proposals = proposals
        self.groundtruth = groundTruth
    
    def RPNRegLoss(self):
        pass

    def RPNClsLoss(self):
        pass

    def DetectRegLoss(self):
        pass

    def DetecClsLoss(self):
        pass
