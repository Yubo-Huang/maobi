import numpy as np
import tensorflow.compat.v1 as tf

class space():
    
    def __init__(self, shape, maximum, minimum):
        self.shape = shape
        self.maximum = maximum
        self.minimum = minimum
        self.dtype   = tf.float32
        
class state_space(space):
    name = 'state_space'
    def __init__(self, shape, maximum, minimum):
        super().__init__(shape, maximum, minimum)
        
class action_space(space):
    name = 'action_space'
    def __init__(self, shape, maximum, minimum):
        super().__init__(shape, maximum, minimum)