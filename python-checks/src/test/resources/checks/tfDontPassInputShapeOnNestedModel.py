import tensorflow as tf

class NonCompliantModel(tf.keras.Model):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))  # Noncompliant {{Remove this `input_shape` argument, it is deprecated.}}
                        #^^^^^^^^^^^^^^^^^^^^^^^^^^^

class CompliantModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

def some_other_function():
    some_other_function()

some_other_function()
