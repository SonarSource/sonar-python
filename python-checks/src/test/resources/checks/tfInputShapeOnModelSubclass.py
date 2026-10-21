import tensorflow as tf

class NonCompliantModel(tf.keras.Model):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))  # Noncompliant {{Remove this `input_shape` argument, it is deprecated.}}
                        #^^^^^^^^^^^^^^^^^^^^^^^^^^^

class CompliantModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

class UnrelatedClass():
    def __init__(self, input_shape):
        self.input_shape = input_shape
class UnrelatedClass2(UnrelatedClass):
    def __init__(self, input_shape):
        super().__init__(input_shape=(32, 24, 24, 3))

from tf.keras import Model as KerasModel
class NonCompliantModel2(KerasModel):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))  # Noncompliant {{Remove this `input_shape` argument, it is deprecated.}}
                        #^^^^^^^^^^^^^^^^^^^^^^^^^^^

class CompliantModel2(KerasModel):
    def __init__(self):
        super().something_else(input_shape=(32, 24, 24, 3))

import keras
class NonCompliantModel3(keras.Model):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))  # Noncompliant {{Remove this `input_shape` argument, it is deprecated.}}
                        #^^^^^^^^^^^^^^^^^^^^^^^^^^^
class CompliantModel2(KerasModel):
    def some_function(self):
        super().something_else(input_shape=(32, 24, 24, 3))

class FalseNegative1(KerasModel):
    def __init__(self):
        some_var = super()
        some_var.__init__(input_shape=(32, 24, 24, 3)) # FN because we only match on qualified expressions

class UnknownSymbol:
    ...

UnknownSymbol = 42

class NonCompliantModel4(UnknownSymbol):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))


def some_other_function():
    some_other_function()

some_other_function()

class UnrelatedClass3():
    def __init__(self):
        some_other_function().__init__(input_shape=(32, 24, 24, 3))


from tensorflow.keras.models import Model as CanonicalModel

class CanonicalModelSubclass(CanonicalModel):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))  # Noncompliant

class LocalModel:
    pass

class UnrelatedModelSubclass(LocalModel):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))

class PlainBase:
    def __init_subclass__(cls, model_factory=None):
        pass
    def __init__(self, **kwargs):
        pass

class PlainChild(PlainBase, model_factory=CanonicalModel):
    def __init__(self):
        super().__init__(input_shape=(32, 24, 24, 3))
