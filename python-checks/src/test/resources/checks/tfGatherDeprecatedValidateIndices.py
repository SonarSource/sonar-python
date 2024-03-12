import tensorflow as tf

unrelated_unknown_function()
def known_function(): ...
known_function()

x = tf.constant([[1, 2], [3, 4]])
y = tf.gather(x, [1], validate_indices=True)  # Noncompliant {{`validate_indices` is deprecated.}}
                     #^^^^^^^^^^^^^^^^^^^^^
y2 = tf.gather(x, [1], validate_indices=False) # Noncompliant {{`validate_indices` is deprecated.}}
                      #^^^^^^^^^^^^^^^^^^^^^^
y3 = tf.gather(x, [1], validate_indices=None)  # Noncompliant {{`validate_indices` is deprecated.}}
                      #^^^^^^^^^^^^^^^^^^^^^
y4 = tf.gather(x, [1])
y5 = tf.gather(x, [1], True) # Noncompliant {{`validate_indices` is deprecated.}}
                      #^^^^
y6 = tf.gather(x, [1], False) # Noncompliant {{`validate_indices` is deprecated.}}
                      #^^^^^
y7 = tf.gather(x, [1], None) # Noncompliant {{`validate_indices` is deprecated.}}
                      #^^^^

def other_name_tensorflow():
    import tensorflow as tf2
    a = tf2.constant([[1, 2], [3, 4]])
    b = tf2.gather(x, [1], validate_indices=True)  # Noncompliant {{`validate_indices` is deprecated.}}
                          #^^^^^^^^^^^^^^^^^^^^^
    from tensorflow import gather as some_other_name
    c = some_other_name(x, [1], validate_indices=True)  # Noncompliant {{`validate_indices` is deprecated.}}
                               #^^^^^^^^^^^^^^^^^^^^^
