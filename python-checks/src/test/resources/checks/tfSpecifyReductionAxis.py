from tensorflow import tf

x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

tf.reduce_sum(x, axis=0)
tf.reduce_sum(x, 0)
tf.reduce_sum(x) # Noncompliant
tf.reduce_sum(x, **some_dict)
tf.reduce_sum(x, **{})
tf.reduce_sum(x, *some_list)
tf.reduce_sum(x, *[])
tf.reduce_sum(x, keepdims=True, axis=0)

tf.reduce_mean(x, axis=0)
tf.reduce_mean(x, 0)
tf.reduce_all(x, axis=0)
tf.reduce_euclidean_norm(x, axis=0)
tf.reduce_logsumexp(x, axis=0)
tf.reduce_max(x, axis=0)
tf.reduce_min(x, axis=0)
tf.reduce_std(x, axis=0)
tf.reduce_sum(x, axis=0)
tf.reduce_variance(x, axis=0)
tf.reduce_prod(x, axis=0)
tf.reduce_any(x, axis=0)
tf.reduce_variance(x, axis=0)
tf.reduce_sum(x, axis=0)
tf.reduce_std(x, axis=0)
tf.reduce_euclidean_norm(x, axis=0)
tf.reduce_variance(x, axis=0)
tf.reduce_logsumexp(x, axis=0)
tf.reduce_all(x, axis=0)
tf.reduce_any(x, axis=0)
tf.reduce_min(x, axis=0)
tf.reduce_max(x, axis=0)

# Non compliants
y = tf.reduce_sum(x) # Noncompliant {{Provide a value for the axis argument.}}
   #^^^^^^^^^^^^^
tf.reduce_mean(x) # Noncompliant
tf.reduce_min(x) # Noncompliant
tf.reduce_max(x) # Noncompliant
tf.reduce_std(x) # Noncompliant
tf.reduce_variance(x) # Noncompliant
tf.reduce_prod(x) # Noncompliant
tf.reduce_any(x) # Noncompliant
tf.reduce_all(x) # Noncompliant
tf.reduce_euclidean_norm(x) # Noncompliant
tf.reduce_logsumexp(x) # Noncompliant
tf.reduce_variance(x) # Noncompliant
tf.reduce_std(x) # Noncompliant

def other_name():
    import tensorflow as tf2
    x2 = tf2.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tf2.reduce_sum(x2) # Noncompliant
    tf2.reduce_mean(x2, 2)

    from tensorflow import reduce_mean as rm
    rm(x2) # Noncompliant {{Provide a value for the axis argument.}}
   #^^
    rm(x2, 2)

def unrelated():
    unknown_symbol_function()
