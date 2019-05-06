import tensorflow as tf

arr = tf.random_normal([3, 4, 5])
arr = arr[1:, :, :]
print(arr)
