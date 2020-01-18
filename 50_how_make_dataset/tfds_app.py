# 模拟数据制作成内存对象数据集
# tfds 加载数据集
import tensorflow as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()  # 启动动态图
print(tfds.list_builders())  # 查看有效的数据集

# 加载数据集
ds_train, ds_test = tfds.load(
    name='mnist',
    split=["train", "test"]
)
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)  # 用tf.data.Dataset 接口数据集

for feature in ds_train.take(1):
    image, label = feature['image'], feature['label']
    print(image, label)
