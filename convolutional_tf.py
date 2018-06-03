import os
import model
import tensorflow as tf
import sys
import numpy as np

import input_data

if "RESULT_DIR" in os.environ:
    result_dir = os.environ['RESULT_DIR']
else:
    import tempfile
    import shutil
    result_dir = tempfile.mkdtemp()

data_dir = os.environ.get('DATA_DIR', os.getcwd())

train_images_file = ""
train_labels_file = ""
test_images_file = ""
test_labels_file = ""
VALIDATION_SIZE = 5000


def main(argv):
    if len(argv) < 5:
        sys.exit("Not enough arguments provided.")

    global train_images_file, train_labels_file, test_images_file, test_labels_file, custom_images_in_bucket

    i = 1
    while i <= 10:
        arg = str(argv[i])
        if arg == "--trainImagesFile":
            train_images_file = str(argv[i + 1])
        elif arg == "--trainLabelsFile":
            train_labels_file = str(argv[i + 1])
        elif arg == "--testImagesFile":
            test_images_file = str(argv[i + 1])
        elif arg == "--testLabelsFile":
            test_labels_file = str(argv[i + 1])
        elif arg == "--customImagesInBucket":
            custom_images_in_bucket = str(argv[i + 1])
        i += 2


if __name__ == "__main__":
    main(sys.argv)


class DataSets(object):
    pass


data_sets = DataSets()

train_images_file = os.path.join(data_dir, train_images_file)
train_labels_file = os.path.join(data_dir,train_labels_file)
test_images_file = os.path.join(data_dir, test_images_file)
test_labels_file = os.path.join(data_dir, test_labels_file)

train_images, train_labels, test_images, test_labels = input_data.read_data_sets_from_gzip_file(train_images_file, train_labels_file, test_images_file, test_labels_file,one_hot=True)

if custom_images_in_bucket == 'True':
    train_images_custom_images, train_labels_custom_images = input_data.read_data_sets_from_png_file(data_dir, None, True)
    if type(train_images_custom_images) is np.ndarray and type(train_labels_custom_images) is np.ndarray:
        final_train_images = np.vstack([train_images, train_images_custom_images])
        final_train_labels = np.vstack([train_labels, train_labels_custom_images])
    else:
        final_train_images = train_images
        final_train_labels = train_labels
else:
    final_train_images = train_images
    final_train_labels = train_labels

print('total number of images used for training : ', len(final_train_images))
# set aside a few images for validation and tuning
validation_images = final_train_images[:VALIDATION_SIZE]
validation_labels = final_train_labels[:VALIDATION_SIZE]

# train images and labels
train_images = final_train_images[VALIDATION_SIZE:]
train_labels = final_train_labels[VALIDATION_SIZE:]

# update data_sets class
data_sets.train = input_data.DataSet(train_images, train_labels)
data_sets.validation = input_data.DataSet(validation_images, validation_labels)
data_sets.test = input_data.DataSet(test_images, test_labels)

data = data_sets

# model
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    path = saver.save(
        sess, os.path.join(result_dir, 'convolutional.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
