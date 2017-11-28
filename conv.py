#!/home/ty/.virtualenvs/py3tf110/bin/python3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os, datetime
import itertools
from random import randint

def unique_dir_name():
    return os.path.join(os.getcwd(), "%02d-" % randint(0, 99) + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def rescale(aa):
    return (255*(aa-np.min(aa))/(np.max(aa)-np.min(aa)))

def cnn_model_fn(features, labels, mode):
    with tf.device("/device:GPU:0"):
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
              inputs=input_layer,
              filters=8,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              name="myconv1")
        #y = tf.nn.softmax(tf.matmul(x, W) + b)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        print(pool1.shape)
        conv2 = tf.layers.conv2d(
              inputs=pool1,
              filters=8,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              name="myconv2")
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        print(pool2.shape)
        conv3 = tf.layers.conv2d(
              inputs=pool2,
              filters=8,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              name="myconv3")
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        print(pool3.shape)
        conv4 = tf.layers.conv2d(
              inputs=pool3,
              filters=8,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              name="myconv4")
        #y = tf.nn.softmax(tf.matmul(x, W) + b)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

        print(pool4.shape)

        pool_flat = tf.reshape(pool4, [-1, 8])
        dense = tf.layers.dense(inputs=pool_flat, units=20, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
              inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

          # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

          # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

          # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
              "accuracy": tf.metrics.accuracy(
                  labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
              mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    start = time()
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=None,
        shuffle=True)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=unique_dir_name())
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=50000,
        hooks=[logging_hook])
    print("train time: ", time() - start)

    start = time()
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("test time: ", time() - start)
    print(eval_results)

    for i in range(1,5):
        filters = np.transpose(mnist_classifier.get_variable_value('myconv%d/kernel'%i), (2,3,0,1))
        f, axs = plt.subplots(2, 4)
        for ax, e in zip(itertools.chain.from_iterable(axs), filters[0,:,:,:]):
            try:
                print(e.shape)
                ax.matshow(e, cmap='Greys')
            except:
                import ipdb; ipdb.set_trace()
        plt.show()

main()
