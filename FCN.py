from __future__ import print_function
import tensorflow as tf
import numpy as np
from network.fcn_model import loss
import TensorflowUtils as utils
from read_tfrecorder import input_pipeline
import datetime
from six.moves import xrange
import time
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
# tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "./", "Path to vgg model mat")
#tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


MAX_ITERATION = int(1e5 + 1)
IMAGE_SIZE = 100


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)




def main(argv=None):

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        with tf.device('/gpu:3'):


            train_images, train_annotations = input_pipeline("./train.tfrecords", FLAGS.batch_size)
            validation_images, validation_annotations = input_pipeline("./validation.tfrecords", FLAGS.batch_size)
            test_images, test_annotations = input_pipeline("./test.tfrecords", FLAGS.batch_size)

            keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
            image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
            annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE], name="annotation")
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

            pred_annotation,total_loss = loss(image,annotation,keep_probability)
            output_pred = tf.cast(255*pred_annotation, tf.uint8)
            tf.summary.image("input_image", image, max_outputs=2)
            tf.summary.image("ground_truth", tf.cast(tf.expand_dims(255*annotation,3), tf.uint8), max_outputs=2)
            tf.summary.image("pred_annotation", tf.cast(255*pred_annotation, tf.uint8), max_outputs=2)
            tf.summary.scalar("entropy", total_loss)

            trainable_var = tf.trainable_variables()


            grads = optimizer.compute_gradients(total_loss, var_list=trainable_var)

            train_op = optimizer.apply_gradients(grads)


            print("Setting up summary op...")
            summary_op = tf.summary.merge_all()

            # image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
            # if FLAGS.mode == 'train':
            #     train_dataset_reader = dataset.BatchDatset(train_records, image_options)
            # validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
            config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            print("Setting up Saver...")
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

            sess.run(tf.global_variables_initializer())


            ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            if FLAGS.mode == "train":



                for step in xrange(MAX_ITERATION):
                    start_time = time.time()

                    x,y = sess.run([train_images,train_annotations])
                    feed_dict = {image: x, annotation: y, keep_probability: 0.85}

                    sess.run(train_op, feed_dict=feed_dict)


                    duration = time.time() - start_time

                    if step % 20 == 0:

                        train_loss, summary_str = sess.run([total_loss, summary_op], feed_dict=feed_dict)
                        print('step {:d} \t loss = {:.8f}, ({:.3f} sec/step)'.format(step, train_loss, duration))
                        summary_writer.add_summary(summary_str, step)

                    if step % 500 == 0:
                        vx,vy = sess.run([validation_images,validation_annotations])
                        feed_dict = {image: vx, annotation: vy, keep_probability: 1}
                        validation_loss = sess.run(total_loss, feed_dict=feed_dict)
                        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), validation_loss))
                        saver.save(sess, FLAGS.logs_dir + "model.ckpt", step)



    #
    #         elif FLAGS.mode == "visualize":
    #
    #             tx, ty = sess.run([test_images, test_annotations])
    #             feed_dict = {image: tx, annotation: ty, keep_probability: 1.0}
    #             test_loss, pred = sess.run([total_loss, output_pred], feed_dict=feed_dict)
    #
    #
    #
    #             pred = np.squeeze(pred)
    # #
    #             for step in range(FLAGS.batch_size):
    #                 utils.save_image(tx[step].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+step))
    #                 utils.save_image(ty[step].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+step))
    #                 utils.save_image(pred[step].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+step))
    #                 print("Saved image: %d" % step)


            coord.request_stop()
            coord.join(threads)
            sess.close()

if __name__ == "__main__":
    tf.app.run()