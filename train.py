from __future__ import print_function
import tensorflow as tf
from read_tfrecorder import input_pipeline
from evaluation import evaluate
import datetime
import time


from email_qq import send_email

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("frozen_rate", "0", "retrain VGG19 layers")
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("validation_batch_size", "250", "batch size for validation")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "./", "Path to vgg model mat")

# Module selection

# from network.fcn_model import loss
# tf.flags.DEFINE_string("logs_dir", "logs_fcn_base/", "path to logs directory")

from network.fcn_atrous import loss,iou
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

MAX_ITERATION = int(3e5 + 1)
# MAX_ITERATION = int(20+1)

# 2e5+2 : mIoU = 0.89067960794974155
# 3e5+3 : mIoU = 0.89497932187476004

IMAGE_SIZE = 100

dtype = tf.float32



def main(argv=None):


    train_start_date = time.asctime()

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        with tf.device('/gpu:3'):


            train_images, train_annotations = input_pipeline("./train.tfrecords", FLAGS.batch_size)
            validation_images, validation_annotations = input_pipeline("./validation.tfrecords", FLAGS.validation_batch_size)
            # test_images, test_annotations = input_pipeline("./test.tfrecords", FLAGS.batch_size)

            keep_probability = tf.placeholder(dtype, name="keep_probabilty")

            image = tf.placeholder(dtype, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
            annotation = tf.placeholder(dtype, shape=[None, IMAGE_SIZE, IMAGE_SIZE], name="annotation")

            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)


            pred_annotation,total_loss = loss(image,annotation,keep_probability)
            iou_cul = iou(pred_annotation,annotation)

            tf.summary.image("input_image", image, max_outputs=2)
            tf.summary.image("ground_truth", tf.cast(tf.expand_dims(255*annotation,3), tf.uint8), max_outputs=2)
            tf.summary.image("pred_annotation", tf.cast(255*pred_annotation, tf.uint8), max_outputs=2)
            tf.summary.scalar("entropy", total_loss)
            tf.summary.scalar("IOU", iou_cul)

            trainable_var = tf.trainable_variables()

            grads = optimizer.compute_gradients(total_loss, var_list=trainable_var)

            train_op = optimizer.apply_gradients(grads)


            print("Setting up summary op...")
            summary_op = tf.summary.merge_all()

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

            # if FLAGS.mode == "train":

            for step in range(MAX_ITERATION):
                start_time = time.time()

                x,y = sess.run([train_images,train_annotations])
                feed_dict = {image: x, annotation: y, keep_probability: 0.85}

                sess.run(train_op, feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % 10 == 0:

                    train_loss = sess.run(total_loss, feed_dict=feed_dict)
                    print('step {:d} \t loss = {:.8f}, ({:.3f} sec/step)'.format(step, train_loss, duration))
                if step % 50 == 0:
                    vx,vy = sess.run([validation_images,validation_annotations])
                    feed_dict = {image: vx, annotation: vy, keep_probability: 1.0}
                    iou_validation, validation_loss,summary_str = sess.run([iou_cul,total_loss,summary_op], feed_dict=feed_dict)
                    print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), validation_loss))
                    print("%s ---> Validation_IOU: %g" % (datetime.datetime.now(),iou_validation))
                    summary_writer.add_summary(summary_str, step)

                if step % 50000 == 0:
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", step)



            coord.request_stop()
            coord.join(threads)
            sess.close()


            result = evaluate()
            result = "After "+ str(MAX_ITERATION) + " steps\n" + "mIoU = " + str(result) + "\n"
            result += ('start at '+train_start_date+'\n')
            result += ('end at '+time.asctime()+'\n')
            send_email(result)


if __name__ == "__main__":
    tf.app.run()
