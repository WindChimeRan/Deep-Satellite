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
tf.flags.DEFINE_integer("validation_batch_size", "4", "batch size for validation")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "./", "Path to vgg model mat")



# resnext
tf.app.flags.DEFINE_integer('cardinality', 2, '''Cadinality, number of paths in each block''')
tf.app.flags.DEFINE_integer('block_unit_depth', 64, '''the depth of each split. 64 for cifar10
in Figure 7 of the paper''')
tf.app.flags.DEFINE_string('bottleneck_implementation', 'b', '''To use Figure 3b or 3c to
implement''')


## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'v0b3_cont', '''A version number defining the directory to
save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 391, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training

# tf.app.flags.DEFINE_integer('train_steps', 40000, '''Total steps that you want to train''')
# tf.app.flags.DEFINE_boolean('is_full_validation', True, '''Validation w/ full validation set or
# a random batch''')
# tf.app.flags.DEFINE_integer('train_batch_size', 128, '''Train batch size''')
# tf.app.flags.DEFINE_integer('validation_batch_size', 125, '''Validation batch size, better to be
# a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 125, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.001, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 40000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 40000, '''At which step to decay the learning rate''')


## The following flags define hyper-parameters modifying the training network
tf.app.flags.DEFINE_integer('num_resnext_blocks', 3, '''How many blocks do you want,
total layers = 3n + 2, the paper used n=3, 29 layers, as demo''')
tf.app.flags.DEFINE_float('weight_decay', 0.0007, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


## If you want to load a checkpoint and continue training

tf.app.flags.DEFINE_string('ckpt_path', 'logs_v0b3/model.ckpt-79999', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')
tf.app.flags.DEFINE_string('test_ckpt_path', 'model_110.ckpt-79999', '''Checkpoint
directory to restore''')



# Module selection

# from network.fcn_model import loss
# tf.flags.DEFINE_string("logs_dir", "logs_fcn_base/", "path to logs directory")

from network.resnext_atrous import loss,iou
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

MAX_ITERATION = int(2e5 + 1)
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
