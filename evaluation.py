import tensorflow as tf
from read_tfrecorder import input_pipeline


# from network.fcn_atrous import loss,iou
from network.resnext_atrous import loss,iou
FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_string("model_dir", "./", "Path to vgg model mat")
# tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
# tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")

IMAGE_SIZE = 100
TEST_SIZE = 1821

dtype = tf.float32

def evaluate():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
      with tf.device('/gpu:3'):

            test_images, test_annotations = input_pipeline("./test.tfrecords", FLAGS.batch_size)

            keep_probability = tf.placeholder(dtype, name="keep_probabilty")
            image = tf.placeholder(dtype, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
            annotation = tf.placeholder(dtype, shape=[None, IMAGE_SIZE, IMAGE_SIZE], name="annotation")


            pred_annotation, total_loss = loss(image, annotation, 1.0)
            iou_cul = iou(pred_annotation, annotation)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            print("Setting up Saver...")
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")
            else:
                print("No checkpoint file found")
                return



            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            MAX_STEP = TEST_SIZE/FLAGS.batch_size

            iou_total = 0

            for step in range(MAX_STEP):

                x, y = sess.run([test_images, test_annotations])
                feed_dict = {image: x, annotation: y, keep_probability: 1.0}
                iou_test_batch = sess.run(iou_cul,feed_dict=feed_dict)
                iou_total += iou_test_batch

            mIoU = iou_total/MAX_STEP

            print("mIoU = ", mIoU)

            coord.request_stop()
            coord.join(threads)
            sess.close()

    return mIoU

def main(argv=None):

    evaluate()

if __name__ == '__main__':

    tf.app.run()