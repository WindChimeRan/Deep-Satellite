from data.crop import crop
from data.img2tfrecords import *
from  data.preprocess import  wash_data,split_data
import argparse
import shutil

parser = argparse.ArgumentParser(description="initialize the dataset in tfrecords")
parser.add_argument("--stride", type=int, default=100,
                    help="crop the image into stride*stride*channel")

args = parser.parse_args()
stride = args.stride

a = crop()
a.mkdir()
a.set_stride(stride)
a.cropAndSave()
wash_data()
test_files_name,validation_files_name,train_files_name = split_data()
img2bytes("./train.tfrecords",train_files_name)
img2bytes("./validation.tfrecords",validation_files_name)
img2bytes("./test.tfrecords",test_files_name)

shutil.rmtree('data/crop_x')
shutil.rmtree('data/crop_y')