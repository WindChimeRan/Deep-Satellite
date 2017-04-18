from data.crop import crop
from data.img2tfrecords import img2tfrecords
from  data.preprocess import  wash_data
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

b = img2tfrecords()
b.img2bytes()

#shutil.rmtree('data/crop_x')
#shutil.rmtree('data/crop_y')