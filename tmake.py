from data.crop import crop
from data.img2tfrecords import img2tfrecords

a = crop()
a.mkdir()
a.set_stride(100)
a.cropAndSave()

b = img2tfrecords()
b.img2bytes()
