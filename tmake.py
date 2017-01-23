from crop import crop
from img2tfrecords import img2tfrecords

a = crop()
a.mkdir()
a.set_stride(100)
a.cropAndSave()

b = img2tfrecords()
b.mkdir()
b.img2bytes()

print("success!")
