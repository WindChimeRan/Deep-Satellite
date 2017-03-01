# Deep-Satellite

## data
 The images come from dstl
 10 polygon annotated image in total
### train_x and train_y
 use **vis.py** to draw polygon in **train_y** and visualize the 3 bands images **train_x**

### crop.py
 stride = 100 
 crop 10 image and do 4 fold augumentation to 43560

### drawSeg.py
  a manual annotation tool 
  left button down to draw a node 
  right button down to fill the polygon  
  s to save  
  Esc to quit  
  remember to use F5 to refresh after drawing 

### make dataset
```shell
$ python tmake.py
```
  then **data.tfrecords**
### test dataset
  **read_tfrecorder.py** to read and display the dataset

## pretrain model

[Download vgg19.mat](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

 then move it to **/Deep-Satelitte**

## train

```
python train.py
```

## Acknowledgement
- [fast-neural-style with tensorflow](https://github.com/burness/neural_style_tensorflow/tree/master/fast_neural_style)

- [DeepLab-LargeFOV implemented in tensorflow](https://github.com/DrSleep/tensorflow-deeplab-lfov)
