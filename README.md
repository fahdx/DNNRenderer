# DNNRenderer


DNN Render is modle for rednering 3D shapes . yet model result not that good (could be not converge yet). the model first reduce dimantionality using 3D Conv, then use MLP to project 3D to 2D.  the idea take from [RenderNet] (https://proceedings.neurips.cc/paper/2018/file/68d3743587f71fbaa5062152985aff40-Paper.pdf).

### model requirement
1-tensorflow 1 \
2-PIL\
3-trimesh\
4-numpy


## training
```sh
$ python main.py
```
model showd result with cross entropy as loss function howerver the model also contain ssim loss function . you can try it.
