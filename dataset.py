
import os
import tools2

import numpy as np
import trimesh
import mcubes
import math


file = [f for f in os.listdir(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/')
                        if os.path.isfile(os.path.join(
                        '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/',
                        f))]



file_done = [f for f in os.listdir(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/')
                        if os.path.isfile(os.path.join(
                        '/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/',
                        f))]






print(len(file_done))
print('before file length',len(file))


file_new=[]
for f1 in file:

    temp= f1+'-0.png'

    if temp not in file_done:
        file_new.append(f1)
    else:
        print(temp)


print(len(file_new))

for ff in range(len(file_new)):

    y_true_path = '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + file_new[ff]



    y_true = tools2.Data.load_single_voxel_grid(y_true_path, out_vox_res=64)
    y_true=y_true.reshape(64,64,64)




    v, t = mcubes.marching_cubes(y_true, 0)

    mesh1 = trimesh.Trimesh(vertices=v,
                            faces=t, process=True)
    scene = mesh1.scene()
    rotate = trimesh.transformations.rotation_matrix(

        angle=np.radians(2 * math.pi/5)
,
        direction=[5, 0, 0],
        point=scene.centroid)

    trans=trimesh.transformations.translation_matrix([0,0,5])


    for i in range(1):
        trimesh.constants.log.info('Saving image %d', i)

        # # rotate the camera view transform
        # camera_old, _geometry = scene.graph[scene.camera.name]
        # #
        #
        # if i==0:
        #     camera_new = np.dot(rotate, camera_old)
        # if i==1:
        #     camera_new = np.dot(rotate2, camera_old)
        # if i==2:
        #     camera_new = np.dot(rotate3, camera_old)
        #
        # # apply the new transform
        # scene.graph[scene.camera.name] = camera_new

        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        try:
            # increment the file name
            file_name = '/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/'+file_new[ff]+'-' + str(i) + '.png'
            # save a render of the object as a png
            png = scene.save_image(resolution=[512,512], visible=True)
            with open(file_name, 'wb') as f:
                f.write(png)
                f.close()
        except BaseException as E:
            print("unable to save image", str(E))




