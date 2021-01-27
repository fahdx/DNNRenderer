import os
import shutil
import numpy as np
import scipy.io
import tensorflow as tf
import tools2 as tools
import mcubes
import trimesh
from PIL import Image ,ImageOps
from os import listdir
from os.path import isfile, join
vox_res64 = 64
vox_rex256 = 64
batch_size = 1
GPU0 = '0'
re_train=True

#########################
config = {}
config['batch_size'] = batch_size
config['vox_res_x'] = vox_res64
config['vox_res_y'] = vox_rex256
config['train_names'] = ['03001627_chair']
for name in config['train_names']:
    # /media/fahd/My Book/Datasets/03001627_chair/train_125_25d_vox256/
    config['X_train_' + name] = '/media/fahd/My Book/Datasets/' + name + '/train_125_25d_vox256/'
    config['Y_train_' + name] = '/media/fahd/My Book/Datasets/' + name + '/train_125_3d_vox256/'

config['test_names'] = ['03001627_chair']
for name in config['test_names']:
    config['X_test_' + name] = '/media/fahd/My Book/Datasets/' + name + '/test_125_25d_vox256/'
    config['Y_test_' + name] = '/media/fahd/My Book/Datasets/' + name + '/test_125_3d_vox256/'

#########################

class Network:
    def __init__(self, demo_only=False):
        if demo_only:
            return  # no need to creat folders
        self.train_mod_dir = './train_mod_selected/train_mod_k_100_n_1000_epoch_10_3drecgan/'
        self.train_sum_dir = './train_sum/'
        self.test_res_dir = './test_res/'
        self.test_sum_dir = './test_sum/'
        self.is_ssim=False

        print ("re_train:", re_train)
        if os.path.exists(self.test_res_dir):
            if re_train:
                print ("test_res_dir and files kept!")
            else:
                shutil.rmtree(self.test_res_dir)
                os.makedirs(self.test_res_dir)
                print ('test_res_dir: deleted and then created!')
        else:
            os.makedirs(self.test_res_dir)
            print ('test_res_dir: created!')

        if os.path.exists(self.train_mod_dir):
            if re_train:
                if os.path.exists(self.train_mod_dir + 'model.cptk.data-00000-of-00001'):
                    print ('model found! will be reused!')
                else:
                    print ('model not found! error!')
                    exit()
            else:
                shutil.rmtree(self.train_mod_dir)
                os.makedirs(self.train_mod_dir)
                print ('train_mod_dir: deleted and then created!')
        else:
            os.makedirs(self.train_mod_dir)
            print ('train_mod_dir: created!')

        if os.path.exists(self.train_sum_dir):
            if re_train:
                print ("train_sum_dir and files kept!")
            else:
                shutil.rmtree(self.train_sum_dir)
                os.makedirs(self.train_sum_dir)
                print ('train_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.train_sum_dir)
            print ('train_sum_dir: created!')

        if os.path.exists(self.test_sum_dir):
            if re_train:
                print ("test_sum_dir and files kept!")
            else:
                shutil.rmtree(self.test_sum_dir)
                os.makedirs(self.test_sum_dir)
                print ('test_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.test_sum_dir)
            print ('test_sum_dir: created!')




    def aeu(self, X):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[-1, vox_res64,vox_res64,vox_res64,1])
            c_e = [1,64,128,256,512]
            s_e = [0,1 , 1, 1, 1]
            layers_e = []
            unet=[]
            layers_e.append(X)
            print('=====================encoder====================')
            for i in range(1,5,1):
                layer = tools.Ops.conv3d(layers_e[-1],k=4,out_c=c_e[i],str=s_e[i],name='e'+str(i))
                layer=tools.Ops.xxlu(layer, label='lrelu')
                layer = tools.Ops.maxpool3d(layer, k=2,s=2,pad='SAME')
                layers_e.append(layer)
                [_, d1, d2, d3, cc] = layers_e[-1].get_shape()
                d1 = int(d1);
                d2 = int(d2);
                d3 = int(d3);
                cc = int(cc)
                temp = tf.reshape(layers_e[-1], [-1, int(d1), int(d2), int(d3) * int(cc)])
                print('temp',temp)
                unet.append(temp)

            ### fc
            [_, d1, d2, d3, cc] = layers_e[-1].get_shape()
            d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
            lfc = tf.reshape(layers_e[-1],[-1, int(d1),int(d2),int(d3)*int(cc)])
            cc=int(d3)*int(cc)
            print('before lfc',lfc)

            layer=tf.contrib.slim.conv2d(inputs=lfc, num_outputs=cc, kernel_size=1, activation_fn=None)
            layer = tools.Ops.xxlu(layer, label='relu')

        with tf.device('/gpu:'+GPU0):
            print('========================decoder===============================')
            lfc = tools.Ops.xxlu(tools.Ops.conv2d(layer, k=4, out_c=cc, str=2, name='conv2_1' ), label='relu')


            c_d = [0,512,256,128,64,16,16,16,8,8,8,4 ]
            s_d = [0,2,2,2,2,2,1,1,1,2,1,1,2]
            layers_d = []
            layers_d.append(lfc)
            for j in range(1,5,1):
                u_net = True
                if u_net:
                   if j >=2 and j<5:

                        layer = tf.concat([layers_d[-1], unet[-j]],axis=-1)

                        layer = tools.Ops.deconv2d(layer, k=2,out_c=c_d[j], str=s_d[j],name='d'+str(len(layers_d)))

                else:
                    layer = tools.Ops.deconv2d(layers_d[-1],k=4,out_c=c_d[j],str=s_d[j],name='d'+str(len(layers_d)))
                layer = tools.Ops.xxlu(layer, label='relu')
                layers_d.append(layer)

            ###
            shortcut=layer
            layer = tools.Ops.conv2d(layer, k=3, out_c=64, str=1, name='res1_64' )
            layer = tools.Ops.conv2d(layer, k=3, out_c=64, str=1, name='res2_64')
            layer = tools.Ops.conv2d(layer, k=3, out_c=64, str=1, name='res3_64')
            layer=tf.add(layer,shortcut)

            layer = tools.Ops.conv2d(layer, k=3, out_c=16, str=1, name='link')
            layer = tools.Ops.xxlu(layer, label='relu')

            shortcut = layer
            layer = tools.Ops.conv2d(layer, k=3, out_c=16, str=1, name='res1_16')
            layer = tools.Ops.conv2d(layer, k=3, out_c=16, str=1, name='res2_16')
            layer = tools.Ops.conv2d(layer, k=3, out_c=16, str=1, name='res3_16')
            layer = tf.add(layer, shortcut)

            layer = tools.Ops.conv2d(layer, k=3, out_c=8, str=1, name='link2')
            layer = tools.Ops.xxlu(layer, label='relu')
            layer = tools.Ops.deconv2d(layer, k=4, out_c=4, str=2, name='lay1')
            layer = tools.Ops.xxlu(layer, label='relu')
            layer = tools.Ops.deconv2d(layer, k=4, out_c=2, str=2, name='lay2')
            layer = tools.Ops.xxlu(layer, label='relu')
            layer = tools.Ops.deconv2d(layer, k=4, out_c=1, str=2, name='dlast')

            ###
            Y_sig = tf.nn.sigmoid(layer)


        return Y_sig



    def build_graph(self):
        self.X = tf.placeholder(shape=[5, vox_res64, vox_res64, vox_res64, 1], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, 512,512,1], dtype=tf.float32)

        with tf.variable_scope('aeu'):
            self.Y_pred = self.aeu(self.X)
            #exit(0)

        with tf.device('/gpu:'+GPU0):
            ################################ ae loss

            Y_ = tf.reshape(self.Y_pred, shape=[-1, 512,512,1])


            gt=tf.reshape(self.Y, shape=[5, 512,512,1])


            y=tf.reshape(Y_, shape=[5,512, 512,1])


            if self.is_ssim:

                y = tf.image.convert_image_dtype(y, tf.float32)
                gt = tf.image.convert_image_dtype(gt, tf.float32)

                self.aeu_loss =tf.image.ssim(gt,y, max_val=1,filter_size=5)
                self.aeu_loss=tf.reduce_mean(self.aeu_loss)
            else:
                self.aeu_loss = tf.reduce_mean(
                    -tf.reduce_mean( gt * tf.log(y + 1e-8), reduction_indices=[1]) -
                    tf.reduce_mean( (1 - gt) * tf.log(1 - y + 1e-8),
                                   reduction_indices=[1]))




            self.aeu_gan_g_loss = 100*self.aeu_loss

        with tf.device('/gpu:'+GPU0):
            aeu_var = [var for var in tf.trainable_variables() if var.name.startswith('aeu')]

            self.aeu_g_optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8).\
                            minimize(self.aeu_gan_g_loss, var_list=aeu_var)


        print (tools.Ops.variable_count())
        self.sum_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0

        self.sess = tf.Session(config=config)
        self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
        self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

        path = self.train_mod_dir
        #path = './Model_released/'   # to retrain our released model
        if os.path.isfile(path + 'model.cptk.data-00000-of-00001'):
            print ('restoring saved model')
            self.saver.restore(self.sess, path + 'model.cptk')
        else:
            print ('initilizing model')
            self.sess.run(tf.global_variables_initializer())

        return 0

    def train(self, data):
        for epoch in range(14,25):

            data.shuffle_X_Y_files(label='train')
            total_train_batch_num = 1355

            file_done = [f for f in os.listdir(
                '/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/')
                         if os.path.isfile(os.path.join(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/',
                    f))]

            print('total_train_batch_num:', total_train_batch_num)
            shift=0
            for i in range(total_train_batch_num):

                #################### training (need to be improved for seting training batch) --[bad code :( ]--
                f1 = file_done[i + shift].split('-')[0]
                f2 = file_done[i + 1 + shift].split('-')[0]
                f3 = file_done[i + 2 + shift].split('-')[0]
                f4 = file_done[i + 3 + shift].split('-')[0]
                f5 = file_done[i + 4 + shift].split('-')[0]

                Y_train_batch1 = tools.Data.load_single_voxel_grid(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + str(f1), out_vox_res=64)

                Y_train_batch1 = Y_train_batch1.reshape((1, 64, 64, 64, 1))

                Y_train_batch2 = tools.Data.load_single_voxel_grid(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + str(f2), out_vox_res=64)

                Y_train_batch2 = Y_train_batch2.reshape((1, 64, 64, 64, 1))

                Y_train_batch3 = tools.Data.load_single_voxel_grid(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + str(f3), out_vox_res=64)

                Y_train_batch3 = Y_train_batch3.reshape((1, 64, 64, 64, 1))

                Y_train_batch4 = tools.Data.load_single_voxel_grid(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + str(f4), out_vox_res=64)

                Y_train_batch4 = Y_train_batch4.reshape((1, 64, 64, 64, 1))

                Y_train_batch5 = tools.Data.load_single_voxel_grid(
                    '/media/fahd/My Book/Datasets/03001627_chair/train_125_3d_vox256/' + str(f5), out_vox_res=64)

                Y_train_batch5 = Y_train_batch5.reshape((1, 64, 64, 64, 1))

                Y_train_batch = np.concatenate(
                    (Y_train_batch1, Y_train_batch2, Y_train_batch3, Y_train_batch4, Y_train_batch5), axis=0)

                print(Y_train_batch.shape)

                np.seterr(divide='ignore', invalid='ignore')
                # the dataset I made for each  shape I made three 3 rotation in y-axis degree=45 for each

                str1 = '-0.png'  # original pose
                str2 = '-1.png'  # first rotation 45
                str3 = '-2.png'  # second rotation 90

                # please note model cant render the images could be due to rotation!

                f1 += str1
                f2 += str1
                f3 += str1
                f4 += str1
                f5 += str1

                shift += 4

                # Open the image form working directory
                # for each image norm from  0-255 to 0-1

                image = Image.open('/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/' + f1)

                image = ImageOps.grayscale(image)
                img0 = np.array(image, dtype=np.float32)
                img0 = img0 / 255
                img0 = img0.reshape((1, 512, 512, 1))

                image = Image.open('/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/' + f2)

                image = ImageOps.grayscale(image)
                img1 = np.array(image, dtype=np.float32)
                img1 = img1 / 255
                img1 = img1.reshape((1, 512, 512, 1))

                image = Image.open('/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/' + f3)

                image = ImageOps.grayscale(image)
                img2 = np.array(image, dtype=np.float32)
                img2 = img2 / 255
                img2 = img2.reshape((1, 512, 512, 1))

                image = Image.open('/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/' + f4)

                image = ImageOps.grayscale(image)
                img3 = np.array(image, dtype=np.float32)
                img3 = img3 / 255
                img3 = img3.reshape((1, 512, 512, 1))

                image = Image.open('/media/fahd/My Book/Datasets/03001627_chair/train_image_3D_fixed_view/' + f5)

                image = ImageOps.grayscale(image)
                img4 = np.array(image, dtype=np.float32)
                img4 = img4 / 255
                img4 = img4.reshape((1, 512, 512, 1))

                imgs = np.concatenate((img0, img1, img2, img3, img4), axis=0)

                self.sess.run(self.aeu_g_optim, feed_dict={self.X: Y_train_batch, self.Y: imgs})

                y_pred_1, aeu_loss_c, sum_train = self.sess.run(
                    [self.Y_pred, self.aeu_loss, self.sum_merged],
                    feed_dict={self.X: Y_train_batch, self.Y: imgs})

                if i % 200 == 0:
                    self.sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
                print('ep:', epoch, 'i:', i, 'train aeu loss:', aeu_loss_c)

                #################### testing
                if i % 50 == 0:
                    ###########################

                    # gradient between 0 and 1 for 256*256

                    for sample in range(5):
                        img0 = imgs[sample, :, :, :]
                        mat = img0.reshape((512, 512))
                        mat = np.clip(255 * mat, 0, 255).astype(np.uint8)

                        # Creates PIL image
                        img = Image.fromarray(mat, 'L')
                        img.save(
                            "output/gt/img0 - epoch" + str(epoch) + "- i" + str(i) + " - sample" + str(sample) + ".png",
                            "png")

                        # reshape to 2d
                        print(y_pred_1.shape)
                        img0 = y_pred_1[sample, :, :, :]
                        mat = np.reshape(img0, (512, 512))
                        mat = np.clip(255 * mat, 0, 255).astype(np.uint8)

                        # Creates PIL image
                        img = Image.fromarray(mat, 'L')
                        img.save(
                            "output/img0 - epoch" + str(epoch) + "- i" + str(i) + " - sample" + str(sample) + ".png",
                            "png")

                    ##########################

                #### model saving
                if i % 600 == 0 and i > 0:
                    self.saver.save(self.sess, save_path=self.train_mod_dir + 'model.cptk')
                    print('ep:', epoch, 'i:', i, 'model saved!')

            data.stop_queue = True

            # I used to create image (gt)  in flow however I found its very slow .
            def render(self, v, t):
                mesh1 = trimesh.Trimesh(vertices=v,
                                        faces=t, process=True)
                scene = mesh1.scene()
                rotate = trimesh.transformations.rotation_matrix(
                    angle=45,
                    direction=[0, -1, 0],
                    point=scene.centroid)

                for i in range(3):
                    trimesh.constants.log.info('Saving image %d', i)

                    # rotate the camera view transform
                    camera_old, _geometry = scene.graph[scene.camera.name]

                    camera_new = np.dot(rotate, camera_old)

                    # apply the new transform
                    scene.graph[scene.camera.name] = camera_new

                    # is passed don't save the image
                    try:
                        # increment the file name
                        file_name = './render/v_' + str(i) + '.png'
                        # save a render of the object as a png
                        png = scene.save_image(resolution=[512, 512], visible=True)
                        with open(file_name, 'wb') as f:
                            f.write(png)
                            f.close()
                    except BaseException as E:
                        print("unable to save image", str(E))
            #########################
        if __name__ == '__main__':
            data = tools.Data(config)
            data.daemon = True
            data.start()
            net = Network()
            net.build_graph()
            net.train(data)


