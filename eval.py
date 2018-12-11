# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import sys
from cs import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog


class arguement():
    def __init__(self,loss,model,image):

        self.loss_model = loss
        self.image_size = 256
        self.model_file = model
        self.image_file = image

#FLAGS = arguement('vgg_16',"models/wave.ckpt-done","img/flower.jpg")


def main(models,img_c):
    #tf.app.flags.DEFINE_string("image_file",img_c, "")

    #FLAGS = tf.app.flags.FLAGS
    FLAGS = arguement('vgg_16',models,img_c)
    # Get image's height and width.
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            # Make sure 'generated' directory exists.
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                #start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                #end_time = time.time()
                #tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                #tf.logging.info('Done. Please check %s.' % generated_file)


#if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run()

class mywindow(QtWidgets.QWidget, Ui_MainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.img = 'img/ee.jpg'
    # 定义槽函数
    def openimage(self):

        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        print(imgName)
        self.img = imgName
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(png)
    def styletransfer(self):
        #imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        #print(imgName)

        #tf.app.run()
        #tf.app.flags.DEFINE_string("image_file", self.img, "")
        tf.logging.set_verbosity(tf.logging.INFO)
        print(self.img)
        main("models/wave.ckpt-done",self.img)
        # 利用qlabel显示图片
        imgName = "generated/res.jpg"
        png = QtGui.QPixmap(imgName).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)

    def styletransfer1(self):
        #imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        #print(imgName)

        #tf.app.run()
        #tf.app.flags.DEFINE_string("image_file", self.img, "")
        tf.logging.set_verbosity(tf.logging.INFO)
        print(self.img)
        main("models/denoised_starry.ckpt-done",self.img)
        # 利用qlabel显示图片
        imgName = "E:/styletransfer/generated/res.jpg"
        png = QtGui.QPixmap(imgName).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)

    def styletransfer2(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        print(self.img)
        main("models/cubist.ckpt-done",self.img)
        imgName = "generated/res.jpg"
        png = QtGui.QPixmap(imgName).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())