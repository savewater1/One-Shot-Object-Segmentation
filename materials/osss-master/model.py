import os
import tensorflow as tf
import numpy as np
from FCN.Model import FCN32

class Model:
    TRAIN_MODE="TRAIN"
    TEST_MODE="TEST"

    def __init__(self, 
        inp_size: (int, int), 
        support_img: tf.Tensor=None,  # example masks to learn from
        query_img: tf.Tensor=None,
        mode: str="TEST" # controls dropouts
    ):
        ######## Inputs #########
        if support_img is not None:
            assert support_img.shape[0] == 2 and support_img.shape[2:] == [224, 224, 3]
            self.support_img = support_img
        else:
            self.support_img = tf.placeholder(tf.float32, [2, None, 224, 224, 3])
        
        ######## Model #########
        self.masked_simg = self.support_img[0] * self.support_img[1]
        self.encoder = FCN32(inp_size, 21, inp_img=query_img, mode=mode)
        self.query_img = self.encoder.img
        self.features = self.encoder.get_output('conv7')
        self.generator = tf.keras.applications.vgg16.VGG16(input_tensor=self.masked_simg)
        self.gen_features = tf.matmul(self.generator.layers[-2].output, self.generator.layers[-1].weights[0]) + self.generator.layers[-1].weights[1]
        self.gen_weight_size = 4096
        self.weight_hash_mat = np.zeros(shape=(self.gen_weight_size+1, 1000))
        np.random.seed(0)
        self.weight_hash_mat[
            np.arange(self.weight_hash_mat.shape[0]),
            np.random.choice(1000, size=self.weight_hash_mat.shape[0])
        ] = np.random.choice([-1,1], size=self.weight_hash_mat.shape[0])
        self.weight_hash_tensor = tf.Variable(self.weight_hash_mat.T, dtype=tf.float32, name='w_hash_mat', trainable=False)
        self.weight_hash = tf.matmul(self.gen_features, self.weight_hash_tensor)
        self.gen_weight = self.weight_hash[:, :-1]
        self.gen_bias = self.weight_hash[:, -1]
        weights_shape = tf.shape(self.gen_weight)
        self.pre_act = tf.reduce_sum(
                self.features * tf.reshape(self.gen_weight, (weights_shape[0], 1, 1, weights_shape[1])),
                axis=-1
             ) + tf.reshape(self.gen_bias, (weights_shape[0], 1, 1))
        self.output = tf.nn.sigmoid(self.pre_act)

        ######### Savers #########
        fcn_var_to_save = []
        for i in self.encoder.layers:
            fcn_var_to_save.extend(self.encoder.layers[i].values())
        
        self.fcn_saver = tf.train.Saver(var_list=fcn_var_to_save)

        # Using saver for keras VGG because keras' restore_weights wasn't working
        vgg_var_to_save = []
        for l in self.generator.layers:
            vgg_var_to_save.extend(l.weights)

        self.vgg_saver = tf.train.Saver(var_list=vgg_var_to_save)
    
    def restore_FCN32_from_ckpt(self, sess: tf.Session, ckpt_file: str):
        self.fcn_saver.restore(sess, ckpt_file)
    
    def save_model(self, sess: tf.Session, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        feat_path = os.path.join(dir_path, 'feat')
        gen_path = os.path.join(dir_path, 'gen')
        for p in [feat_path, gen_path]:
            if not os.path.exists(p):
                os.mkdir(p)
        self.fcn_saver.save(sess, os.path.join(feat_path, "model.ckpt"))
        self.vgg_saver.save(sess, os.path.join(gen_path, "model.ckpt"))
        np.save(os.path.join(dir_path, "w_hash_mat.npy"), self.weight_hash_mat)
    
    def restore_model(self, sess: tf.Session, dir_path: str):
        self.restore_FCN32_from_ckpt(sess, os.path.join(os.path.join(dir_path, "feat"), 'model.ckpt'))
        self.vgg_saver.restore(sess, os.path.join(os.path.join(dir_path, "gen"), 'model.ckpt'))
        self.weight_hash_mat = np.load(os.path.join(dir_path, "w_hash_mat.npy"))
        sess.run(tf.assign(self.weight_hash_tensor, self.weight_hash_mat.T))