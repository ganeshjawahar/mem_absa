import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
from past.builtins import xrange
import time as tim

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm
        self.pad_idx = config.pad_idx
        self.pre_trained_context_wt = config.pre_trained_context_wt
        self.pre_trained_target_wt = config.pre_trained_target_wt

        self.input = tf.placeholder(tf.int32, [self.batch_size, 1], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, 3], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

        self.show = config.show

        self.hid = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
      self.global_step = tf.Variable(0, name="global_step")

      self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
      self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
      self.ASP = tf.Variable(tf.random_normal([self.pre_trained_target_wt.shape[0], self.edim], stddev=self.init_std))
      self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))
      self.BL_W = tf.Variable(tf.random_normal([2 * self.edim, 1], stddev=self.init_std))
      self.BL_B = tf.Variable(tf.zeros([1, 1]))

      # Location Encoding
      self.T_A = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))
      self.T_B = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))

      # m_i = sum A_ij * x_ij + T_A_i
      Ain_c = tf.nn.embedding_lookup(self.A, self.context)
      Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
      Ain = tf.add(Ain_c, Ain_t)

      # c_i = sum B_ij * u + T_B_i
      Bin_c = tf.nn.embedding_lookup(self.B, self.context)
      Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
      Bin = tf.add(Bin_c, Bin_t)

      ASPin = tf.nn.embedding_lookup(self.ASP, self.input)
      ASPout2dim = tf.reshape(ASPin, [-1, self.edim])
      self.hid.append(ASPout2dim)

      for h in xrange(self.nhop):
        '''
        Bi-linear scoring function for a context word and aspect term
        '''
        til_hid = tf.tile(self.hid[-1], [1, self.mem_size])
        til_hid3dim = tf.reshape(til_hid, [-1, self.mem_size, self.edim])
        a_til_concat = tf.concat(axis=2, values=[til_hid3dim, Ain])
        til_bl_wt = tf.tile(self.BL_W, [self.batch_size, 1])
        til_bl_3dim = tf.reshape(til_bl_wt, [self.batch_size, -1, 2 * self.edim])
        att = tf.matmul(a_til_concat, til_bl_3dim, adjoint_b = True)
        til_bl_b = tf.tile(self.BL_B, [self.batch_size, self.mem_size])
        til_bl_3dim = tf.reshape(til_bl_b, [-1, self.mem_size, 1])
        g = tf.nn.tanh(tf.add(att, til_bl_3dim))
        g_2dim = tf.reshape(g, [-1, self.mem_size])
        P = tf.nn.softmax(g_2dim)

        probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
        Bout = tf.matmul(probs3dim, Bin)
        Bout2dim = tf.reshape(Bout, [-1, self.edim])

        Cout = tf.matmul(self.hid[-1], self.C)
        Dout = tf.add(Cout, Bout2dim)

        if self.lindim == self.edim:
            self.hid.append(Dout)
        elif self.lindim == 0:
            self.hid.append(tf.nn.relu(Dout))
        else:
            F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
            G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
            K = tf.nn.relu(G)
            self.hid.append(tf.concat(axis=1, values=[F, K]))
    def build_model(self):
      self.build_memory()

      self.W = tf.Variable(tf.random_normal([self.edim, 3], stddev=self.init_std))
      z = tf.matmul(self.hid[-1], self.W)

      self.loss = tf.nn.softmax_cross_entropy_with_logits(z, self.target)

      self.lr = tf.Variable(self.current_lr)
      self.opt = tf.train.GradientDescentOptimizer(self.lr)

      params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W, self.ASP, self.BL_W, self.BL_B]
      grads_and_vars = self.opt.compute_gradients(self.loss,params)
      clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                 for gv in grads_and_vars]

      inc = self.global_step.assign_add(1)
      with tf.control_dependencies([inc]):
          self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

      tf.initialize_all_variables().run()

      self.correct_prediction = tf.argmax(z, 1)

    def train(self, data):
      source_data, source_loc_data, target_data, target_label, _ = data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.float32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size, 3]) # one-hot-encoded
      context = np.ndarray([self.batch_size, self.mem_size])
      
      if self.show:
        from utils import ProgressBar
        bar = ProgressBar('Train', max=N)

      rand_idx, cur = np.random.permutation(len(source_data)), 0
      for idx in xrange(N):
        if self.show: bar.next()
        
        context.fill(self.pad_idx)
        time.fill(self.mem_size)
        target.fill(0)
        
        '''
        Initilialize all the padding vector to 0 before backprop.
        TODO: Code is 5x slower due to the following initialization.
        '''
        emb_a = self.A.eval()
        emb_a[self.pad_idx,:] = 0
        emb_b = self.B.eval()
        emb_b[self.pad_idx,:] = 0
        emb_c = self.C.eval()
        emb_c[self.pad_idx,:] = 0
        emb_ta = self.T_A.eval()
        emb_ta[self.mem_size,:] = 0
        emb_tb = self.T_B.eval()
        emb_tb[self.mem_size,:] = 0
        self.sess.run(self.A.assign(emb_a))
        self.sess.run(self.B.assign(emb_b))
        self.sess.run(self.C.assign(emb_c))  
        self.sess.run(self.T_A.assign(emb_ta))
        self.sess.run(self.T_B.assign(emb_tb))

        for b in xrange(self.batch_size):
            m = rand_idx[cur]
            x[b][0] = target_data[m]
            target[b][target_label[m]] = 1
            time[b,:len(source_loc_data[m])] = source_loc_data[m]
            context[b,:len(source_data[m])] = source_data[m]
            cur = cur + 1

        a, loss, self.step = self.sess.run([self.optim,
                                            self.loss,
                                            self.global_step],
                                            feed_dict={
                                                self.input: x,
                                                self.time: time,
                                                self.target: target,
                                                self.context: context})
        cost += np.sum(loss)
      
      if self.show: bar.finish()
      _, train_acc = self.test(data)
      return cost/N/self.batch_size, train_acc

    def test(self, data):
      source_data, source_loc_data, target_data, target_label, _ = data
      N = int(math.ceil(len(source_data) / self.batch_size))
      cost = 0

      x = np.ndarray([self.batch_size, 1], dtype=np.int32)
      time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
      target = np.zeros([self.batch_size, 3]) # one-hot-encoded
      context = np.ndarray([self.batch_size, self.mem_size])
      context.fill(self.pad_idx)

      m, acc = 0, 0
      for i in xrange(N):
        target.fill(0)
        time.fill(self.mem_size)
        context.fill(self.pad_idx)

        emb_a = self.A.eval()
        emb_a[self.pad_idx,:] = 0
        emb_b = self.B.eval()
        emb_b[self.pad_idx,:] = 0
        emb_c = self.C.eval()
        emb_c[self.pad_idx,:] = 0
        emb_ta = self.T_A.eval()
        emb_ta[self.mem_size,:] = 0
        emb_tb = self.T_B.eval()
        emb_tb[self.mem_size,:] = 0
        self.sess.run(self.A.assign(emb_a))
        self.sess.run(self.B.assign(emb_b))
        self.sess.run(self.C.assign(emb_c))       
        self.sess.run(self.T_A.assign(emb_ta))
        self.sess.run(self.T_B.assign(emb_tb))
        
        raw_labels = []
        for b in xrange(self.batch_size):
          x[b][0] = target_data[m]
          target[b][target_label[m]] = 1
          time[b,:len(source_loc_data[m])] = source_loc_data[m]
          context[b,:len(source_data[m])] = source_data[m]
          m += 1
          raw_labels.append(target_label[m])

        loss = self.sess.run([self.loss],
                                        feed_dict={
                                            self.input: x,
                                            self.time: time,
                                            self.target: target,
                                            self.context: context})
        cost += np.sum(loss)

        predictions = self.sess.run(self.correct_prediction, feed_dict={self.input: x,
                                                     self.time: time,
                                                     self.target: target,
                                                     self.context: context})

        for b in xrange(self.batch_size):
          if raw_labels[b] == predictions[b]:
            acc = acc + 1

      return cost, acc/float(len(source_data))

    def run(self, train_data, test_data):
      print('training...')
      self.sess.run(self.A.assign(self.pre_trained_context_wt))
      self.sess.run(self.B.assign(self.pre_trained_context_wt))
      self.sess.run(self.ASP.assign(self.pre_trained_target_wt))
      for idx in xrange(self.nepoch):
        print('epoch '+str(idx)+'...')
        train_loss, train_acc = self.train(train_data)
        test_loss, test_acc = self.test(test_data)
        print('train-loss=%.2f;train-acc=%.2f;test-acc=%.2f;' % (train_loss, train_acc, test_acc))
