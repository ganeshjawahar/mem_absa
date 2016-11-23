import os
import pprint
import tensorflow as tf

from data import read_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 7, "number of hops [7]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("pretrain_file", "data/glove.6B.300d.txt", "pre-trained glove vectors file path [../data/glove.6B.300d.txt]")
flags.DEFINE_string("train_data", "data/Laptop_Train_v2.xml", "train gold data set path [./data/Laptop_Train_v2.xml]")
flags.DEFINE_string("test_data", "data/Laptops_Test_Gold.xml", "test gold data set path [./data/Laptops_Test_Gold.xml]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def init_word_embeddings(word2idx):
  import numpy as np
  wt = np.random.normal(0, FLAGS.init_std, [len(word2idx), FLAGS.edim])
  with open(FLAGS.pretrain_file, 'r') as f:
    for line in f:
      content = line.strip().split()
      if content[0] in word2idx:
        wt[word2idx[content[0]]] = np.array(map(float, content[1:]))
  return wt

def main(_):
  source_count, target_count = [], []
  source_word2idx, target_word2idx = {}, {}

  train_data = read_data(FLAGS.train_data, source_count, source_word2idx, target_count, target_word2idx)
  test_data = read_data(FLAGS.test_data, source_count, source_word2idx, target_count, target_word2idx)
  
  FLAGS.pad_idx = source_word2idx['<pad>']
  FLAGS.nwords = len(source_word2idx)
  FLAGS.mem_size = train_data[4] if train_data[4] > test_data[4] else test_data[4]

  pp.pprint(flags.FLAGS.__flags)

  print('loading pre-trained word vectors...')
  FLAGS.pre_trained_context_wt = init_word_embeddings(source_word2idx)
  FLAGS.pre_trained_target_wt = init_word_embeddings(target_word2idx)

  with tf.Session() as sess:
    model = MemN2N(FLAGS, sess)
    model.build_model()
    model.run(train_data, test_data)  

if __name__ == '__main__':
  tf.app.run()
