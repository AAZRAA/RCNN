import os
import numpy as np
import tensorflow as tf
from eval.evaluate import accuracy
from tensorflow.contrib import slim
from loss.loss import cross_entropy_loss
 
 
class RCNN(object):
    def __init__(self,
                 num_classes,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 epoch,
                 dropout_keep_prob,
                 context_dim,
                 hidden_dim):
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.epoch = epoch
        self.dropout_keep_prob = dropout_keep_prob
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.model()
 
    def model(self):
        # 词向量映射
        with tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
 
        # Recurrent Structure（CNN）
        with tf.name_scope("bi_rnn"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.context_dim)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=embedding_inputs,
                                                                             dtype=tf.float32)
 
        with tf.name_scope("context"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")
 
        with tf.name_scope("word_representation"):
            y2 = tf.concat([c_left, embedding_inputs, c_right], axis=2, name="word_representation")
            embedding_size = 2 * self.context_dim + self.embedding_dim
 
        # max_pooling层
        with tf.name_scope("max_pooling"):
            fc = tf.layers.dense(y2, self.hidden_dim, activation=tf.nn.relu, name='fc1')
            fc_pool = tf.reduce_max(fc, axis=1)
 
        # output层
        with tf.name_scope("output"):
            self.logits = tf.layers.dense(fc_pool, self.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")
 
        # 损失函数
        self.loss = cross_entropy_loss(logits=self.logits, labels=self.input_y)
 
        # 优化函数
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)
 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optim = slim.learning.create_train_op(total_loss=self.loss, optimizer=optimizer, update_ops=update_ops)
 
        # 准确率
        self.acc = accuracy(logits=self.logits, labels=self.input_y)
 
    def fit(self, train_x, train_y, val_x, val_y, batch_size):
        # 创建模型保存路径
        if not os.path.exists('./saves/rcnn'): os.makedirs('./saves/rcnn')
        if not os.path.exists('./train_logs/rcnn'): os.makedirs('./train_logs/rcnn')
 
        # 开始训练
        train_steps = 0
        best_val_acc = 0
        # summary
        tf.summary.scalar('val_loss', self.loss)
        tf.summary.scalar('val_acc', self.acc)
        merged = tf.summary.merge_all()
 
        # 初始化变量
        sess = tf.Session()
        writer = tf.summary.FileWriter('./train_logs/rcnn', sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
 
        for i in range(self.epoch):
            batch_train = self.batch_iter(train_x, train_y, batch_size)
            for batch_x, batch_y in batch_train:
                train_steps += 1
                feed_dict = {self.input_x: batch_x, self.input_y: batch_y}
                _, train_loss, train_acc = sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)
 
                if train_steps % 1000 == 0:
                    feed_dict = {self.input_x: val_x, self.input_y: val_y}
                    val_loss, val_acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)
 
                    summary = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(summary, global_step=train_steps)
 
                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        saver.save(sess, "./saves/rcnn/", global_step=train_steps)
 
                    msg = 'epoch:%d/%d,train_steps:%d,train_loss:%.4f,train_acc:%.4f,val_loss:%.4f,val_acc:%.4f'
                    print(msg % (i, self.epoch, train_steps, train_loss, train_acc, val_loss, val_acc))
 
        sess.close()
 
    def batch_iter(self, x, y, batch_size=32, shuffle=True):
        """
        生成batch数据
        :param x: 训练集特征变量
        :param y: 训练集标签
        :param batch_size: 每个batch的大小
        :param shuffle: 是否在每个epoch时打乱数据
        :return:
        """
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
 
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_len))
            x_shuffle = x[shuffle_indices]
            y_shuffle = y[shuffle_indices]
        else:
            x_shuffle = x
            y_shuffle = y
        for i in range(num_batch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_len)
            yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])
 
    def predict(self, x):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./saves/rcnn/')
        saver.restore(sess, ckpt.model_checkpoint_path)
 
        feed_dict = {self.input_x: x}
        logits = sess.run(self.logits, feed_dict=feed_dict)
        y_pred = np.argmax(logits, 1)
        return y_pred
