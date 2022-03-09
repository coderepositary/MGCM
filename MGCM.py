#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import csv
import shutil
from PIL import Image
from scipy.misc import imsave
from text_cnn import TextCNN
from datagenerator import ImageDataGenerator
from only_textual_datagenerator import TextualDataGenerator
from datetime import datetime
import pickle
import time
import sys
from tensorflow.contrib.data import Iterator
from model import *
"""
Configuration Part.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
# Path to the textfiles for the trainings and validation set
option="/home/liujinhuan/horse2zebra/horse2zebra/new_ijk_data1/"
train_file = option+'train_ijk_shuffled_811.txt'
val_file = option+'valid_ijk_shuffled_811.txt'
test_file = option+'test_ijk_shuffled_811.txt'
# Learning params
num_epochs = 100
batch_size = 420
max_l=43
dropout_rate = 0.5
num_classes = 2
filters=[2, 3, 4, 5]
hidden_units = [100, 2]
img_w = 300
EPS = 1e-12
# How often we want to write the tf.summary data to disk
display_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp2/finetune_alexnet/tensorboard"
checkpoint_path = "tmp2/tune_alexnet/checkpoints"

def get_idx_from_sent(sent, word_idx_map, max_l):
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(word_idx_map, max_l):
    train_i_idx, train_j_idx, train_k_idx = [], [], []
    valid_i_idx, valid_j_idx, valid_k_idx = [], [], []
    test_i_idx, test_j_idx, test_k_idx = [], [], []

    print('loading text data')
    print('now():' + str(datetime.now()))
    train_i,train_j,train_k = pickle.load(
        open(option+"AUC_new_dataset_train_811.pkl", "rb"))
    valid_i,valid_j,valid_k = pickle.load(
        open(option+"AUC_new_dataset_valid_811.pkl", "rb"))
    print("valid_i.shape",len(valid_i[0]))
    test_i,test_j,test_k = pickle.load(
        open(option+"AUC_new_dataset_test_811.pkl", "rb"))
    print('text data loaded')
    print('now():' + str(datetime.now()))

    for i in range(len(train_i)):
        train_sent_i_idx = get_idx_from_sent(train_i[i], word_idx_map, max_l)
        train_sent_j_idx = get_idx_from_sent(train_j[i], word_idx_map, max_l)
        train_sent_k_idx = get_idx_from_sent(train_k[i], word_idx_map, max_l)
        train_i_idx.append(train_sent_i_idx)
        train_j_idx.append(train_sent_j_idx)
        train_k_idx.append(train_sent_k_idx)

    for i in range(len(valid_i)):
        valid_sent_i_idx = get_idx_from_sent(valid_i[i], word_idx_map, max_l)
        valid_sent_j_idx = get_idx_from_sent(valid_j[i], word_idx_map, max_l)
        valid_sent_k_idx = get_idx_from_sent(valid_k[i], word_idx_map, max_l)
        valid_i_idx.append(valid_sent_i_idx)
        valid_j_idx.append(valid_sent_j_idx)
        valid_k_idx.append(valid_sent_k_idx)

    for i in range(len(test_i)):
        test_sent_i_idx = get_idx_from_sent(test_i[i], word_idx_map, max_l)
        test_sent_j_idx = get_idx_from_sent(test_j[i], word_idx_map, max_l)
        test_sent_k_idx = get_idx_from_sent(test_k[i], word_idx_map, max_l)
        test_i_idx.append(test_sent_i_idx)
        test_j_idx.append(test_sent_j_idx)
        test_k_idx.append(test_sent_k_idx)

    train_i_idx = np.array(train_i_idx, dtype="int")
    train_j_idx = np.array(train_j_idx, dtype="int")
    train_k_idx = np.array(train_k_idx, dtype="int")
    valid_i_idx = np.array(valid_i_idx, dtype="int")
    valid_j_idx = np.array(valid_j_idx, dtype="int")
    valid_k_idx = np.array(valid_k_idx, dtype="int")
    test_i_idx = np.array(test_i_idx, dtype="int")
    test_j_idx = np.array(test_j_idx, dtype="int")
    test_k_idx = np.array(test_k_idx, dtype="int")

    return [train_i_idx,train_j_idx,train_k_idx,valid_i_idx,valid_j_idx,valid_k_idx,test_i_idx,test_j_idx,test_k_idx]


print ("loading w2v data...")
text_x = pickle.load(open('/home/liujinhuan/liujinhuan/cloth.binary.p', 'rb'),encoding='iso-8859-1')
text_revs, text_W, text_W2, text_word_idx_map, text_vocab = text_x[0], text_x[1], text_x[2], text_x[3], text_x[4]
datasets = make_idx_data(text_word_idx_map, max_l)
train_text_i,train_text_j,train_text_k = datasets[0], datasets[1], datasets[2]
valid_text_i,valid_text_j,valid_text_k = datasets[3], datasets[4], datasets[5]
test_text_i,test_text_j,test_text_k= datasets[6], datasets[7], datasets[8]

"""
Main Part of the finetuning Script.
"""
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 shuffle=False)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  shuffle=False)
    test_data = ImageDataGenerator(test_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  shuffle=False)
    tra_text_i = TextualDataGenerator(train_text_i,
                                    batch_size_text=int(batch_size/3))
    tra_text_j = TextualDataGenerator(train_text_j,
                                    batch_size_text=int(batch_size/3))
    tra_text_k = TextualDataGenerator(train_text_k,
                                    batch_size_text=int(batch_size/3))
    val_text_i = TextualDataGenerator(valid_text_i,
                                    batch_size_text=int(batch_size/3))
    val_text_j = TextualDataGenerator(valid_text_j,
                                    batch_size_text=int(batch_size/3))
    val_text_k = TextualDataGenerator(valid_text_k,
                                    batch_size_text=int(batch_size/3))
    tes_text_i = TextualDataGenerator(test_text_i,
                                    batch_size_text=int(batch_size/3))
    tes_text_j = TextualDataGenerator(test_text_j,
                                    batch_size_text=int(batch_size/3))
    tes_text_k = TextualDataGenerator(test_text_k,
                                    batch_size_text=int(batch_size/3))

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    i_text_ite = Iterator.from_structure(tra_text_i.data.output_types,
                                         tra_text_i.data.output_shapes)
    j_text_ite = Iterator.from_structure(tra_text_j.data.output_types,
                                         tra_text_j.data.output_shapes)
    k_text_ite = Iterator.from_structure(tra_text_k.data.output_types,
                                         tra_text_k.data.output_shapes)
    print("tr_data.data.output_shapes",tr_data.data.output_shapes)

    next_batch = iterator.get_next()
    next_batchi = i_text_ite.get_next()
    next_batchj = j_text_ite.get_next()
    next_batchk = k_text_ite.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)
test_init_op = iterator.make_initializer(test_data.data)
tra_text_i_init_op = i_text_ite.make_initializer(tra_text_i.data)
tra_text_j_init_op = j_text_ite.make_initializer(tra_text_j.data)
tra_text_k_init_op = k_text_ite.make_initializer(tra_text_k.data)
val_text_i_init_op = i_text_ite.make_initializer(val_text_i.data)
val_text_j_init_op =j_text_ite.make_initializer(val_text_j.data)
val_text_k_init_op = k_text_ite.make_initializer(val_text_k.data)
tes_text_i_init_op = i_text_ite.make_initializer(tes_text_i.data)
tes_text_j_init_op = j_text_ite.make_initializer(tes_text_j.data)
tes_text_k_init_op = k_text_ite.make_initializer(tes_text_k.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
text_x_i = tf.placeholder(tf.int32,[int(batch_size/3),train_text_i.shape[1]])
text_x_j = tf.placeholder(tf.int32,[int(batch_size/3),train_text_j.shape[1]])
text_x_k = tf.placeholder(tf.int32,[int(batch_size/3),train_text_k.shape[1]])

keep_prob = tf.placeholder(tf.float32)
outfile = "record2/visual"+ "batch_size_" + str(batch_size) + str(datetime.now().strftime('%H-%M-%S') )+".txt"
with tf.variable_scope("model_texti") as scope:
    model_texti = TextCNN(embedding_weights=text_W,input=text_x_i,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)
with tf.variable_scope("model_textj") as scope:
    model_textj = TextCNN(embedding_weights=text_W,input=text_x_j,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)
with tf.variable_scope("model_textj",reuse=True) as scope:
    model_textk = TextCNN(embedding_weights=text_W,input=text_x_k,filter_sizes=filters,embedding_size=img_w,sequence_length=train_text_i.shape[1],dropout_keep_prob=keep_prob)

#record by txt
file = open(outfile, "w+")
count = 0
best_validation_auc_score = 0.0
for _learning_rate in [0.0002]:
    for lamda in [0.5]:
        for n_hidden in [2**7]:
            count = count+1
            learning_rate = _learning_rate
            # lamda=_lamda
            print("lamda: {} learning_rate:{} n_hidden:{}\n".format(lamda,learning_rate,n_hidden))
            file.write("lamda: {} learning_rate:{} n_hidden:{}\n".format(lamda,learning_rate,n_hidden))
            file.write("begin time:{}".format(datetime.now()))

            input_A = x[0:batch_size:3]
            input_B = x[1:batch_size:3]
            input_B1 = x[2:batch_size:3]
            text_output11 = model_texti.h_drop
            text_output12 = model_textj.h_drop
            text_output13 = model_textk.h_drop

            text_W1 = tf.get_variable("generator_bpr_text_W1"+str(count), shape=(400,n_hidden), dtype=tf.float32,trainable=True,)
            text_W2 = tf.get_variable("generator_bpr_text_W2"+str(count), shape=(400,n_hidden),dtype=tf.float32,trainable=True,)
            text_b1 = tf.get_variable("generator_bpr_text_tb1"+str(count), shape=(n_hidden,),dtype=tf.float32,trainable=True, )
            text_b2 = tf.get_variable("generator_bpr_text_tb2"+str(count), shape=(n_hidden,),dtype=tf.float32,trainable=True, )
            text_output1 = tf.sigmoid(tf.matmul(text_output11, text_W1) + text_b1)
            text_output2 = tf.sigmoid(tf.matmul(text_output12, text_W2) + text_b2)
            text_output3 = tf.sigmoid(tf.matmul(text_output13, text_W2) + text_b2)

            with tf.variable_scope("Model") as scope:
                enc_A,enc_Bg,fake_B = generator(input_A,text_output11,gf_dim=64, reuse=False, name="generator_encoder_A2Bg"+str(count))
                real_B = discriminator(input_A,input_B, df_dim=64, reuse=False, name="discriminator_B"+str(count)) #判别B
                fake_rec_B = discriminator(input_A,fake_B, df_dim=64, reuse=True, name="discriminator_B"+str(count)) #判别生成的B
                enc_B = encoder(input_B,gf_dim=64, reuse=False, name="generator_encoder_B"+str(count))
                enc_B1 = encoder(input_B1,gf_dim=64, reuse=True, name="generator_encoder_B"+str(count))

            it_sim_ij = tf.reduce_sum(tf.abs(enc_Bg-enc_B),[1, 2,3])
            it_sim_ik = tf.reduce_sum(tf.abs(enc_Bg-enc_B1),[1, 2,3])  #1*batch

            enc_A_W = tf.get_variable('generator_bpr_A_W'+str(count), [512,n_hidden],dtype=tf.float32,trainable=True)
            enc_A_b = tf.get_variable('generator_bpr_A_b'+str(count), [n_hidden],dtype=tf.float32,trainable=True)
            enc_B_W = tf.get_variable('generator_bpr_B_W'+str(count),[512,n_hidden],dtype=tf.float32,trainable=True)
            enc_B_b = tf.get_variable('generator_bpr_B_b'+str(count),[n_hidden],dtype=tf.float32,trainable=True)

            enc_A =tf.sigmoid(tf.matmul(tf.reshape(tf.nn.avg_pool(value = enc_A, ksize = [1,2, 2, 1], strides = [1,1, 1,1], padding = 'VALID'), [int(batch_size/3),-1]),enc_A_W)+enc_A_b)
            enc_B =tf.sigmoid(tf.matmul(tf.reshape(tf.nn.avg_pool(value = enc_B, ksize = [1,2, 2, 1], strides = [1, 1, 1,1], padding = 'VALID'), [int(batch_size/3),-1]),enc_B_W)+enc_B_b)
            enc_B1 =tf.sigmoid(tf.matmul(tf.reshape(tf.nn.avg_pool(value = enc_B1, ksize = [1,2, 2, 1], strides = [1,1, 1, 1], padding = 'VALID'), [int(batch_size/3),-1]),enc_B_W)+enc_B_b)

            ii_sim_ij_v = tf.reduce_sum(enc_A*enc_B,1)
            ii_sim_ik_v = tf.reduce_sum(enc_A*enc_B1,1)
            ii_sim_ij_c = tf.reduce_sum(text_output1*text_output2,1)
            ii_sim_ik_c = tf.reduce_sum(text_output1*text_output3,1)

            mij =0.1*it_sim_ij + lamda*ii_sim_ij_v + (1-lamda)*ii_sim_ij_c
            mik =0.1*it_sim_ik + lamda*ii_sim_ik_v + (1-lamda)*ii_sim_ik_c
            score=tf.subtract(mij,mik)

            # Op for calculating the loss
            with tf.name_scope("cross_ent"):
                g_loss =  0.5 *tf.reduce_mean(tf.squared_difference(fake_rec_B,1))
                pixel_loss = tf.reduce_mean(tf.abs(fake_B-input_B))
                d_loss =0.5 *tf.reduce_mean(tf.square(fake_rec_B)) + 0.5 *tf.reduce_mean(tf.squared_difference(real_B,1))
                reg_loss = tf.reduce_mean(enc_A_W**2)+tf.reduce_mean(enc_B_W**2)*2\
                           +tf.reduce_mean(enc_A_b**2)+tf.reduce_mean(enc_B_b**2)*2\
                           +tf.reduce_mean(text_W1**2)+tf.reduce_mean(text_W2**2)*2\
                           +tf.reduce_mean(text_b1**2)+tf.reduce_mean(text_b2**2)*2
                bpr_loss = -tf.reduce_mean(tf.sigmoid(score))

            # Train op
            with tf.name_scope("train"):
                d_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5) #判别器训练器
                #d_optim = tf.train.GradientDescentOptimizer(learning_rate)
                g_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5) #生成器训练器

                model_vars = tf.trainable_variables()
                g_A_vars = [var for var in model_vars if 'generator' in var.name]
                d_A_vars = [var for var in model_vars if 'discriminator' in var.name]

                d_grads_and_vars = d_optim.compute_gradients(0.01*d_loss, var_list=d_A_vars) #计算判别器参数梯度
                d_train = d_optim.apply_gradients(d_grads_and_vars) #更新判别器参数
                g_grads_and_vars = g_optim.compute_gradients(g_loss+0.1*reg_loss+10000*pixel_loss+bpr_loss, var_list=g_A_vars) #计算生成器参数梯度
                g_train = g_optim.apply_gradients(g_grads_and_vars) #更新生成器参数

                train_op = tf.group(g_train,d_train)

            # Add gradients to summary
            g_A_loss_summ = tf.summary.scalar("g_loss", g_loss)
            d_A_loss_summ = tf.summary.scalar("d_loss", d_loss)
            bpr_loss_summ = tf.summary.scalar("bpr_loss", bpr_loss)

            # Evaluation op: Accuracy of the model
            with tf.name_scope("accuracy"):
                res=tf.expand_dims(score,1)
                val = tf.zeros((int(batch_size/3), 1), tf.float32)
                res=tf.concat([res,val],1)
                res1=tf.argmax(res, 1)
                res2= tf.zeros((int(batch_size/3), 1), tf.int64)
                correct_pred = tf.equal(res1, res2)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Add the accuracy to the summary
            tf.summary.scalar('train_accuracy', accuracy)

            # Merge all summaries together
            merged_summary = tf.summary.merge_all()

            # Initialize the FileWriter
            writer = tf.summary.FileWriter(filewriter_path)

            # Initialize an saver for store model checkpoints
            saver = tf.train.Saver()

            # Get the number of training/validation steps per epoch
            train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
            val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
            test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
            # Start Tensorflow session
            config = tf.ConfigProto(allow_soft_placement=True)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                # Add the model graph to TensorBoard
                writer.add_graph(sess.graph)
                # To continue training from one of your checkpoints
                # saver.restore(sess, "tmp2/tune_alexnet/checkpoints/model_epoch38.ckpt")
                # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                #                                                   filewriter_path))

                # Loop over number of epochs

                for epoch in range(num_epochs):
                    m_ij = []
                    m_ik = []
                    print("{} Epoch number: {}".format(datetime.now(), epoch+1))
                    file.write(str(epoch+1))
                    # Initialize iterator with the training dataset
                    sess.run(training_init_op)
                    sess.run(tra_text_i_init_op)
                    sess.run(tra_text_j_init_op)
                    sess.run(tra_text_k_init_op)
                    pixel_loss_sum = 0.0
                    cyc_loss_sum=0.0
                    g_loss_sum=0.0
                    d_loss_sum=0.0
                    reg_loss_sum=0.0
                    bpr_loss_sum=0.0
                    L_loss_sum = 0.0


                    for step in range(train_batches_per_epoch):
                        # print("step",step)
                        # get next batch of data
                        img_batch = sess.run(next_batch)
                        text_batch_i = sess.run(next_batchi)
                        text_batch_j = sess.run(next_batchj)
                        text_batch_k = sess.run(next_batchk)
                        # And run the training op
                        [ms,train_op_,each_pixel_loss,each_g_loss,each_reg_loss,each_d_loss,each_bpr_loss]= sess.run([merged_summary,train_op,pixel_loss,g_loss,reg_loss,d_loss,bpr_loss],feed_dict={x: img_batch,
                                                                                                                                                                                                     text_x_i: text_batch_i,
                                                                                                                                                                                                     text_x_j: text_batch_j,
                                                                                                                                                                                                     text_x_k: text_batch_k,
                                                                                                                                                                                                     keep_prob: dropout_rate})
                        writer.add_summary(ms)
                        pixel_loss_sum += each_pixel_loss
                        g_loss_sum += each_g_loss
                        reg_loss_sum += each_reg_loss
                        d_loss_sum += each_d_loss
                        bpr_loss_sum += each_bpr_loss
                        L_loss_sum += each_pixel_loss + each_g_loss + each_reg_loss + each_d_loss + each_bpr_loss


                    print("pixel_loss_sum:{} g_loss_sum:{} reg_loss_sum:{} d_loss_sum:{} bpr_loss_sum:{} L_loss_sum:{}".format(pixel_loss_sum,g_loss_sum,reg_loss_sum,d_loss_sum,bpr_loss_sum,L_loss_sum))
                    file.write("pixel_loss_sum:{} g_loss_sum:{} reg_loss_sum:{} d_loss_sum:{} bpr_loss_sum:{} L_loss_sum:{}\n".format(pixel_loss_sum,g_loss_sum,reg_loss_sum,d_loss_sum,bpr_loss_sum,L_loss_sum))
                    print("{} Start Train".format(datetime.now()))
                    sess.run(training_init_op)
                    sess.run(tra_text_i_init_op)
                    sess.run(tra_text_j_init_op)
                    sess.run(tra_text_k_init_op)
                    test_acc = 0.
                    test_count = 0


                    file.write("train begin time:{}".format(datetime.now()))
                    for train_epoch in range(train_batches_per_epoch):
                        img_batch = sess.run(next_batch)
                        text_batch_i = sess.run(next_batchi)
                        text_batch_j = sess.run(next_batchj)
                        text_batch_k = sess.run(next_batchk)

                        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                            text_x_i:text_batch_i,
                                                            text_x_j:text_batch_j,
                                                            text_x_k:text_batch_k,
                                                            keep_prob:1.})

                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count

                    print("{} Train Accuracy = {:.4f}".format(datetime.now(),test_acc))

                    file.write("train end time:{}".format(datetime.now()))
                    file.write("Train Accuracy = {:.4f}\n".format(
                        test_acc))

                    sess.run(validation_init_op)
                    sess.run(val_text_i_init_op)
                    sess.run(val_text_j_init_op)
                    sess.run(val_text_k_init_op)
                    val_acc = 0.
                    val_count = 0

                    file.write("valid begin time:{}".format(datetime.now()))

                    for valid_epoch in range(val_batches_per_epoch):

                        img_batch = sess.run(next_batch)
                        text_batch_i = sess.run(next_batchi)
                        text_batch_j = sess.run(next_batchj)
                        text_batch_k = sess.run(next_batchk)
                        acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                            text_x_i:text_batch_i,
                                                            text_x_j:text_batch_j,
                                                            text_x_k:text_batch_k,
                                                            keep_prob:1.})
                        val_acc += acc
                        val_count += 1
                    val_acc /= val_count
                    print("{}   Validation Accuracy = {:.4f}".format(datetime.now(),
                                                                     val_acc))
                    file.write("valid end time:{}".format(datetime.now()))
                    file.write("    Validation Accuracy = {:.4f}\n".format(
                                                                    val_acc))
                    # print("{} Saving checkpoint of model...".format(datetime.now()))

                    # save checkpoint of the model
                    checkpoint_name = os.path.join(checkpoint_path,
                                                   'model_epoch'+str(epoch+1)+'.ckpt')
                    save_path = saver.save(sess, checkpoint_name)
                    # print("{} Model checkpoint saved at {}".format(datetime.now(),
                    #                                                checkpoint_name))


                    sess.run(test_init_op)
                    sess.run(tes_text_i_init_op)
                    sess.run(tes_text_j_init_op)
                    sess.run(tes_text_k_init_op)
                    # if val_acc > best_validation_auc_score:
                    #     best_validation_auc_score = val_acc
                    test_acc = 0.
                    test_count = 0
                    count_test = 0
                    file.write("test begin time:{}".format(datetime.now()))
                    for test_epoch in range(test_batches_per_epoch):
                        img_batch= sess.run(next_batch)
                        text_batch_i = sess.run(next_batchi)
                        text_batch_j = sess.run(next_batchj)
                        text_batch_k = sess.run(next_batchk)
                        mij_,mik_,fake_B_value,acc = sess.run([mij,mik,fake_B,accuracy], feed_dict={x: img_batch,
                                                                                                text_x_i:text_batch_i,
                                                                                                text_x_j:text_batch_j,
                                                                                                text_x_k:text_batch_k,
                                                                                                keep_prob:1.})
                        for k in range(int(batch_size/3)):
                            imsave("./output/fakeB_2loss0.1/fakeB_"+ str(count_test)+".jpg",((fake_B_value[k]+1)*127.5).astype(np.float32))
                            count_test += 1
                        # m_ij.append(mij_)
                        # m_ik.append(mik_)
                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count
                    # np.savetxt("/home/liujinhuan/neurocomputing2019/output/mij_" + str(epoch) +".csv", np.reshape(m_ij, (-1, 1)), fmt="%f")
                    # np.savetxt("/home/liujinhuan/neurocomputing2019/output/mik_" + str(epoch) +".csv", np.reshape(m_ik, (-1, 1)), fmt="%f")
                    print("{}               Test Accuracy = {:.4f}".format(datetime.now(),
                                                                           test_acc))
                    file.write("                Test Accuracy = {:.4f}\n".format(
                        test_acc))
                    file.write("valid end time:{}".format(datetime.now()))
                    file.flush()


file.write("end time: {}".format(datetime.now()))
file.close()