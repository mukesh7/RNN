import numpy as np
import pandas as pd
import tensorflow as tf
import argparse as ap
import matplotlib.pyplot as plt
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import helpers
from tensorflow.python.layers import core as layers_core
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument("--lr",help="initial learning rate")
parser.add_argument("--batch_size",help="training batch size (1 and multiples of 5)")
parser.add_argument("--init",help="initialization method")
parser.add_argument("--save_dir",help="saving dir for weights")
parser.add_argument("--dropout_prob",help="saving dir for weights")
#parser.add_argument("--decode_method",help="decoder_method")
#parser.add_argument("--beam_width",help="beam_width")
args=parser.parse_args()

LEARNING_RATE = float(args.lr)
DROP   = float(args.dropout_prob)
dr_temp = float(args.dropout_prob)

def weight_xavier_init(n_inputs, n_outputs, init):
    if(init == 1):
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



vout = open("train/summaries.txt", "r")
vout = vout.read()
vout = vout.replace('\n',' ')
vout = vout.split(' ')
len_out = np.unique(np.asarray(vout))
#len_out = np.append(len_out, 'PAD')
#len_out = np.append(len_out, 'EOS')

vin = open("train/train.combined", "r")
vin = vin.read()
vin = vin.replace('\n',' ')
vin = vin.split(' ')
len_in = np.unique(np.asarray(vin))
temp = np.append(len_in, len_out)
temp = np.unique(temp)

dct = {}
rdct = {}
for i in range(len(temp)):
    dct[temp[i]] = i+2
    rdct[i+2] = temp[i]
dct['PAD'] = 0
dct['EOS'] = 1
rdct[0] = 'PAD'
rdct[1] = 'EOS'
len_in = np.append(len_in, 'PAD')
len_in = np.append(len_in, 'EOS')

vin = open("train/train.combined", "r")
vin = vin.read()
vin = vin.split('\n')
for i in range(len(vin)):
    temp = vin[i].split(' ')
    for j in range(len(temp)):
        temp[j] = dct[temp[j]]
    vin[i] = temp


vout = open("train/summaries.txt", "r")
vout = vout.read()
vout = vout.split('\n')
for i in range(len(vout)):
    temp1=[]
    temp = vout[i].split(' ')
    for j in range(len(temp)-1):
        temp1.append(dct[temp[j]])
    vout[i] = temp1


valin = open("dev/dev.combined", "r")
valin = valin.read()
valin = valin.split('\n')
for i in range(len(valin)):
    temp = valin[i].split(' ')
    for j in range(len(temp)):
        temp[j] = dct[temp[j]]
    valin[i] = temp

valout = open("dev/summaries.txt", "r")
valout = valout.read()
valout = valout.split('\n')
for i in range(len(valout)):
    temp1=[]
    temp = valout[i].split(' ')
    for j in range(len(temp)-1):
        temp1.append(dct[temp[j]])
    valout[i] = temp1

testin = open("test/test.combined", "r")
testin = testin.read()
testin = testin.split('\n')
for i in range(len(testin)):
    temp = testin[i].split(' ')
    for j in range(len(temp)):
        temp[j] = dct[temp[j]]
    testin[i] = temp





tf.reset_default_graph()
sess = tf.InteractiveSession()
PAD = 0
EOS = 1

vocab_size = len(dct)
input_embedding_size = 256

encoder_hidden_units = 256
decoder_hidden_units = encoder_hidden_units * 2

#####Encoder
encoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None,None), dtype=tf.int32, name='decoder_targets')
decoder_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_lengths')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_cell = LSTMCell(encoder_hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )

#encoder_fw_outputs
#encoder_fw_final_state
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)   
#index_in_epoch=0
#def attention(encoder_outputs, attention_size):
#hidden_size = inputs.shape[2].value
#if index_in_epoch==0:
#    abcd= 256
#else:
#    abcd= encoder_inputs_length.get_shape().as_list()[0]
#
#w_omega = tf.get_variable("w_omega", shape=[decoder_hidden_units,input_embedding_size], initializer=weight_xavier_init(decoder_hidden_units,input_embedding_size, args.init))
#b_omega = bias_variable([input_embedding_size])
#u_omega = bias_variable([input_embedding_size])


#w_omega = tf.Variable(tf.random_normal([512, 256], stddev=0.1),dtype=tf.float32)
#b_omega = tf.Variable(tf.random_normal([256], stddev=0.1),dtype=tf.float32)
#u_omega = tf.Variable(tf.random_normal([256], stddev=0.1),dtype=tf.float32)

########## DROPOUT ######################
#encoder_outputs = tf.nn.dropout(encoder_outputs,DROP)
############################################
#inputs = tf.transpose(encoder_outputs, [1,0,2])

    
#alpha_list=[]

decoder_cell = LSTMCell(decoder_hidden_units)
#attention_mechanism = tf.contrib.seq2seq.LuongAttention(512, encoder_outputs)
#attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_size=256)



encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))


#W = tf.Variable(tf.random_uniform([decoder_hidden_units,vocab_size], -1, 1), dtype=tf.float32)
#b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
W = tf.get_variable("W", shape=[decoder_hidden_units,vocab_size], initializer=weight_xavier_init(decoder_hidden_units,vocab_size, args.init))
b = bias_variable([vocab_size])



#Watt = tf.Variable(tf.random_uniform([vocab_size,vocab_size], -1, 1), dtype=tf.float32)
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
#ab=attention(encoder_outputs,50)
#ab = tf.cast(ab, tf.int32)
def loop_fn_initial():

    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
#    zer= tf.zeros(shape=(512,512,100),dtype=tf.float32)
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
#    po=tf.cast(previous_output,tf.float32)
    
    def get_next_input():
        ###########DROP OUT################
#        prev = tf.nn.dropout(prev,DROP)
#        output_logits = tf.add(tf.matmul(prev_drop, W), b)
        
        output_logits = tf.add(tf.matmul(previous_output, W), b)  #att problem
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        ############ DROP OUT##################
#        next_input = tf.nn.embedding_lookup(embeddings, prediction)
#        next_input = tf.nn.dropout(next_input,DROP)
#        
#        print(next_input.get_shape())
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended
                                                  
    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)
    
    
def loop_fn(time, previous_output, previous_state, previous_loop_state):
#    po = tf.Variable(previous_output,dtype=tf.float32)
#    previous_output=po
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
#        po1 = tf.concat((tt,previous_output),0)
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
#batch_size = 100




decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

#att= alpha_list
decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)
index_in_epoch =0
index_in_epoch2 = 0
epochs_completed =0 
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

sess.run(tf.global_variables_initializer())
loss_track = []



#def batch(inputs, max_sequence_length=None):
#    """
#    Args:
#        inputs:
#            list of sentences (integer lists)
#        max_sequence_length:
#            integer specifying how large should `max_time` dimension be.
#            If None, maximum sequence length would be used
#    
#    Outputs:
#        inputs_time_major:
#            input sentences transformed into time-major matrix 
#            (shape [max_time, batch_size]) padded with 0s
#        sequence_lengths:
#            batch-sized list of integers specifying amount of active 
#            time steps in each input sequence
#    """
#    
#    sequence_lengths = [len(seq) for seq in inputs]
#    batch_size = len(inputs)
#    
#    if max_sequence_length is None:
#        max_sequence_length = max(sequence_lengths)
#    
#    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
#    
#    for i, seq in enumerate(inputs):
#        for j, element in enumerate(seq):
#            inputs_batch_major[i, j] = element
#
#    # [batch_size, max_time] -> [max_time, batch_size]
#    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
#
#    return inputs_time_major, sequence_lengths

def next_b(lst, s, e):
    return lst[s:e]

def next_feed(batch_size):
    global vin
    global vout
    global index_in_epoch
    global epochs_completed
#    num_examples = 25000
    start = index_in_epoch
    index_in_epoch += batch_size
    end = index_in_epoch
    batch = next_b(vin,start,end)
    batch_out = next_b(vout,start,end)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, m = helpers.batch(
        [(sequence) + [EOS] for sequence in batch_out]
    )
    
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_lengths:m
    }


def next_feedval(batch_size):
    global index_in_epoch2
    start = index_in_epoch2
    index_in_epoch2 += batch_size
    end = index_in_epoch2

    
    batch = next_b(valin,start,end)
    batch_out = next_b(valout,start,end)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, m = helpers.batch(
        [(sequence) + [EOS] + [PAD]*150 for sequence in batch_out]
    )
    
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_lengths:m
    }

def next_feedtest(batch_size):
    global index_in_epoch
    start = index_in_epoch
    index_in_epoch += batch_size
    end = index_in_epoch
    
    batch = next_b(testin,start,end)
    batch_out = next_b(testin,start,end)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, m = helpers.batch(
        [[PAD]*250 for sequence in batch_out]
    )
    
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_lengths:m
    }



#LEARNING_RATE = float(args.lr) #0.00015
EPOCHS = 12
#DROPOUT_PROB = 0.8
patience = 5
#DROPOUT_HIDDEN = 0.6
BATCH_SIZE = int(args.batch_size)



batch_size  = BATCH_SIZE
loss_val = 0 
v_loss=[]
t_loss=[]
old_loss=3232323
for epoch in range(EPOCHS):    
    loss_epoch=0
    loss_val = 0
    DROP = dr_temp
    for batch in range(int(len(vin)/batch_size)):
            fd = next_feed(batch_size)
            _,l = sess.run([train_op,loss], fd)
            loss_epoch+=l
#            loss_track.append(l)
#            print("epoch {} and loss {}".format())
            if(batch%10==0):
                print(batch)
            #print('batch {}'.format(batch))
            
    print('epoch: {}, loss: {}'.format(epoch,loss_epoch))
    t_loss.append(loss_epoch)
    DROP = 1
    for batch1 in range(int(len(valin)/batch_size)):
            fd = next_feedval(batch_size)
            l = sess.run([loss], fd)
            loss_val+=l[0]
#            loss_track.append(l)
#            print("epoch {} and loss {}".format())
    new_loss = loss_val
    v_loss.append(loss_val)
    old_loss = loss_val
    
    ############### EARLY STOPPING ###################
    if(old_loss > new_loss):
        patience = patience-1
                #LEARNING_RATE *= 0.5
    else:
        patience = 5
        saver = tf.train.Saver()
        saver.save(sess, args.save_dir+'/model.ckpt')
#        saver.save(sess, 'train/model.ckpt')      #AWS
        
#            if(val_loss_new[0] < 900):
#                print("DONE")
#                break
    if(patience == 0):
        break
    
    old_loss= new_loss
    #################### ANNEAL #####################
 
#    old_loss = loss_val
#    v_loss.append(loss_val)
    
    
    
    
    print('epoch: {}, val loss:  {},lr: {}'.format(epoch,loss_val,LEARNING_RATE))
    
    
    index_in_epoch=0
    index_in_epoch2=0   
np.savetxt("plot.csv", np.column_stack((t_loss, v_loss)), delimiter=",")

#saver = tf.train.Saver()
#saver.restore(sess, 'model.ckpt')

#fd = next_feedval(1)
#l = sess.run([loss], fd)
#loss_track.append(l)
#print(l)
##print('batch {}'.format(batch))
#print('  minibatch loss: {}'.format(sess.run(loss, fd)))
#predict_val = sess.run(decoder_prediction, fd)
#for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
##        print('  sample {}:'.format(i + 1))
##           print('    input     > {}'.format(inp))
#print('    predicted > {}'.format(pred))
#if i >= 2:
#break    
    
    
        
#predict_ = sess.run(decoder_prediction, feed(0))
#            print('batch {}'.format(batch))
#            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            
#####Attention
#attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
#attention_mechanism = tf.contrib.seq2seq.LuongAttention(
#    encoder_hidden_units, attention_states,
#    memory_sequence_length=encoder_inputs_length)
#
#decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
#    decoder_cell, attention_mechanism,
#    attention_layer_size=encoder_hidden_units)

######Helper
#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
#    embedding_decoder,
#    tf.fill([batch_size], tgt_sos_id), tgt_eos_id)
#
#decoder = tf.contrib.seq2seq.BasicDecoder(
#    decoder_cell, helper, encoder_final_state,
#    output_layer=projection_layer)

