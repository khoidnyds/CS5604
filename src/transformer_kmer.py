# Much of the implementation came from the Tensorflow Transformer tutorial:
# https://www.tensorflow.org/text/tutorials/transformer

# ########################################################################
# # Setup
# ########################################################################
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from pyfaidx import Fasta
from utils import build_kmers, read_fasta_to_dict, build_kmer_token_list

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# ########################################################################
# # Dataset
# ########################################################################

# Sequence Settings
GENOME_CUTOFF_SIZE = 16700 # 16700
ksize = 5
start_token = "<S>"
end_token = "<E>"
# Hyperparameters
num_layers = 4 # 6
d_model = 100 # 128
dff = 256 # 512
num_heads = 4 # 8
dropout_rate = 0.1

# Create kmer tokenizer table
kmer_tokens = build_kmer_token_list("data/dataset_1000.fasta", ksize)
kmer_tokens = sorted(kmer_tokens)
kmer_tokens.insert(0, start_token)
kmer_tokens.append(end_token)
print(kmer_tokens)
kmer_token_dict = { kmer : kmer_tokens.index(kmer) + 1 for kmer in kmer_tokens }
# kmer_token_dict[start_token] = 1    # Set index of 1 for start token
# kmer_token_dict[end_token] = 
# sorted_kmer_list = sorted(list(kmer_token_dict.items()), key=lambda x: (x[1], x[0]))
# print(sorted_kmer_list)
kmer_lookup = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
        list(kmer_token_dict.keys()),
        list(kmer_token_dict.values()),
        value_dtype=tf.int64
    ),
    num_oov_buckets=1
)
reverse_kmer_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        list(kmer_token_dict.values()),
        list(kmer_token_dict.keys()),
        key_dtype=tf.int64,
        value_dtype=tf.string,
    ),
    default_value="UNDEFINED ID"
)


# ########################################################################
# # Text tokenization & detokenization
# ########################################################################

# Take in a genome file (A,C,G,T)s
# Create numerical representation of each 11 length k-mer in the sequence
def tokenize_sequence(seq_tensor):
    return kmer_lookup.lookup(seq_tensor)

def detokenize_sequence(tok_seq_tensor):
    return reverse_kmer_lookup.lookup(tok_seq_tensor)

def tokenize_pairs(inputseq, targetseq):
    i_tokens = tokenize_sequence(inputseq)
    t_tokens = tokenize_sequence(targetseq)
    return i_tokens, t_tokens


# Input data should be a kmer, target is the subsequent kmer
def dataset_from_fasta(fasta_filepath, train=0.7, val=0.2, test=0.1):
    """
    Create a tensorflow dataset from .fasta file

    Each gene sequence is split into a series of kmers
    Every kmer up to (# kmers - 1) is used as input data
    Every kmer past the 1st is used as the target kmer for the previous kmer in the seqence
    """
    assert math.isclose(train+val+test, 1)
    # Each fasta file should become its own "sentence"
    gene_seqs = read_fasta_to_dict(fasta_filepath)

    nt = int(train*len(gene_seqs.keys()))
    nv = int((train+val)*len(gene_seqs.keys()))
    train_labels = list(gene_seqs.keys())[:nt]
    val_labels = list(gene_seqs.keys())[nt:nv]
    test_labels = list(gene_seqs.keys())[nv:]
    # print(train_labels, "\n\n\n")
    # print(val_labels, "\n\n\n")
    # print(test_labels, "\n\n\n")

    def create_dataset_subset(seq_labels):
        i_kmers = []
        t_kmers = []
        for seq_label in seq_labels:
            seq = gene_seqs[seq_label][:GENOME_CUTOFF_SIZE]
            kmers = build_kmers(seq, ksize)
            i_kmers.append([start_token])
            kmers.insert(0, start_token)
            kmers.append(end_token)
            t_kmers.append(kmers)
        print(len(i_kmers))
        print(len(t_kmers))
        i_ragged = tf.constant(i_kmers) # .to_tensor(default_value='', shape=[None, 17000])
        t_ragged = tf.ragged.constant(t_kmers).to_tensor() # .to_tensor(default_value='', shape=[None, 17000])
        # i_ragged = tf.ragged.map_flat_values(tokenize_kmer, i_ragged).to_tensor(default_value=tokenize_kmer(""), shape=[None, 17000])
        # t_ragged = tf.ragged.map_flat_values(tokenize_kmer, t_ragged).to_tensor(default_value=tokenize_kmer(""), shape=[None, 17000])
        # print(i_ragged)
        # print(t_ragged)
        return tf.data.Dataset.from_tensor_slices((i_ragged, t_ragged))
    
    trainset = create_dataset_subset(train_labels)
    valset = create_dataset_subset(val_labels)
    testset = create_dataset_subset(test_labels)
    return trainset, valset, testset

    # return tf.data.Dataset.from_tensor_slices((i_kmers, t_kmers))


def make_batches(ds):
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE))
        


BUFFER_SIZE = 20000
BATCH_SIZE = 16

def create_data_batches():
    logging.info("Beginning data batch creation")
    print("making dataset")
    trainset, valset, testset = dataset_from_fasta("data/dataset_1000.fasta")
    print("making training batches")
    train_batches = make_batches(trainset)
    print("Made training batches: ", train_batches)
    val_batches = make_batches(valset)
    print("Made validation batches: ", val_batches)
    test_batches = make_batches(testset)
    print("Made testing batches: ", test_batches)
    tf.data.experimental.save(train_batches, "data/full/training_batches")
    tf.data.experimental.save(val_batches, "data/full/validation_batches")
    tf.data.experimental.save(test_batches, "data/full/test_batches")
    logging.info("Done saving data batches to disk.")

    return train_batches, val_batches, test_batches


def load_data_batches():
    logging.info("Loading data batches from disk")
    train_batches = tf.data.experimental.load("data/full/training_batches")
    val_batches = tf.data.experimental.load("data/full/validation_batches")
    test_batches = tf.data.experimental.load("data/full/test_batches")
    return train_batches, val_batches, test_batches


# print("Creating dataset from 1000 FASTA file")
# trainset, valset, testset = dataset_from_fasta("data/dataset_1000.fasta")
# print("\nDone creating dataset from 1000 FASTA file")
# train_batches = make_batches(trainset)
# for inp, tar in train_batches.take(1):
#     tf.print(inp, summarize=10)
#     tf.print(tar, summarize=10)


# create_data_batches()


train_batches, val_batches, test_batches = load_data_batches()
for inp, tar in train_batches.take(1):
    tf.print(inp, summarize=10)
    tf.print(tar, summarize=10)
    print(detokenize_sequence(tar[0]))
    print(detokenize_sequence(tf.cast(tf.constant(2564), dtype=tf.int64)))
    print(tokenize_sequence(tf.constant(end_token)))


# ########################################################################
# # Positional encoding
# ########################################################################


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ########################################################################
# # Masking
# ########################################################################


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ########################################################################
# # Scaled dot product attention
# ########################################################################


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# ########################################################################
# # Multi-head attention
# ########################################################################


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # query weights
        self.wk = tf.keras.layers.Dense(d_model)  # key weights
        self.wv = tf.keras.layers.Dense(d_model)  # value weights

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # does the split
        return tf.transpose(x, perm=[0, 2, 1, 3])                       # does the transpose

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


# ########################################################################
# # Point wise feed forward network
# ########################################################################


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),  # dff == # hidden layer nodes
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# ########################################################################
# # Encoder layer
# ########################################################################


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Send input through multi-head attention layer
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        # Send that through the FFN
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# ########################################################################
# # Decoder layer
# ########################################################################


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# ########################################################################
# # Encoder
# ########################################################################


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 pe_input, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(pe_input, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# ########################################################################
# # Decoder
# ########################################################################


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 pe_target, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            pe_target, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# ########################################################################
# # Create the Transformer
# ########################################################################


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, is_training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp, tar)

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, is_training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, is_training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


# ########################################################################
# # Optimizer
# ########################################################################
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# ########################################################################
# # Loss and metrics
# ########################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# ########################################################################
# # Training and checkpointing
# ########################################################################
def build_transformer():
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=len(list(kmer_token_dict.keys())) + 1,
        target_vocab_size=len(list(kmer_token_dict.keys())) + 1,
        pe_input=GENOME_CUTOFF_SIZE,
        pe_target=GENOME_CUTOFF_SIZE,
        rate=dropout_rate)

    checkpoint_path = "./checkpoints/full"

    ckpt = tf.train.Checkpoint(transformer=transformer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    
    return transformer


# transformer = Transformer(
#     num_layers=num_layers,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=len(list(kmer_token_dict.keys())) + 1,
#     target_vocab_size=len(list(kmer_token_dict.keys())) + 1,
#     pe_input=GENOME_CUTOFF_SIZE,
#     pe_target=GENOME_CUTOFF_SIZE,
#     rate=dropout_rate)

# checkpoint_path = "./checkpoints/full"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')


# # The @tf.function trace-compiles train_step into a TF graph for faster
# # execution. The function specializes to the precise shape of the argument
# # tensors. To avoid re-tracing due to the variable sequence lengths or variable
# # batch sizes (the last batch is smaller), use input_signature to specify
# # more generic shapes.

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    tf.print("Running train step")
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     is_training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


# BUFFER_SIZE = 20000
# BATCH_SIZE = 64

# train_examples, val_examples = examples['train'], examples['validation']
# train_batches = make_batches(train_examples)
# val_batches = make_batches(val_examples)
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# tf.print("HERE")
# EPOCHS = 5
# for epoch in range(EPOCHS):
#     start = time.time()

#     train_loss.reset_states()
#     train_accuracy.reset_states()

#     # inp -> portuguese, tar -> english
#     for (batch, (inp, tar)) in enumerate(train_batches):
#         train_step(inp, tar)

#         if batch % 50 == 0:
#             tf.print(
#                 f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     if (epoch + 1) % 1 == 0:
#         ckpt_save_path = ckpt_manager.save()
#         tf.print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

#     tf.print(
#         f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

#     tf.print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')




# output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
# output_array = output_array.write(0, tokenize_sequence(tf.constant([start_token])))
# encoder_input = tokenize_sequence(tf.constant([start_token]))[tf.newaxis]
# seq_len = tf.size(encoder_input)
# for i in tf.range(10):
#     # print(seq_len)
#     # print(GENOME_CUTOFF_LEN - seq_len.numpy())
#     output = tf.transpose(output_array.stack())
#     # paddings = tf.constant([[0,0], [0, GENOME_CUTOFF_SIZE - tf.size(encoder_input).numpy()]])
#     # padded_input = tf.pad(encoder_input, paddings, mode='CONSTANT')
#     print("encoder input: ", encoder_input)
#     # print("padded input: ", padded_input)
#     # predictions, _ = transformer([encoder_input, output], is_training=False)
#     predictions, _ = transformer([encoder_input, output], is_training=False)

#     # select the last token from the seq_len dimension
#     predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

#     print("predictions: ", predictions)
#     print("prediction probabilities: ", tf.nn.softmax(predictions))
#     print("max probability: ", tf.reduce_max(tf.nn.softmax(predictions)))

#     predicted_id = tf.argmax(predictions, axis=-1)

#     # concatentate the predicted_id to the output which is given to the decoder
#     # as its input.
#     print("predicted id: ", predicted_id)
#     seq_len += 1
#     # encoder_input = tf.stack([tf.cast(encoder_input, tf.int32), tf.cast(predicted_id, tf.int32)])
#     # encoder_input = tf.reshape(tf.concat([tf.cast(encoder_input, tf.int32), tf.cast(predicted_id, tf.int32)], 1), [1, seq_len])
#     output_array = output_array.write(i+1, predicted_id[0])
#     print(output_array)

#     # if predicted_id == 6:
#     #     break

# output = tf.transpose(output_array.stack())
# print(output)
# print(detokenize_sequence(output))
