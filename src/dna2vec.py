from pyfaidx import Fasta
from utils import build_kmers
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
import io
import logging
from pathlib import Path

BATCH_SIZE = 1024
BUFFER_SIZE = 10000


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns+1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


class DNA2Vec():
    def __init__(self, input_path, model_path):
        self.out_path = model_path.joinpath("dna2vec")
        Path.mkdir(self.out_path, parents=True, exist_ok=True)

        self.data = Fasta(str(input_path))
        self.kmer = 13
        self.window_size = 8
        self.num_ns = 9
        self.embedding_dim = 300
        self.num_epochs = 2
        self.val_split = 0.2

        logging.info(
            f"Initializing embedding space with word2vec model, kmer:{self.kmer}, window size:{self.window_size}, negative sample:{self.num_ns}, embedding dimension:{self.embedding_dim}")

        logging.info("Transform sequence to k-mer words")
        self.corpus = pd.DataFrame([build_kmers(x[:].seq, self.kmer)
                                    for x in tqdm(self.data, total=len(self.data.keys()))]).fillna(np.NaN)

    def build(self):
        """
        pipeline function
        """
        self.build_vocab()
        self.build_dataset()

        logging.info("Training model")
        word2vec = Word2Vec(self.vocab_size, self.embedding_dim, self.num_ns)
        word2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy', 'mse'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.out_path.joinpath("tensorboard"), histogram_freq=1)

        word2vec.fit(self.train_data,
                     epochs=self.num_epochs,
                     verbose=True,
                     callbacks=[tensorboard_callback, early_stop],
                     validation_data=self.val_data)
        weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

        out_v = io.open(self.out_path.joinpath(
            'vectors.tsv'), 'w', encoding='utf-8')
        out_m = io.open(self.out_path.joinpath(
            'metadata.tsv'), 'w', encoding='utf-8')
        logging.info(
            f"Writing embedding space to {self.out_path}| {out_v.name} {out_m.name}")

        for index, word in enumerate(self.vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            if word is np.NaN:
                continue
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")
        out_v.close()
        out_m.close()

    def build_vocab(self):
        """
        Build vocabulary from the input
        """
        logging.info("Building vocabulary")

        vocab, index = {}, 1  # start indexing from 1
        vocab['<pad>'] = 0  # add a padding token
        for _, sentence in self.corpus.iterrows():
            for token in sentence.values:
                if token not in vocab:
                    vocab[token] = index
                    index += 1

        self.vocab = vocab
        self.vocab_size = len(vocab)

    def build_dataset(self):
        """
        Build dataset from the input
        """
        logging.info("Building dataset")

        self.sequences = [[self.vocab[word] for word in self.corpus.iloc[sent, :]]
                          for sent in range(self.corpus.shape[0])][:500]

        targets, contexts, labels = self.generate_training_data()

        targets = np.array(targets)
        contexts = np.array(contexts)[:, :, 0]
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices(
            ((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE)

        val_size = int(self.val_split*labels.shape[0])
        train_size = labels.shape[0]-val_size
        val_data = dataset.take(val_size)
        train_data = dataset.skip(train_size)

        self.train_data = train_data.batch(
            BATCH_SIZE, drop_remainder=True).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_data = val_data.batch(
            BATCH_SIZE, drop_remainder=True).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def generate_training_data(self):
        """
        Generates skip-gram pairs with negative sampling for a list of sequences
        (int-encoded sentences) based on window size, number of negative samples
        and vocabulary size.

        Args:
            sequences ([type]): [description]
            window_size ([type]): [description]
            num_ns ([type]): [description]
            vocab_size ([type]): [description]
            seed ([type]): [description]

        Returns:
            [type]: [description]
        """

        logging.info("Generating training data set")

        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []

        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
            self.vocab_size)

        # Iterate over all sequences (sentences) in dataset.
        for sequence in tqdm(self.sequences):

            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=self.vocab_size,
                sampling_table=sampling_table,
                window_size=self.window_size,
                negative_samples=0)

            # Iterate over each positive skip-gram pair to produce training examples
            # with positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=self.num_ns,
                    unique=True,
                    range_max=self.vocab_size,
                    name="negative_sampling")

                # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                    negative_sampling_candidates, 1)

                context = tf.concat(
                    [context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0]*self.num_ns, dtype="int64")

                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels
