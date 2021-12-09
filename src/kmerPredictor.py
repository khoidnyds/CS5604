
import logging

import tensorflow as tf

from transformer_kmer import ksize, start_token, end_token
from utils import build_kmer_token_list

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings



class KmerPredictor():
    def __init__(self, transformer):
        # if not isinstance(transformer, Transformer):
        #     raise TypeError("transformer must be a Transformer object")
        self.transformer = transformer
        self.predictions = None
        self.encoder_input = None
        self.output_array = None
        self.output = None
        self.current_kmer_num = None
        self.is_output_updated = True

        self.kmer_token_dict = None
        self.kmer_lookup = None
        self.reverse_kmer_lookup = None
        self._initialize_kmer_tables()


    def _initialize_kmer_tables(self):
        # Create kmer tokenizer table
        kmer_tokens = build_kmer_token_list("data/dataset_1000.fasta", ksize)
        kmer_tokens = sorted(kmer_tokens)
        kmer_tokens.insert(0, start_token)
        kmer_tokens.append(end_token)
        self.kmer_token_dict = { kmer : kmer_tokens.index(kmer) + 1 for kmer in kmer_tokens }
        # self.kmer_token_dict[start_token] = 1    # Set index of 1 for start token
        # sorted_kmer_list = sorted(list(kmer_token_dict.items()), key=lambda x: (x[1], x[0]))
        # print(sorted_kmer_list)
        self.kmer_lookup = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(self.kmer_token_dict.keys()),
                list(self.kmer_token_dict.values()),
                value_dtype=tf.int64
            ),
            num_oov_buckets=1
        )
        self.reverse_kmer_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(self.kmer_token_dict.values()),
                list(self.kmer_token_dict.keys()),
                key_dtype=tf.int64,
                value_dtype=tf.string,
            ),
            default_value="UNDEFINED ID"
        )

    def debug_get_highest_prob_kmer(self, kmer_probs):
        predicted_kmer_token = tf.argmax(kmer_probs, axis=-1)
        predicted_kmer = self.detokenize_sequence(predicted_kmer_token)[0][0].numpy().decode('UTF-8')
        return predicted_kmer

    def debug_get_n_highest_prob_kmers(self, kmer_probs, n):
        assert n >= 1
        results = tf.math.top_k(kmer_probs, k=n)
        return self.detokenize_sequence(tf.cast(results.indices, dtype=tf.int64))


    def tokenize_sequence(self, seq_tensor):
        return self.kmer_lookup.lookup(seq_tensor)

    def detokenize_sequence(self, tok_seq_tensor):
        return self.reverse_kmer_lookup.lookup(tok_seq_tensor)

    def start_kmer_prediction(self):
        self.output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        self.output_array = self.output_array.write(0, self.tokenize_sequence(tf.constant([start_token])))
        self.current_kmer_num = 1 # Accounts for the initial starting token
        self.encoder_input = self.tokenize_sequence(tf.constant([start_token]))[tf.newaxis]
        

    def get_next_kmer_probabilities(self):
        # Only do prediction if the output has been updated since previous prediction
        if self.is_output_updated:
            # Have transformer generate kmer probabilities based on current output
            self.output = tf.transpose(self.output_array.stack())
            predictions, _ = self.transformer([self.encoder_input, self.output], is_training=False)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            probabilities = tf.nn.softmax(predictions)

            self.is_output_updated = False
            return probabilities
            # return self.predictions_to_dict(predictions)
        else:
            print("Cannot generate next kmer without updating outputs")
            return None

        
    def predictions_to_dict(self, predictions):
        # Returns a dict of Kmer(str tensor) -> probability
        prediction_dict = {}
        for idx in range(tf.size(predictions)):
            prediction_dict[self.reverse_kmer_lookup.lookup(idx)] = predictions[idx]
        return prediction_dict


    def feedback_next_kmer(self, next_kmer):
        # Update the transformer output with actual next kmer
        if self.is_output_updated:
            print("Warning, updating kmer outputs before predicting next kmer")
        # kmer_tensor = tf.squeeze(self.kmer_lookup.lookup(next_kmer), axis=[0])
        kmer_tensor = self.kmer_lookup.lookup(tf.constant(next_kmer))[tf.newaxis]
        self.output_array = self.output_array.write(self.current_kmer_num, kmer_tensor)
        self.current_kmer_num += 1
        self.is_output_updated = True



# learning_rate = CustomSchedule(100)

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)

# transformer = Transformer(
#     num_layers=4,
#     d_model=100,
#     num_heads=4,
#     dff=256,
#     input_vocab_size=2564-1 + 1,
#     target_vocab_size=2564-1 + 1,
#     pe_input=16700,
#     pe_target=16700,
#     rate=0.1)

# checkpoint_path = "./checkpoints/len_16700/train"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')


# # transformer = tf.saved_model.load("models/transformer/test")
# predictor = KmerPredictor(transformer)
# # Basic compression loop could look like:
# kmers = build_kmers("GATCACTATAC", ksize)
# predictor.start_kmer_prediction()
# for kidx in range(len(kmers)):
#     # Get kmer predictions
#     probabilities = predictor.get_next_kmer_probabilities()
#     iterate_arithmetic_encoder(probabilities)
#     predicted_kmer_token = tf.argmax(probabilities, axis=-1)
#     predicted_kmer = predictor.detokenize_sequence(predicted_kmer_token)[0][0].numpy().decode('UTF-8')
#     print("Highest prob: ", tf.math.reduce_max(probabilities))
#     print("Predicted next kmer: ", predicted_kmer)
#     # Feed true next 
#     predictor.feedback_next_kmer(kmers[kidx])
# print(predictor.output)
# print(predictor.detokenize_sequence(predictor.output))