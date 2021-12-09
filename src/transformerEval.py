
import logging

import tensorflow as tf

from transformer_kmer import load_data_batches, loss_function, accuracy_function, build_transformer

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings



transformer = build_transformer()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=test_step_signature)
def test_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    tf.print("Running test step")
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     is_training=False)
        loss = loss_function(tar_real, predictions)

    test_loss(loss)
    test_accuracy(accuracy_function(tar_real, predictions))

def measure_test_loss_and_accuracy(test_batches):

    test_loss.reset_states()
    test_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(test_batches):
        test_step(inp, tar)
        tf.print(f'Batch {batch} Loss {test_loss.result():.4f} Accuracy {test_accuracy.result():.4f}')
    tf.print(f'Test Batches Total Loss {test_loss.result():.4f} Accuracy {test_accuracy.result():.4f}')

def evaluate_test_loss_and_accuracy():
    train_batches, val_batches, test_batches = load_data_batches()
    measure_test_loss_and_accuracy(test_batches)