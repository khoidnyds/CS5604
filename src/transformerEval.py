
import logging

import tensorflow as tf
# from tfa.metrics import F1Score

from transformer_kmer import load_data_batches, loss_function, accuracy_function, build_transformer, checkpoint_path
from kmerPredictor import KmerPredictor

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


precision_object = tf.keras.metrics.Precision(top_k=1)
recall_object = tf.keras.metrics.Recall(top_k=1)

metrics = {
    "precision": [],
    "recall": [],
    "f1": [],
    "loss": [],
    "accuracy": [],
}

transformer = build_transformer()
kmerPredictor = KmerPredictor(transformer)

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

test_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=test_step_signature)
def measure_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = transformer([inp, tar_inp],
                                is_training=False)
    probabilities = tf.nn.softmax(predictions)
    loss = loss_function(tar_real, predictions)
    measure_precision(tar_real, probabilities)
    measure_recall(tar_real, probabilities)

    test_loss(loss)
    test_accuracy(accuracy_function(tar_real, predictions))


def measure_precision_recall_f1(data_batches):

    test_loss.reset_states()
    test_accuracy.reset_states()
    precision_object.reset_states()
    recall_object.reset_states()

    for (batch, (inp, tar)) in enumerate(data_batches):
        measure_step(inp, tar)
        f1_score = calculate_f1(precision_object.result(), recall_object.result())
        tf.print(f'Batch {batch} Precision {precision_object.result():.4f}')
        tf.print(f'Batch {batch} Recall {recall_object.result():.4f}')
        tf.print(f'Batch {batch} F1 {f1_score:.4f}')
        tf.print(f'Batch {batch} Loss {test_loss.result():.4f}')
        tf.print(f'Batch {batch} Accuracy {test_accuracy.result():.4f}')
    
    tf.print(f'Total Precision {precision_object.result()}')
    tf.print(f'Total Recall {recall_object.result()}')
    tf.print(f'Total F1 {f1_score}')
    tf.print(f'Total Loss {test_loss.result()}')
    tf.print(f'Total Accuracy {test_accuracy.result()}')
    metrics["precision"].append(precision_object.result())
    metrics["recall"].append(recall_object.result())
    metrics["f1"].append(f1_score)
    metrics["loss"].append(test_loss.result())
    metrics["accuracy"].append(test_accuracy.result())

def evaluate_test_set():
    metrics["precision"].clear()
    metrics["recall"].clear()
    metrics["f1"].clear()
    metrics["loss"].clear()
    metrics["accuracy"].clear()

    train_batches, val_batches, test_batches = load_data_batches()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
    epoch = -1
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        tf.print("Latest checkpoint: ", ckpt_manager.latest_checkpoint)
        tf.print('Latest checkpoint restored!!')
        epoch = int(ckpt_manager.latest_checkpoint.split("-")[1])
    tf.print("EPOCH: ", epoch)
    measure_precision_recall_f1(test_batches)
    tf.print("---------------------------------------------")
    tf.print("Metrics: ", metrics)
    tf.print("Final Precisions: ", metrics["precision"])
    tf.print("Final Recall: ", metrics["recall"])
    tf.print("Final F1: ", metrics["f1"])
    tf.print("Final Loss: ", metrics["loss"])
    tf.print("Final Accuracy: ", metrics["accuracy"])
    
def get_validation_metrics(start_epoch=1, end_epoch=19):
    metrics["precision"].clear()
    metrics["recall"].clear()
    metrics["f1"].clear()
    metrics["loss"].clear()
    metrics["accuracy"].clear()

    train_batches, val_batches, test_batches = load_data_batches()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    for epoch in range(start_epoch, end_epoch+1):
        ckpt_file = checkpoint_path + '/ckpt-' + str(epoch)
        ckpt.restore(ckpt_file)
        tf.print("Restored Checkpoint epoch: ", ckpt_file)
        tf.print("EPOCH: ", epoch)
        measure_precision_recall_f1(val_batches)
        tf.print("---------------------------------------------")
    tf.print("Metrics: ", metrics)
    tf.print("Final Precisions: ", metrics["precision"])
    tf.print("Final Recall: ", metrics["recall"])
    tf.print("Final F1: ", metrics["f1"])
    tf.print("Final Loss: ", metrics["loss"])
    tf.print("Final Accuracy: ", metrics["accuracy"])
    

def measure_precision(real, pred):
    lookup_size = tf.cast(kmerPredictor.kmer_lookup.size(), tf.int32)
    real_one_hot = tf.one_hot(real, lookup_size)
    # tf.print("Real one hot: ", real_one_hot)
    precision_object.update_state(real_one_hot, pred)

def measure_recall(real, pred):
    lookup_size = tf.cast(kmerPredictor.kmer_lookup.size(), tf.int32)
    real_one_hot = tf.one_hot(real, lookup_size)
    # tf.print("Real one hot: ", real_one_hot)
    recall_object.update_state(real_one_hot, pred)

def calculate_f1(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))