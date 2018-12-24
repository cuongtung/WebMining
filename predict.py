#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

from tensorflow.python.eager import context
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import state_ops
# Parameters
# =====================================================================================================================

# Data Parameters       ===============================================


tf.flags.DEFINE_string("positive_data_file", "./data1/Test/positive.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data1/Test/negative.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1544002567/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_path", "./runs/1544002567/checkpoints", "Checkpoint directory from training result")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

#FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
#checkpoint_dir="./runs/1530695379/"
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    print(x_raw)
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab.txt")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    y_test = np.argmax(y_test, axis=1)
   # print(y_test)
else:
    x_raw = ["tuyệt vời", "Giá hơi cao. Đành đợi 1 năm nữa rồi lấy em."]
    y_test = [1, 0]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab.txt")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    acc, acc_op = tf.metrics.accuracy(labels=y_test, predictions=all_predictions)

    rec1, rec_op1 = tf.metrics.precision(labels=y_test[0:int(len(y_test) / 2 + 1)], predictions=all_predictions[0:int(len(all_predictions) / 2 + 1)])
    rec0, rec_op0 = tf.metrics.precision(labels=y_test[int(len(y_test) / 2 + 1):], predictions=all_predictions[int(len(all_predictions) / 2 + 1):])
    pre1, pre_op1 = tf.metrics.recall(labels=y_test[0:int(len(y_test) / 2 + 1)],
                                         predictions=all_predictions[0:int(len(all_predictions) / 2 + 1)])
    pre0, pre_op0 = tf.metrics.recall(labels=y_test[int(len(y_test) / 2 + 1):],
                                         predictions=all_predictions[int(len(all_predictions) / 2 + 1):])


    # predict the class using your classifier
   # scorednn = list(DNNClassifier.predict_classes(input_fn=lambda: input_fn(testing_set)))
    #scoreArr = np.array(scorednn).astype(int)

    # run the session to compare the label with the prediction
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    v = sess.run(acc_op)  # accuracy

    r1 = sess.run(rec_op1)  # recall
    r0 = sess.run(rec_op0)  # recall
    p1 = sess.run(pre_op1)  # precision
    p0 = sess.run(pre_op0)  # precision

    print("accuracy ", v)
    
    correct_predictions = float(sum(all_predictions == y_test))#y_test:nhãn đúng,all_predictions:nhãn dự đoán
    TP1=float(sum(all_predictions[:int(len(all_predictions)/2)+1]==y_test[:int(len(y_test)/2)+1]))
    TN1=float(sum(all_predictions[int(len(all_predictions)/2)+1:]==y_test[int(len(all_predictions)/2)+1:]))
    FN1=float(float(len(all_predictions)/2)-TP1)
    FP1=float(float(len(all_predictions)/2)-TN1)


    Precision_positive=TP1/(TP1+FP1)
    Recall_positive=TP1/(TP1+FN1)
    F1_positive=2*Precision_positive*Recall_positive/(Precision_positive+Recall_positive)
    Precision_nagetive = TN1 / (TN1 + FN1)
    Recall_nagetive = TN1 / (TN1 + FP1)
    F1_nagetive = 2 * Precision_nagetive * Recall_nagetive / (Precision_nagetive + Recall_nagetive)
    print("Precision positive: ",Precision_positive)
    print("Recall positive: ",Recall_positive)
    print("F1 positive: ",F1_positive)
    print("Precision nagetive: ", Precision_nagetive)
    print("Recall nagetive: ", Recall_nagetive)
    print("F1 nagetive: ", F1_nagetive)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w',encoding="UTF-8") as f:
    csv.writer(f).writerows(predictions_human_readable)
