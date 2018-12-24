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

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data1/Test/positive.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data1/Test/negative.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1543965393/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_path", "./runs/1543965393/checkpoints", "Checkpoint directory from training result")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here

while True:
	comment = input("Please enter comment: ")
	print("Comment of you: " + str(comment))
	if(len(comment)==0):
	  break;
	pre_comment=data_helpers.clean_data(comment)
	pre_comment=data_helpers.normalize_Text(pre_comment)
	pre_comment=data_helpers.convert_Abbreviation(pre_comment)
	pre_comment=data_helpers.tokenize(pre_comment)
	pre_comment=data_helpers.remove_Stopword(pre_comment)
	x_raw = [pre_comment]
        

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
	#if y_test is not None:
	#correct_predictions = float(sum(all_predictions == y_test))
	#print("Total number of test examples: {}".format(len(x_raw)))
	#print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

	# Save the evaluation to a csv
	predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
	#out_path = os.path.join(FLAGS.checkpoint_dir, "", "prediction1.csv")
	#print("Saving evaluation to {0}".format(out_path))
	#with open(out_path, 'w') as f:
	#    csv.writer(f).writerows(predictions_human_readable)
	#print(predictions_human_readable)
	if batch_predictions==[1]:
		print("Comment of you: " + str(comment))
		print("Your comment is Positive!!!")
	else:
		print("Comment of you: " + str(comment))
		print("Your comment is Negative!!!")
