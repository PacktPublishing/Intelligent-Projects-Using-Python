# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


path = '/home/santanu/Downloads/Mobile_App/'

MODEL_NAME = 'model'

# Freeze the graph

input_graph_path = path + MODEL_NAME+'_cnn.pbtxt'
checkpoint_path = path + 'model_ckpt_cnn'
input_saver_def_path = ""
input_binary = False
output_node_names = 'positive_sentiment_probability'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = path + 'optimized_cnn'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
output_frozen_graph_name, clear_devices, "")

input_graph_def = tf.GraphDef()

with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["inputs/X" ],#an array of the input node(s)
        ["positive_sentiment_probability"],
        tf.int32.as_datatype_enum # an array of output nodes
        )

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())
