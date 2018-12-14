import tensorflow as tf

converter = tf.contrib.lite.TocoConverter.from_saved_model('home/santanu/Downloads/Mobile App/model/')
tflite_model = converter.convert()
open("home/santanu/Downloads/Mobile App/model/movie_review_model.tflite", "wb").write(tflite_model)

