from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

import numpy as np
import pandas as pd
import os
import time
import argparse
import pdb
import h5py
import time
import json
from utils import plot_test

parser = argparse.ArgumentParser() 

#Commands
parser.add_argument('--train', action = 'store_true', help = "train the model")
parser.add_argument('--test',  action = 'store_true', help = "test the model")
parser.add_argument('--test_file', action = 'store_true',  help = 'test the entire file')
parser.add_argument('--plot_sample',  action = 'store_true', help = "plot predicted sample")

#Training and test parameters
parser.add_argument('--epoch', default = 150, type = int, help = 'number of epcohs to train')
parser.add_argument('--batch_size', default = 64, type = int, help = 'mini batch size')
parser.add_argument('--rnn', default = 20, type = int, help = 'number rnn units')
parser.add_argument('--len', default = 40 ,type = int, help = 'length of sequence')
parser.add_argument('--drop', default = 0.001 ,type = float, help = 'dropout rate')

#Input and output
parser.add_argument('--model_name', default ='ar_toy',type = str, help = 'name of model to save')
parser.add_argument('--save_path', default ='checkpoints/',type = str, help = "path of save model")

#Task parameters
parser.add_argument('--test_steps', default =5 ,type = int, help = 'number of future steps to predict')
parser.add_argument('--test_sample', default = '0:6' , type = str, help = 'a sequence of T test sample specify by the start_index:end_index')

args = parser.parse_args()

if args.test:
	tf.enable_eager_execution(config = config)


class Pipeline(object):

	'''
	A class used to set up the data pipelines for different datasets during model training
	'''

	def __init__(self, data, seq_len, batch_size):

		self.data = data
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.buffer_size = 8000

	def split_input_target(self, chunk):
		
		input_data = chunk[:-1]
		target_text = chunk[1:]
		
		return input_data, target_text

	def data_pipe(self):

		examples_per_epoch = len(self.data)//self.seq_len
		#print ('examples_per_epoch', examples_per_epoch)
		# Create training examples / targets
		tf_dataset = tf.data.Dataset.from_tensor_slices(self.data)
		dataset_slide = tf_dataset.window(self.seq_len, shift=1, drop_remainder=True)
		sequences_slide = dataset_slide.flat_map(lambda window: window.batch(self.seq_len))
		examples_per_epoch = len(self.data) - self.seq_len + 1

		dataset = sequences_slide.map( self.split_input_target, num_parallel_calls = 4)

		steps_per_epoch = examples_per_epoch//self.batch_size

		dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
		dataset = dataset.repeat()
		dataset = dataset.prefetch(1)

		return dataset, steps_per_epoch

class Predict(object):

	def __init__(self, model_test, test_data, input_lens, num_samples):
		
		self.model_test = model_test
		self.test_data = test_data
		self.input_lens = input_lens
		self.num_samples = num_samples


	def prediction(self):

		all_predictions = []
		all_targets = []

		for idx in range(len(self.input_lens)):

			input_len = self.input_lens[idx]
			num_sample = self.num_samples[idx]
			start_idx = 0 if args.test_file else int(args.test_sample.split(':')[0])
			end_idx = start_idx + input_len 

			for sample_idx in range(num_sample):
				self.model_test.reset_states()
				
				input_test = self.test_data[start_idx: end_idx].reshape(1, -1, self.test_data.shape[-1]).astype('float32') 
				test_target = self.test_data[end_idx: end_idx + args.test_steps ].reshape(1, -1, self.test_data.shape[-1]).astype('float32') 

				start_idx = end_idx + args.test_steps
				end_idx = start_idx + input_len

				input_seq = input_test
				all_targets.append(test_target)

				for step in range(args.test_steps):
					# pick last prediction in sequence
					prediction = self.model_test(input_test).numpy()[:, -1:, :] 
					all_predictions.append(prediction[0])
					
					input_test = prediction

		all_predictions = np.array(all_predictions).reshape(1,-1, self.test_data.shape[-1])
		all_targets = np.array(all_targets).reshape(1, -1, self.test_data.shape[-1])

		test_loss = rmse(all_targets, all_predictions)

		return test_loss, input_seq, all_targets, all_predictions

class RNN(object):
	
	def __init__(self, units, drop_rate, feature_size, batch_size, state):
		
		self.units = units
		self.drop = drop_rate
		self.feature_size = feature_size
		self.batch_size = batch_size
		self.state = state
		
	def rnn_model(self):

		model_input = tf.keras.layers.Input(batch_shape = (self.batch_size, None, self.feature_size))

		rnn =  tf.keras.layers.CuDNNGRU( self.units,
			return_sequences=True,
			recurrent_initializer='glorot_uniform',
			stateful= self.state)(model_input)
		rnn = tf.keras.layers.Dropout(self.drop)(rnn)

		model_output =  tf.keras.layers.Dense(self.feature_size, activation = 'sigmoid')(rnn)

		return model_input, model_output


def rmse(y_true, y_pred):
	
	'''
	root mean squared loss function used to train model
	'''

	return tf.math.sqrt(tf.losses.mean_squared_error(y_true, y_pred))


def get_rnn(rnn_units, drop_rate, batch_size, num_features, state = False):
	
		
	feature_size = num_features
	model_input, model_output = RNN(rnn_units, drop_rate, feature_size, batch_size, state).rnn_model()
		
	model = tf.keras.models.Model(inputs = model_input , outputs = model_output)
	
	return model

def sin_wave(amp = 1, w = 1, y = 0):
	
	x = np.arange(0, 100*np.pi, 0.31)

	return amp* np.sin( w* x) + y

def main(train, val, num_features):

	if args.train:
		train_data, steps_per_epoch_train = Pipeline(train, args.len, args.batch_size).data_pipe()
		val_data, steps_per_epoch_val = Pipeline(val, args.len, args.batch_size).data_pipe() 

		model = get_rnn(args.rnn, args.drop, args.batch_size, num_features)

		model.compile(
		optimizer = tf.train.AdamOptimizer(),
		loss = rmse)
		
		#tf.keras.utils.plot_model(model)
		print (model.summary())

		tf.keras.backend.set_session(tf.Session(config=config))

		checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
		filepath= args.save_path + args.model_name,
		save_weights_only=True,
		monitor = 'val_loss',
		save_best_only = True 
		)

		tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(args.model_name))

		history = model.fit(train_data, epochs= args.epoch, steps_per_epoch=steps_per_epoch_train,
		validation_data = val_data, validation_steps = steps_per_epoch_val, 
		callbacks=[checkpoint_callback, tensorboard])

		min_train_loss = min(history.history['loss'])
		min_train_val_loss = min(history.history['val_loss'])

		print ('\n train: {}, validation: {}'.format(min_train_loss, min_train_val_loss)) 

	if args.test:

		start_time = time.time()
		batch_size = 1
		model_test = get_rnn( args.rnn, args.drop, batch_size, num_features, state = True)
		model_test.load_weights(args.save_path + args.model_name)
		
		test_data = val

		if args.test_file:
			input_lens = [24, 48, 96, 164, 192]
			num_samples = []

			for input_len in input_lens:
				num_samples.append(len(test_data) // ( input_len + args.test_steps))

		elif args.test_sample:
			start, end = args.test_sample.split(':')
			start_sample_idx = int(start)
			end_sample_idx = int(end) + 1

			input_lens = [ end_sample_idx - start_sample_idx ] 
			num_samples = [1]	


		test_loss, input_seq, all_targets, all_predictions = Predict(model_test, test_data, input_lens, num_samples).prediction()
		print ('\n test_loss: {}'.format(test_loss)) 
		print ('inference time :{}'.format(time.time() - start_time))

		if args.plot_sample:
			plot_test(input_seq, all_targets, all_predictions, args.model_name)


if __name__ == '__main__':


	wave_1 = sin_wave(amp = 0.5, w=1, y = 0.5 ).reshape(-1, 1)
	wave_2 = sin_wave(amp = 0.25, w=1, y = 0.5 ).reshape(-1, 1)
	wave_3 = sin_wave(amp = 0.4, w=2, y = 0.5 ).reshape(-1, 1)
	data = np.concatenate([wave_1, wave_2, wave_3], axis = -1)
	
	num_features = data.shape[-1]

	train = data[:int (len(data) * 0.8)].round(3).astype('float32')
	val = data[int (len(data) * 0.8):].round(3).astype('float32')


	main(train, val, num_features)

