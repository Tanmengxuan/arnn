import matplotlib.pyplot as plt
import numpy as np

def plot_test(input_test, target, prediction, model_name):

	
	features = input_test.shape[-1]

	target = np.concatenate( [input_test, target], axis = 1 )
	steps = range(target.shape[1])
	start_pred = input_test.shape[1]

	fig, ax = plt.subplots(figsize = (20,8))

	for feature_idx in range(features):

		plt.plot(steps[:], target[:,:,feature_idx].ravel(), label = 'target_{}'.format(feature_idx), color = '#1f77b4')
		plt.plot(steps[start_pred:], prediction[:,:,feature_idx].ravel(), label = 'prediciton_{}'.format(feature_idx), linestyle='dashed', color = '#ff7f0e')

	ax.legend()
	plt.xlabel('x', fontsize = 15)
	plt.ylabel('Y', fontsize = 15)
	plt.title(model_name)
	
	#plt.show()
	save_path = model_name + '.png'
	fig.savefig(save_path)
	print ('\n fig saved in {}'.format(save_path))

