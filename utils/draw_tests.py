import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from colour import Color
import matplotlib.patheffects as pe


def read_test(folder_proofs, metric,title="", save_folder="./"):
	plt.figure(figsize=(7,5))
	c1_1 = Color("#ff0000")
	c1_2 = Color("#FFFA00")
	c2_1 = Color("#00b9ff")
	c2_2 = Color("#F200FF")
	c3_1 = Color("#FFBC00")
	c3_2 = Color("#00FF13")
	colors1 = list(c1_1.range_to(c1_2,5))
	colors2 = list(c2_1.range_to(c2_2,5))
	colors = [c1_1,c2_1,c3_1,c3_2,c2_2]
	proofs = os.listdir(folder_proofs)
	for i,folder in enumerate(proofs):
		proof_path = os.path.join(folder_proofs,folder)

		list_increments = os.listdir(proof_path)
		for j,increment in enumerate(list_increments):
			values_x=[]
			values_y=[]
			path_increment = os.path.join(proof_path, increment)
			list_events = os.listdir(path_increment)
			for event_file in list_events:
				print("Processing event file: .../{}/{}".format( increment, event_file))
				for e in tf.train.summary_iterator(os.path.join(path_increment, event_file)):
					for v in e.summary.value:
						if v.tag in [metric]:
							values_x.append(e.step)
							values_y.append(v.simple_value)
				plt.plot(values_x,values_y,colors[j%len(colors)].hex_l,linewidth=2,solid_capstyle="round",zorder=2)
				plt.plot(values_x[0], values_y[0],colors[j%len(colors)].hex_l , markersize=8,marker="o",linewidth=2,zorder=3)
				plt.axvline(values_x[0], linestyle='--', color='k',zorder=1) # vertical lines
		plt.axvline(values_x[-1], linestyle='--', color='k',zorder=1,linewidth=1) # vertical lines
	
	plt.yticks(np.arange(0, 1, step=0.1),fontsize=16)
	plt.xticks(fontsize=13)
	plt.xlabel("iteration")
	plt.ylabel(metric)
	plt.gca().yaxis.grid(True,linestyle='--')
	plt.savefig(save_folder+"/"+folder_proofs+"_"+metric+"_"+title+'.png')


def _shell_exec():
	"""
	Executes the program from a shell

	:return: None
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-i',
		'--input_folder',
		type=str,
		help='The root folder where all the tests are located (e.g. ../summaries/CIFAR-10/TR_BASE).',
		required=True)
	parser.add_argument(
		'-o',
		'--output_folder',
		type=str,
		help='The root folder where the images are saved.',
		required=True)
	parser.add_argument('-m', '--metric', type=str,
						help='the metric to be read (e.g. \'loss\' or  \'accuracy\'',
						required=True)
	
	parser.add_argument('-n', '--name', type=str,
						help='the name of dataset for saving the image',
						required=True)

	args = vars(parser.parse_args())
	print(args)
	
	read_test(args['input_folder'],args['metric'],args['name'],save_folder=args['output_folder'])
	return 0


if __name__ == '__main__':
	_shell_exec()
