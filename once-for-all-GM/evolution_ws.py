# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import numpy as np
from tqdm import tqdm
import json
from elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3_layer_settting
from utils import make_divisible
import torch

__all__ = ['EvolutionFinder']


def count_conv_flop(out_size, in_channels, out_channels, kernel_size, groups):
	out_h = out_w = out_size
	delta_ops = in_channels * out_channels * kernel_size * kernel_size * out_h * out_w / groups
	return delta_ops

def count_flops_given_config(net_config, image_size=224):
	flops = 0
	# first conv
	flops += count_conv_flop((image_size + 1) // 2, 3, net_config['first_conv']['out_channels'], 3, 1)
	# blocks
	fsize = (image_size + 1) // 2
	for block in net_config['blocks']:
		mb_conv = block['mobile_inverted_conv'] if 'mobile_inverted_conv' in block else block['conv']
		if mb_conv is None:
			continue
		out_fz = int((fsize - 1) / mb_conv['stride'] + 1)
		# if mb_conv['mid_channels'] is None:
		if 'in_channel_list' in mb_conv.keys():
			mb_conv['in_channels'] = mb_conv['in_channel_list'][0]
		if 'out_channel_list' in mb_conv.keys():
			mb_conv['out_channels'] = mb_conv['out_channel_list'][0]
		if 'kernel_size_list' in mb_conv.keys():
			mb_conv['kernel_size'] = mb_conv['kernel_size_list'][0]
		if 'expand_ratio_list' in mb_conv.keys():
			mb_conv['expand_ratio'] = mb_conv['expand_ratio_list'][0]
		mb_conv['mid_channels'] = round(mb_conv['in_channels'] * mb_conv['expand_ratio'])
		if mb_conv['expand_ratio'] != 1:
			# inverted bottleneck
			flops += count_conv_flop(fsize, mb_conv['in_channels'], mb_conv['mid_channels'], 1, 1)
		# depth conv
		flops += count_conv_flop(out_fz, mb_conv['mid_channels'], mb_conv['mid_channels'],
								 mb_conv['kernel_size'], mb_conv['mid_channels'])
		if mb_conv['use_se']:
			# SE layer
			se_mid = make_divisible(mb_conv['mid_channels'] // 4, divisor=8)
			flops += count_conv_flop(1, mb_conv['mid_channels'], se_mid, 1, 1)
			flops += count_conv_flop(1, se_mid, mb_conv['mid_channels'], 1, 1)
		# point linear
		flops += count_conv_flop(out_fz, mb_conv['mid_channels'], mb_conv['out_channels'], 1, 1)
		fsize = out_fz
	# final expand layer
	flops += count_conv_flop(fsize, net_config['final_expand_layer']['in_channels'],
							 net_config['final_expand_layer']['out_channels'], 1, 1)
	# feature mix layer
	flops += count_conv_flop(1, net_config['feature_mix_layer']['in_channels'],
							 net_config['feature_mix_layer']['out_channels'], 1, 1)
	# classifier
	flops += count_conv_flop(1, net_config['classifier']['in_features'],
							 net_config['classifier']['out_features'], 1, 1)
	return flops / 1e6  # MFLOPs


class EvolutionFinder:

	def __init__(self, accuracy_predictor, arch_manager, **kwargs):
		# self.efficiency_predictor = efficiency_predictor
		self.accuracy_predictor = accuracy_predictor
		self.arch_manager = arch_manager

		# evolution hyper-parameters
		self.arch_mutate_prob = kwargs.get('arch_mutate_prob', 0.1)
		self.resolution_mutate_prob = kwargs.get('resolution_mutate_prob', 0.5)
		# self.population_size = kwargs.get('population_size', 50)
		self.population_size = kwargs.get('population_size', 100)
		self.max_time_budget = kwargs.get('max_time_budget', 500)
		self.parent_ratio = kwargs.get('parent_ratio', 0.25)
		self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

		self.depth_list = [2, 3, 4]

	# @property
	# def arch_manager(self):
	# 	return self.accuracy_predictor.arch_encoder

	def update_hyper_params(self, new_param_dict):
		self.__dict__.update(new_param_dict)

	def random_valid_sample(self, constraint_lower, constraint_upper, input_size=224):
		while True:
			sample = self.arch_manager.random_sample_arch()

			new_sample = {}
			new_ks_list = []
			new_expand_ratio_list = []
			index = 0
			for i in range(5):
				for depth_id in range(sample['d'][i]):
					new_ks_list.append(sample['ks'][depth_id+index])
					new_expand_ratio_list.append(sample['e'][depth_id+index])
				index += max(self.depth_list)
			new_sample['ks'] = new_ks_list
			new_sample['e'] = new_expand_ratio_list
			new_sample['d'] = sample['d']

			efficiency = self.get_flops(new_sample, input_size)
			if efficiency <= constraint_upper and efficiency >= constraint_lower:
				return sample, efficiency

	def get_flops(self, sample, input_size):
		net = OFAMobileNetV3_layer_settting(
			dropout_rate=0, width_mult_list=1.0, ks_list=sample['ks'], expand_ratio_list=sample['e'], depth_list=sample['d'],
		)
		flops = count_flops_given_config(net.config, image_size=input_size)
		del net
		return flops

	def mutate_sample(self, sample, constraint_lower, constraint_upper, input_size=224):
		while True:
			new_sample = copy.deepcopy(sample)

			# self.arch_manager.mutate_resolution(new_sample, self.resolution_mutate_prob)
			self.arch_manager.mutate_arch(new_sample, self.arch_mutate_prob)

			efficiency = self.get_flops(new_sample, input_size)
			if efficiency <= constraint_upper and efficiency >= constraint_lower:
				return new_sample, efficiency

	def crossover_sample(self, sample1, sample2, constraint_lower, constraint_upper, input_size=224):
		while True:
			new_sample = copy.deepcopy(sample1)
			for key in new_sample.keys():
				if not isinstance(new_sample[key], list):
					new_sample[key] = random.choice([sample1[key], sample2[key]])
				else:
					for i in range(len(new_sample[key])):
						new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])

			efficiency = self.get_flops(new_sample, input_size)
			if efficiency <= constraint_upper and efficiency >= constraint_lower:
				return new_sample, efficiency

	def run_evolution_search(self, constraint_lower, constraint_upper, input_size=224, verbose=False, **kwargs):
		"""Run a single roll-out of regularized evolution to a fixed time budget."""
		self.update_hyper_params(kwargs)

		mutation_numbers = int(round(self.mutation_ratio * self.population_size))
		parents_size = int(round(self.parent_ratio * self.population_size))

		best_valids = [-100]
		population = []  # (validation, sample, latency) tuples
		child_pool = []
		efficiency_pool = []
		best_info = None
		if verbose:
			print('Generate random population...')
		for _ in range(self.population_size):
			sample, efficiency = self.random_valid_sample(constraint_lower, constraint_upper, input_size=input_size)
			child_pool.append(sample)
			efficiency_pool.append(efficiency)

		accs = self.accuracy_predictor.get_arch_acc(child_pool, efficiency_pool)
		print('finish calculate the accuracy of models')
		for i in range(self.population_size):
			population.append((accs[i], child_pool[i], efficiency_pool[i]))

		if verbose:
			print('Start Evolution...')
		# After the population is seeded, proceed with evolving the population.
		with tqdm(total=self.max_time_budget, desc='Searching with constraint (%s)' % constraint_upper,
		          disable=(not verbose)) as t:
			for i in range(self.max_time_budget):
				parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
				acc = parents[0][0]
				t.set_postfix({
					'acc': parents[0][0]
				})
				if not verbose and (i + 1) % 100 == 0:
					print('Iter: {} Acc: {}'.format(i + 1, parents[0][0]))

				if acc > best_valids[-1]:
					best_valids.append(acc)
					best_info = parents[0]
				else:
					best_valids.append(best_valids[-1])

				population = parents
				child_pool = []
				efficiency_pool = []

				print('perfrom mutation')
				for j in range(mutation_numbers):
					par_sample = population[np.random.randint(parents_size)][1]
					# Mutate
					new_sample, efficiency = self.mutate_sample(par_sample, constraint_lower, constraint_upper, input_size=input_size)
					child_pool.append(new_sample)
					efficiency_pool.append(efficiency)

				print('perfrom crossover')
				for j in range(self.population_size - mutation_numbers):
					par_sample1 = population[np.random.randint(parents_size)][1]
					par_sample2 = population[np.random.randint(parents_size)][1]
					# Crossover
					new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2, constraint_lower, constraint_upper, input_size=input_size)
					child_pool.append(new_sample)
					efficiency_pool.append(efficiency)

				print('get model acc')
				accs = self.accuracy_predictor.get_arch_acc(child_pool, efficiency_pool)
				for j in range(self.population_size):
					population.append((accs[j], child_pool[j], efficiency_pool[j]))

				t.update(1)

		return best_valids, best_info



