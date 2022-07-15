# import tensorflow as tf
import torch
import os
import numpy as np

from code_blocks import (
	headers_network_evaluate,
	linear_activation,
	sigmoid_activation,
	relu_activation,
)

def weight2str(weight):
	weight = weight.numpy()
	result = "{"
	for row in weight:
		result += "{"
		for n in row:
			result+=str(n)+","
		result = result[:-1]
		result += "},"
	result = result[:-1]
	return result + "}"

def bias2str(bias):
	bias = bias.numpy()
	result = "{"
	for n in bias:
		result+=str(n)+","
	result = result[:-1]
	return result + "}"

def get_linear_loop(_in_dim, _out_dim, _in, _out, _layer):
	loop = """
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			"""+_out+"""[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				"""+_out+"""[i] += """+_in+"""[j] * """+_layer+"""_weight[i][j];
			}
			"""+_out+"""[i] += """+_layer+"""_bias[i];
			"""+_out+"""[i] = relu("""+_out+"""[i]);
		}
	"""
	return loop

def get_rnn_loop(_in_dim, _out_dim, _in, _out):
	loop = """
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				dummy[i] += """+_in+"""[j] * rnn_weight_ih[i][j] + hidden[j] * rnn_weight_hh[i][j];
			}
			dummy[i] = dummy[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			"""+_out+"""[i] = tanhf(dummy[i]);
		}
	"""
	return loop

def get_lstm_loop(_in_dim, _out_dim, _in, _out):
	loop = """
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy_i[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				dummy_i[i] += """+_in+"""[j] * rnn_weight_ih[i][j] + hidden[j] * rnn_weight_hh[i][j];
			}
			dummy_i[i] = dummy_i[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy_i[i] = sigmoid(dummy_i[i]);
		}

		for (int i = """+str(_out_dim)+"""; i < """+str(2*_out_dim)+"""; i++) {
			dummy_f[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				dummy_f[i] += """+_in+"""[j] * rnn_weight_ih[i][j] + hidden[j] * rnn_weight_hh[i][j];
			}
			dummy_f[i] = dummy_f[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy_f[i] = sigmoid(dummy_f[i]);
		}

		for (int i = """+str(2*_out_dim)+"""; i < """+str(3*_out_dim)+"""; i++) {
			dummy_g[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				dummy_g[i] += """+_in+"""[j] * rnn_weight_ih[i][j] + hidden[j] * rnn_weight_hh[i][j];
			}
			dummy_g[i] = dummy_g[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy_g[i] = tanhf(dummy_g[i]);
		}

		for (int i = """+str(3*_out_dim)+"""; i < """+str(4*_out_dim)+"""; i++) {
			dummy_o[i] = 0;
			for (int j = 0; j < """+str(_in_dim)+"""; j++) {
				dummy_o[i] += """+_in+"""[j] * rnn_weight_ih[i][j] + hidden[j] * rnn_weight_hh[i][j];
			}
			dummy_o[i] = dummy_o[i] + rnn_bias_ih[i] + rnn_bias_hh[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			dummy_o[i] = sigmoid(dummy_o[i]);
		}

		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			output_hidden["""+str(_out_dim)+"""+i] = dummy_f[i] * output_hidden["""+str(_out_dim)+"""+i] + dummy_i[i] * dummy_g[i];
		}
		for (int i = 0; i < """+str(_out_dim)+"""; i++) {
			output_hidden[i] = dummy_o[i] * tanhf(output_hidden["""+str(_out_dim)+"""+i]);
		}
	"""
	return loop

def generate_lstm_helper(path, state_dim, action_dim, goal_dim, output_path=None):

	state_dim = state_dim
	action_dim = action_dim
	goal_dim = goal_dim

	# TODO: check if the policy is really a mlp policy
	model = torch.load(path, map_location="cpu")
	hidden_dim = model['linear1.weight'].numpy().shape[0]

	structure = """static int state_dim = """+str(int(state_dim))+""";\n"""+\
				"""static int action_dim = """+str(int(action_dim))+""";\n"""+\
				"""static int goal_dim = """+str(int(goal_dim))+""";\n"""+\
				"""static int hidden_dim = """+str(int(hidden_dim))+""";\n"""
	
	# linear_1_weight 
	structure += """static const float linear_1_weight[%d][%d] = """%(hidden_dim,state_dim+goal_dim)+\
						weight2str(model['linear1.weight']) + ";\n"
	# linear_1_bias  
	structure += """static const float linear_1_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear1.bias']) + ";\n"

	# linear_rnn_weight
	structure += """static const float linear_rnn_weight[%d][%d] = """%(hidden_dim,state_dim+action_dim)+\
						weight2str(model['linear_rnn.weight']) + ";\n"
	# linear_rnn_bias
	structure += """static const float linear_rnn_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear_rnn.bias']) + ";\n"
	
	# Only LSTM version
	# rnn_weight_ih
	structure += """static const float rnn_weight_ih[%d][%d] = """%(4*hidden_dim,hidden_dim)+\
						weight2str(model['rnn.weight_ih_l0']) + ";\n"
	# rnn_bias_ih
	structure += """static const float rnn_bias_ih[%d] = """%(4*hidden_dim)+\
						bias2str(model['rnn.bias_ih_l0']) + ";\n"
	# rnn_weight_hh
	structure += """static const float rnn_weight_hh[%d][%d] = """%(4*hidden_dim,hidden_dim)+\
						weight2str(model['rnn.weight_hh_l0']) + ";\n"
	# rnn_bias_hh
	structure += """static const float rnn_bias_hh[%d] = """%(4*hidden_dim)+\
						bias2str(model['rnn.bias_hh_l0']) + ";\n"

	# linear_3_weight
	structure += """static const float linear_3_weight[%d][%d] = """%(hidden_dim,2*hidden_dim)+\
						weight2str(model['linear3.weight']) + ";\n"
	# linear_3_bias
	structure += """static const float linear_3_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear3.bias']) + ";\n"
	# linear_4_weight
	structure += """static const float linear_4_weight[%d][%d] = """%(hidden_dim,hidden_dim)+\
						weight2str(model['linear4.weight']) + ";\n"
	# linear_4_bias
	structure += """static const float linear_4_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear4.bias']) + ";\n"
	# linear_mean_weight
	structure += """static const float linear_mean_weight[%d][%d] = """%(action_dim,hidden_dim)+\
						weight2str(model['mean_linear.weight']) + ";\n"
	# linear_mean_bias
	structure += """static const float linear_mean_bias[%d] = """%(action_dim)+\
						bias2str(model['mean_linear.bias']) + ";\n\n"

	variables = """static float linear_input[%d];\n"""%(state_dim+goal_dim)+\
				"""static float rnn_input[%d];\n"""%(state_dim+action_dim)+\
				"""static float output_linear1[%d];\n"""%(hidden_dim)+\
				"""static float output_linear_rnn[%d];\n"""%(hidden_dim)+\
				"""static float output_hidden[%d] = {0.0};\n"""%(2*hidden_dim)+\
				"""static float dummy_i[%d] = {0.0};\n"""%(hidden_dim)+\
				"""static float dummy_f[%d] = {0.0};\n"""%(hidden_dim)+\
				"""static float dummy_g[%d] = {0.0};\n"""%(hidden_dim)+\
				"""static float dummy_o[%d] = {0.0};\n"""%(hidden_dim)+\
				"""static float output_cat[%d];\n"""%(2*hidden_dim)+\
				"""static float output_linear3[%d];\n"""%(hidden_dim)+\
				"""static float output_linear4[%d];\n"""%(hidden_dim)+\
				"""static float output_action[%d]={0,0,0,0};\n"""%(action_dim)+\
				"""static const float goal[%d]={0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0};\n\n"""%(goal_dim)

	code = ""

	# input_loop 
	code += """
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
			rnn_input[i] = state_array[i];
		}
		for (int i = 0; i < goal_dim; i++) {
			linear_input[state_dim + i] = goal[i];
		}
		for (int i = 0; i < action_dim; i++) {
			rnn_input[state_dim + i] = action[i];
		}\n
	"""
	
	# linear1_loop
	code += get_linear_loop(_in_dim = state_dim+goal_dim,
								_out_dim = hidden_dim,
								_in = "linear_input",
								_out = "output_linear1",
								_layer = "1") +"\n"
	
	# linear_rnn_loop
	code += get_linear_loop(_in_dim = state_dim+action_dim,
								_out_dim = hidden_dim,
								_in = "rnn_input",
								_out = "output_linear_rnn",
								_layer = "rnn") +"\n"
	
	# rnn_loop
	code += get_lstm_loop(_in_dim = hidden_dim,
						_out_dim = hidden_dim,
						_in = "output_linear_rnn",
						_out = "output_hidden") +"\n"

	# cat_loop
	code += """
		for (int i = 0; i < hidden_dim; i++) {
			output_cat[i] = output_linear1[i];
			output_cat[hidden_dim + i] = output_hidden[i];
		}
	""" +"\n"

	# linear3_loop
	code += get_linear_loop(_in_dim = 2*hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_cat",
								_out = "output_linear3",
								_layer = "3") +"\n"

	# linear4_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_linear3",
								_out = "output_linear4",
								_layer = "4") +"\n"

	# linear_mean_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = action_dim,
								_in = "output_linear4",
								_out = "output_action",
								_layer = "mean") +"\n"


	## assign network outputs to control
	assignment = """
		control_n->thrust_0 = output_action[0];
		control_n->thrust_1 = output_action[1];
		control_n->thrust_2 = output_action[2];
		control_n->thrust_3 = output_action[3];	

		for (int i=0; i<4; i++) {
			action[i] = output_action[i];
		}
		for (int i=0; i<%d; i++) {
			hidden[i] = output_hidden[i];
		}
	"""%(2*hidden_dim)

	## construct the network evaluation function
	controller_eval = """
	void networkEvaluate(struct control_t_n *control_n, const float *state_array, float *action, float *hidden) {
	"""
	controller_eval += code 
	## assignment to control_n
	controller_eval += assignment

	## closing bracket
	controller_eval += """
	}
	"""

	## combine the all the codes
	source = ""
	## headers
	source += headers_network_evaluate
	## helper functions
	source += linear_activation
	source += sigmoid_activation
	source += relu_activation
	## the network evaluation function
	source += structure
	source += variables
	source += controller_eval

	if output_path:
		with open(output_path, 'w') as f:
			f.write(source)

	return source

def generate_rnn_helper(path, state_dim, action_dim, goal_dim, output_path=None):

	state_dim = state_dim
	action_dim = action_dim
	goal_dim = goal_dim

	# TODO: check if the policy is really a mlp policy
	model = torch.load(path, map_location="cpu")
	hidden_dim = model['linear1.weight'].numpy().shape[0]

	structure = """static int state_dim = """+str(int(state_dim))+""";\n"""+\
				"""static int action_dim = """+str(int(action_dim))+""";\n"""+\
				"""static int goal_dim = """+str(int(goal_dim))+""";\n"""+\
				"""static int hidden_dim = """+str(int(hidden_dim))+""";\n"""
	
	# linear_1_weight 
	structure += """static const float linear_1_weight[%d][%d] = """%(hidden_dim,state_dim+goal_dim)+\
						weight2str(model['linear1.weight']) + ";\n"
	# linear_1_bias  
	structure += """static const float linear_1_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear1.bias']) + ";\n"

	# linear_rnn_weight
	structure += """static const float linear_rnn_weight[%d][%d] = """%(hidden_dim,state_dim+action_dim)+\
						weight2str(model['linear_rnn.weight']) + ";\n"
	# linear_rnn_bias
	structure += """static const float linear_rnn_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear_rnn.bias']) + ";\n"
	
	# Only LSTM version
	# rnn_weight_ih
	structure += """static const float rnn_weight_ih[%d][%d] = """%(hidden_dim,hidden_dim)+\
						weight2str(model['rnn.weight_ih_l0']) + ";\n"
	# rnn_bias_ih
	structure += """static const float rnn_bias_ih[%d] = """%(hidden_dim)+\
						bias2str(model['rnn.bias_ih_l0']) + ";\n"
	# rnn_weight_hh
	structure += """static const float rnn_weight_hh[%d][%d] = """%(hidden_dim,hidden_dim)+\
						weight2str(model['rnn.weight_hh_l0']) + ";\n"
	# rnn_bias_hh
	structure += """static const float rnn_bias_hh[%d] = """%(hidden_dim)+\
						bias2str(model['rnn.bias_hh_l0']) + ";\n"

	# linear_3_weight
	structure += """static const float linear_3_weight[%d][%d] = """%(hidden_dim,2*hidden_dim)+\
						weight2str(model['linear3.weight']) + ";\n"
	# linear_3_bias
	structure += """static const float linear_3_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear3.bias']) + ";\n"
	# linear_4_weight
	structure += """static const float linear_4_weight[%d][%d] = """%(hidden_dim,hidden_dim)+\
						weight2str(model['linear4.weight']) + ";\n"
	# linear_4_bias
	structure += """static const float linear_4_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear4.bias']) + ";\n"
	# linear_mean_weight
	structure += """static const float linear_mean_weight[%d][%d] = """%(action_dim,hidden_dim)+\
						weight2str(model['mean_linear.weight']) + ";\n"
	# linear_mean_bias
	structure += """static const float linear_mean_bias[%d] = """%(action_dim)+\
						bias2str(model['mean_linear.bias']) + ";\n\n"

	variables = """static float linear_input[%d];\n"""%(state_dim+goal_dim)+\
				"""static float rnn_input[%d];\n"""%(state_dim+action_dim)+\
				"""static float output_linear1[%d];\n"""%(hidden_dim)+\
				"""static float output_linear_rnn[%d];\n"""%(hidden_dim)+\
				"""static float output_hidden[%d] = {0.0};\n"""%(2*hidden_dim)+\
				"""static float dummy[%d] = {0.0};\n"""%(hidden_dim)+\
				"""static float output_cat[%d];\n"""%(2*hidden_dim)+\
				"""static float output_linear3[%d];\n"""%(hidden_dim)+\
				"""static float output_linear4[%d];\n"""%(hidden_dim)+\
				"""static float output_action[%d]={0,0,0,0};\n"""%(action_dim)+\
				"""static const float goal[%d]={0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0};\n\n"""%(goal_dim)

	code = ""

	# input_loop 
	code += """
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
			rnn_input[i] = state_array[i];
		}
		for (int i = 0; i < goal_dim; i++) {
			linear_input[state_dim + i] = goal[i];
		}
		for (int i = 0; i < action_dim; i++) {
			rnn_input[state_dim + i] = action[i];
		}\n
	"""
	
	# linear1_loop
	code += get_linear_loop(_in_dim = state_dim+goal_dim,
								_out_dim = hidden_dim,
								_in = "linear_input",
								_out = "output_linear1",
								_layer = "1") +"\n"
	
	# linear_rnn_loop
	code += get_linear_loop(_in_dim = state_dim+action_dim,
								_out_dim = hidden_dim,
								_in = "rnn_input",
								_out = "output_linear_rnn",
								_layer = "rnn") +"\n"
	
	# rnn_loop
	code += get_rnn_loop(_in_dim = hidden_dim,
						_out_dim = hidden_dim,
						_in = "output_linear_rnn",
						_out = "output_hidden") +"\n"

	# cat_loop
	code += """
		for (int i = 0; i < hidden_dim; i++) {
			output_cat[i] = output_linear1[i];
			output_cat[hidden_dim + i] = output_hidden[i];
		}
	""" +"\n"

	# linear3_loop
	code += get_linear_loop(_in_dim = 2*hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_cat",
								_out = "output_linear3",
								_layer = "3") +"\n"

	# linear4_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_linear3",
								_out = "output_linear4",
								_layer = "4") +"\n"

	# linear_mean_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = action_dim,
								_in = "output_linear4",
								_out = "output_action",
								_layer = "mean") +"\n"


	## assign network outputs to control
	assignment = """
		control_n->thrust_0 = output_action[0];
		control_n->thrust_1 = output_action[1];
		control_n->thrust_2 = output_action[2];
		control_n->thrust_3 = output_action[3];	

		for (int i=0; i<4; i++) {
			action[i] = output_action[i];
		}
		for (int i=0; i<%d; i++) {
			hidden[i] = output_hidden[i];
		}
	"""%(hidden_dim)

	## construct the network evaluation function
	controller_eval = """
	void networkEvaluate(struct control_t_n *control_n, const float *state_array, float *action, float *hidden) {
	"""
	controller_eval += code 
	## assignment to control_n
	controller_eval += assignment

	## closing bracket
	controller_eval += """
	}
	"""

	## combine the all the codes
	source = ""
	## headers
	source += headers_network_evaluate
	## helper functions
	source += linear_activation
	source += sigmoid_activation
	source += relu_activation
	## the network evaluation function
	source += structure
	source += variables
	source += controller_eval

	if output_path:
		with open(output_path, 'w') as f:
			f.write(source)

	return source


def generate_ff_helper(source_path, output_path=None):

	# TODO: check if the policy is really a mlp policy
	source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),source_path)
	model = torch.load(source_path, map_location="cpu")
	model_key = np.unique([key.replace('.weight','').replace('.bias','') for key in model.keys() if 'actor' in key and 'log_std' not in key])
	model_latent_key = np.sort([key for key in model_key if 'latent_pi' in key]).tolist()
	model_mu_key = np.sort([key for key in model_key if 'mu' in key]).tolist()

	state_dim = model[model_latent_key[0]+".weight"].shape[1]
	action_dim = model[model_mu_key[-1]+".bias"].shape[0]


	structure = """static int state_dim = """+str(int(state_dim))+""";\n"""+\
				"""static int action_dim = """+str(int(action_dim))+""";\n"""
	
	# linear_latents
	for key in model_latent_key:
		structure += """static const float %s_weight[%d][%d] = """%(key.replace('.','_'),*model[key+".weight"].shape)+\
						weight2str(model[key+'.weight']) + ";\n"
		structure += """static const float %s_bias[%d] = """%(key.replace('.','_'),*model[key+".bias"].shape)+\
						bias2str(model[key+'.bias']) + ";\n"

	# linear_mus
	for key in model_mu_key:
		structure += """static const float %s_weight[%d][%d] = """%(key.replace('.','_'),*model[key+".weight"].shape)+\
						weight2str(model[key+'.weight']) + ";\n"
		structure += """static const float %s_bias[%d] = """%(key.replace('.','_'),*model[key+".bias"].shape)+\
						bias2str(model[key+'.bias']) + ";\n"

	variables = """static float linear_input[%d];\n"""%(state_dim)
	for key in model_latent_key:
		variables += """static float output_%s[%d];\n"""%(key.replace('.','_'),*model[key+".bias"].shape)
	variables += """static float output_action[%d]={-1,-1,-1,-1};\n\n"""%(action_dim)


	# input_loop 
	code = """
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
		}\n
	"""
	
	# linear_loop
	in_keys = ["linear_input"]+["output_"+key.replace('.','_') for key in model_latent_key]
	out_keys = ["output_"+key.replace('.','_') for key in model_latent_key] + ["output_action"]
	layer_keys = [key.replace('.','_') for key in model_latent_key] + [key.replace('.','_') for key in model_mu_key]
	for in_key, out_key, layer_key in zip(in_keys, out_keys, layer_keys):
		code += get_linear_loop(_in_dim = model[key+".weight"].shape[1],
									_out_dim = model[key+".weight"].shape[0],
									_in = in_key,
									_out = out_key,
									_layer = layer_key) +"\n"

	## assign network outputs to control
	assignment = """
		control_n->thrust_0 = output_action[0];
		control_n->thrust_1 = output_action[1];
		control_n->thrust_2 = output_action[2];
		control_n->thrust_3 = output_action[3];	

		for (int i=0; i<4; i++) {
			action[i] = output_action[i];
		}
	"""

	## construct the network evaluation function
	controller_eval = """
	void networkEvaluate(struct control_t_n *control_n, const float *state_array, float *action, float *hidden) {
	"""
	controller_eval += code 
	## assignment to control_n
	controller_eval += assignment

	## closing bracket
	controller_eval += """
	}
	"""

	## combine the all the codes
	source = ""
	## headers
	source += headers_network_evaluate
	## helper functions
	source += linear_activation
	source += sigmoid_activation
	source += relu_activation
	## the network evaluation function
	source += structure
	source += variables
	source += controller_eval


	if output_path:
		filename = os.path.basename(source_path).replace('.pth', '.c')
		output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),output_path)
		if not os.path.isdir(output_path):
			os.makedirs(output_path)
		with open(os.path.join(output_path,filename), 'w') as f:
			f.write(source)

	return source



if __name__ == "__main__" :


	# # None 
	# generate_ff_helper(path, 22, 4, output_path=output_path)
	# # LSTM2
	# generate_lstm_helper(path, 22, 4, 0, output_path=output_path)
	# # LSTM-HER
	# generate_lstm_helper(path, 22, 4, 18, output_path=output_path)
	# # LSTM-sgHER
	# generate_lstm_helper(path, 22, 4, 0, output_path=output_path)

	# None 
	generate_ff_helper("source/sb3/sac_policy.pth",output_path="models/sb3")
	# RNN2
	# generate_rnn_helper(path, 22, 4, 0, output_path=output_path)
	# # RNN-HER
	# generate_rnn_helper(path, 22, 4, 18, output_path=output_path)
	# # RNN-sgHER
	# generate_rnn_helper(path, 22, 4, 0, output_path=output_path)