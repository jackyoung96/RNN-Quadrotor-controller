# import tensorflow as tf
import torch

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
				"""+_out+"""[i] += """+_in+"""[j] * linear_"""+_layer+"""_weight[i][j];
			}
			"""+_out+"""[i] += linear_"""+_layer+"""_bias[i];
			"""+_out+"""[i] = tanhf("""+_out+"""[i]);
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


def generate_rnn_helper(path, state_dim, action_dim, goal_dim, output_path=None):

	state_dim = state_dim
	action_dim = action_dim
	goal_dim = goal_dim

	# TODO: check if the policy is really a mlp policy
	model = torch.load(path, map_location="cpu")
	hidden_dim = model['linear1.weight'].numpy().shape[0]

	structure = """static int state_dim = """+str(int(state_dim))+""";\n"""+\
				"""static int action_dim = """+str(int(action_dim))+""";\n"""+\
				"""static int hidden_dim = """+str(int(hidden_dim))+""";\n"""
	if goal_dim > 0:
		structure += """static int goal_dim = """+str(int(goal_dim))+""";\n"""
	
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
				"""static float output_action[%d]={0,0,0,0};\n"""%(action_dim)

	if goal_dim > 0:
		variables += """static const float goal[%d]={0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0};\n"""%(goal_dim)
	variables += "\n"

	code = ""

	# input_loop 
	code += """
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
			rnn_input[i] = state_array[i];
		}
		for (int i = 0; i < action_dim; i++) {
			rnn_input[state_dim + i] = action[i];
		}\n"""
	if goal_dim > 0:
		code += """
		for (int i = 0; i < goal_dim; i++) {
			linear_input[state_dim + i] = goal[i];
		}"""
	
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
		// control_n->thrust_0 = output_action[0];
		// control_n->thrust_1 = output_action[1];
		// control_n->thrust_2 = output_action[2];
		// control_n->thrust_3 = output_action[3];

		control_n->thrust_0 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[0]-action[0]) + action[0];
		control_n->thrust_1 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[1]-action[1]) + action[1];
		control_n->thrust_2 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[2]-action[2]) + action[2];
		control_n->thrust_3 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[3]-action[3]) + action[3];

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


def generate_ff_helper(path, state_dim, action_dim, output_path=None):

	state_dim = state_dim
	action_dim = action_dim

	# TODO: check if the policy is really a mlp policy
	model = torch.load(path, map_location="cpu")
	hidden_dim = model['linear1.weight'].numpy().shape[0]

	structure = """static int state_dim = """+str(int(state_dim))+""";\n"""
	
	# linear_1_weight 
	structure += """static const float linear_1_weight[%d][%d] = """%(hidden_dim,state_dim)+\
						weight2str(model['linear1.weight']) + ";\n"
	# linear_1_bias  
	structure += """static const float linear_1_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear1.bias']) + ";\n"

	# linear_2_weight
	structure += """static const float linear_2_weight[%d][%d] = """%(hidden_dim,hidden_dim)+\
						weight2str(model['linear2.weight']) + ";\n"
	# linear_2_bias
	structure += """static const float linear_2_bias[%d] = """%(hidden_dim)+\
						bias2str(model['linear2.bias']) + ";\n"

	# linear_3_weight
	structure += """static const float linear_3_weight[%d][%d] = """%(hidden_dim,hidden_dim)+\
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

	variables = """static float linear_input[%d];\n"""%(state_dim)+\
				"""static float output_linear1[%d];\n"""%(hidden_dim)+\
				"""static float output_linear2[%d];\n"""%(hidden_dim)+\
				"""static float output_linear3[%d];\n"""%(hidden_dim)+\
				"""static float output_linear4[%d];\n"""%(hidden_dim)+\
				"""static float output_hidden[%d]={0.0};"""%(hidden_dim)+\
				"""static float output_action[%d]={0,0,0,0};\n\n"""%(action_dim)
	code = ""

	# input_loop 
	code += """
		for (int i = 0; i < state_dim; i++) {
			linear_input[i] = state_array[i];
		}\n
	"""
	
	# linear1_loop
	code += get_linear_loop(_in_dim = state_dim,
								_out_dim = hidden_dim,
								_in = "linear_input",
								_out = "output_linear1",
								_layer = "1") +"\n"

	# linear2_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_linear1",
								_out = "output_linear2",
								_layer = "2") +"\n"

	# linear3_loop
	code += get_linear_loop(_in_dim = hidden_dim,
								_out_dim = hidden_dim,
								_in = "output_linear2",
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
		// control_n->thrust_0 = output_action[0];
		// control_n->thrust_1 = output_action[1];
		// control_n->thrust_2 = output_action[2];
		// control_n->thrust_3 = output_action[3];

		control_n->thrust_0 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[0]-action[0]) + action[0];
		control_n->thrust_1 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[1]-action[1]) + action[1];
		control_n->thrust_2 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[2]-action[2]) + action[2];
		control_n->thrust_3 = 4 * (1.0f / 200.0f) / 0.15f * (output_action[3]-action[3]) + action[3];

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
	generate_ff_helper("../artifacts/agent-22Jun30233444:v11/iter0120000_policy.pt", 18, 4, output_path="models/FF_network_evaluate.c")
	# RNN2
	generate_rnn_helper("../artifacts/agent-22Jun23162225:v19/iter0200000_policy.pt", 22, 4, 0, output_path="models/RNN2_network_evaluate.c")
	# RNN-HER
	generate_rnn_helper("../artifacts/agent-22Jun27155927:v10/iter0080000_policy.pt", 18, 4, 18, output_path="models/RNNHER_network_evaluate.c")
	# RNN-sgHER
	generate_rnn_helper("../artifacts/agent-22Jun23223421:v28/iter0236000_policy.pt", 22, 4, 18, output_path="models/RNNsHER_network_evaluate.c")