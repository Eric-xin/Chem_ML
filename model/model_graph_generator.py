import torch
from torchviz import make_dot
# from simple_model_residue import Net
from simple_model_revised import Net

import os
current_dir = os.path.dirname(__file__)

# Initialize the model
input_dim = 141
output_dim = 6 
model = Net(input_dim, output_dim)

# Create a dummy input tensor
dummy_input = torch.randn(2, input_dim)

model_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# model_graph.graph_attr.update(rankdir='LR')

model_graph.render(current_dir + "/structure_visualize/model_structure", format="png")