import torch
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from generator import *
from data.utils import CompositionEntry

import argparse

def make_prediction(entry_str, model_path, input_dim=141, output_dim=6):
    model = Net(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))

    # Initialize the generators
    charge_generator = ChargeDependentAttributeGenerator()
    elemental_generator = ElementalPropertyAttributeGenerator()
    ionicity_generator = IonicityAttributeGenerator()
    stoichiometric_generator = StoichiometricAttributeGenerator()

    # Make predictions
    model.eval()
    with torch.no_grad():
        entry = CompositionEntry(entry_str)
        stoichiometric = np.array(stoichiometric_generator.generate_features([entry])).flatten()
        ionicity = np.array(ionicity_generator.generate_features([entry])).flatten()
        elemental = np.array(elemental_generator.generate_features([entry])).flatten()
        # charge = np.array(charge_generator.generate_features([entry])).flatten()
        
        # feature = np.concatenate([stoichiometric, ionicity, elemental, charge])
        feature = np.concatenate([stoichiometric, ionicity, elemental])
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        prediction = model(feature)
        return prediction

# Example usage:
# prediction = make_prediction("Y2I6", current_dir + '/model/' + 'trained_model.pth', stoichiometric_generator, ionicity_generator, elemental_generator)
# print(prediction)

def print_prediction(pred):
    print("-"*50)
    print("Model predictions:")
    for i, prop in enumerate(['bandgap', 'energy_pa', 'volume_pa', 'magmom_pa', 'fermi', 'delta_e']):
        print(f"{prop}: {pred[0][i].item()}")
    print("-"*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChemML model inference')
    parser.add_argument('--entry', type=str, help='Entry Chemical Formula')
    parser.add_argument('--model', type=str, help='Model Path')
    parser.add_argument('--network', type=str, default='model.simple_model_residue.Net', help='Path to the model class')
    args = parser.parse_args()

    print("-"*50)
    print("Arguments:")
    for arg in vars(parser.parse_args()):
        print(f"\t{arg}: {getattr(parser.parse_args(), arg)}")
    print("-"*50)

    args = parser.parse_args()
    model_origin = args.network.split('.')
    # print(model_origin)
    module_name = '.'.join(model_origin[:-1])
    class_name = model_origin[-1]
    print("Module name: {}, Class name: {}".format(module_name, class_name))
    print("-"*50)
    module = __import__(module_name, fromlist=[class_name])
    Net = getattr(module, class_name)

    print("Is the above parameters correct? (y/n)")
    if input() == 'y':
        prediction = make_prediction(args.entry, args.model)
        # print(prediction)
        print_prediction(prediction)
    else:
        print("Exiting...")

# Example usage:
# python model/utils/inference.py --entry "Y2I6" --model "./model/trained_model.pth"