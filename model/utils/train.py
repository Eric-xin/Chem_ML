import matplotlib.pyplot as plt
# from model.simple_model_residue import Net
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# import generator
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from generator.charge import ChargeDependentAttributeGenerator
# from generator.element import ElementalPropertyAttributeGenerator
# from generator.ionic import IonicityAttributeGenerator
# from generator.stoichiometric import StoichiometricAttributeGenerator
# from data.utils.composition import CompositionEntry
from generator import ChargeDependentAttributeGenerator, ElementalPropertyAttributeGenerator, IonicityAttributeGenerator, StoichiometricAttributeGenerator
from data.utils import CompositionEntry

import argparse
from tqdm import tqdm

def read_data(file_path = 'data/datasets/small_set.txt'):
    column_names = [
        'name', 'bandgap', 'energy_pa', 'volume_pa', 'magmom_pa', 
        'fermi', 'hull_distance', 'delta_e'
    ]
    data = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=1)
    data.replace('None', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='ignore')
    return data

def train(Net, input_path='data/datasets/small_set.txt', save_temp=True, temp_path='tmp/', model_path='model/', num_epochs=1000, random_state=42, test_size=0.2, learning_rate=0.001):

    # Generator setup
    charge_generator = ChargeDependentAttributeGenerator()
    elemental_generator = ElementalPropertyAttributeGenerator()
    ionicity_generator = IonicityAttributeGenerator()
    stoichiometric_generator = StoichiometricAttributeGenerator()

    data = read_data(input_path)
    print("data preview:")
    print("data shape: ", data.shape)
    print(data.head())
    print("-"*50)

    features = []
    for i in range(len(data)):
        entry = CompositionEntry(data['name'][i])
        stoichiometric = np.array(stoichiometric_generator.generate_features([entry])).flatten()
        ionicity = np.array(ionicity_generator.generate_features([entry])).flatten()
        elemental = np.array(elemental_generator.generate_features([entry])).flatten()
        feature = np.concatenate([stoichiometric, ionicity, elemental])
        features.append(feature)

    features = np.array(features)
    current_dir = os.getcwd()

    if save_temp:
        if not os.path.exists(temp_path):
            raise FileNotFoundError("The path to save temporary files does not exist.")
    
    if save_temp:
        os.makedirs(temp_path, exist_ok=True)
        np.savetxt(os.path.join(temp_path, "features.csv"), features, delimiter=",")
        target = data[['bandgap', 'energy_pa', 'volume_pa', 'magmom_pa', 'fermi', 'delta_e']].values
        np.savetxt(os.path.join(temp_path, "target.csv"), target, delimiter=",")
        features = pd.read_csv(os.path.join(temp_path, "features.csv"), header=None)
        targets = pd.read_csv(os.path.join(temp_path, "target.csv"), header=None)
    else:
        target = data[['bandgap', 'energy_pa', 'volume_pa', 'magmom_pa', 'fermi', 'delta_e']].values
        features = pd.DataFrame(features)
        targets = pd.DataFrame(target)

    features = torch.tensor(features.values, dtype=torch.float32)
    targets = torch.tensor(targets.values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=random_state)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_train[0])
    print(y_train[0])

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print("input dim {}, output dim {}".format(input_dim, output_dim))
    print("-"*50)
    model = Net(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = criterion(predictions, y_test).item()
        print(f'Mean Squared Error: {mse}')

    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, 'trained_model.pth'))

    if save_temp:
        # os.makedirs(temp_path, exist_ok=True)
        np.savetxt(os.path.join(temp_path, "losses.csv"), np.array(losses), delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the ChemML network model.')
    parser.add_argument('--model_import_path', type=str, default='model.simple_model_residue.Net', help='Path to the model class')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--model_path', type=str, default='./model', help='Path to save the trained model')
    parser.add_argument('--save_temp', action='store_true', help='Flag to save temporary files')
    parser.add_argument('--temp_path', type=str, default='./temp', help='Path to save temporary files if save_temp is set')

    print("-"*50)
    print("Arguments:")
    for arg in vars(parser.parse_args()):
        print(f"\t{arg}: {getattr(parser.parse_args(), arg)}")
    print("-"*50)

    args = parser.parse_args()
    model_origin = args.model_import_path.split('.')
    # print(model_origin)
    module_name = '.'.join(model_origin[:-1])
    class_name = model_origin[-1]
    print("Module name: {}, Class name: {}".format(module_name, class_name))
    print("-"*50)
    module = __import__(module_name, fromlist=[class_name])
    Net = getattr(module, class_name)

    # Continue or not
    print("Continue training? (y/n)")
    continue_training = input()
    if continue_training.lower() == 'y':
        train(Net=Net, learning_rate=args.learning_rate, num_epochs=args.num_epochs, model_path=args.model_path, save_temp=args.save_temp, temp_path=args.temp_path)
    else:
        print("Training stopped.")

    # train(Net=Net, learning_rate=args.learning_rate, num_epochs=args.num_epochs, model_path=args.model_path, save_temp=args.save_temp, temp_path=args.temp_path)

# Example usage:
# python train.py --model_import_path model.simple_model_residue.Net --learning_rate 0.001 --num_epochs 1000 --model_path model/ --save_temp --temp_path tmp/