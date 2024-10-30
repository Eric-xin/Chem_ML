# Chem_ML: Machine Learning for Chemistry

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains machine learning algorithms and tools for chemistry. It is inspired by “A general-purpose machine learning framework for predicting” by Ward et al. [1]. The goal of this project is to provide a general-purpose machine learning framework for predicting materials properties.

## Introduction
It is often hard to embed the chemical structure of a molecule into a machine learning model. Inspired by the work of Ward et al. [1], I use the stiochiometric attributes, elemental attrubutes, electronic structures attributes and ionic attributes to describe the molecule and embed them into a machine learning model. The model is trained on the Quantum Machine Database (QMDB) and is able to predict the properties of a inorganic molecule.

Basically, the model takes the chemical formula of a molecule as input and outputs the properties of the molecule. The properties of the molecule are the band gap, formation energy, and the stability of the molecule. Before training the model, the chemical formula is converted into a set of attributes that describe the molecule. The attributes are then fed into the machine learning model to train the model.

**Below is a schematic of the model:**

![Model](./assets/img/model.png)

> 1. Stoichiometric attributes that depend only on the fractions of elements present and not what those elements actually are. These include the number of elements present in the compound and several Lp norms of the fractions.  
> 2. Elemental property statistics, which are defined as the mean, mean absolute deviation, range, minimum, maximum and mode of 22 different elemental properties. This category includes attributes such as the maximum row on periodic table, average atomic number and the range of atomic radii between all elements present in the material.  
> 3. Electronic structure attributes, which are the average fraction of electrons from the s, p, d and f valence shells between all present elements. These are identical to the attributes used by Meredig et al.
> 4. Ionic compound attributes that include whether it is possible to form an ionic compound assuming all elements are present in a single oxidation state, and two adaptations of the fractional ‘ionic character’ of a compound based on an electronegativitybased measure.

<p align="right">--- Ward et al. [1]</p>

## Datasets
The datasets used in this project are from the Quantum Machine Database (QMDB). The QMDB is a collection of quantum mechanical data for molecules and materials. The `data/datasets/small-data.txt` is a small dataset used for testing the model. The full dataset is not included in this repository due to its size. The full dataset can be downloaded from the [QMDB website](http://quantum-machine.org/datasets/).

## Code Structure

The code is organized as follows:

- `data/`: Contains the data used in the project.
  - `data/datasets/`: Contains the datasets used in the project.
  - `data/properties/`: General chemistry properties.
  - `data/utils/`: Utility functions accessing chemical properties
- `generators/`: Generates attributes for chemicals based on their chemical structure.
- `models/`: Contains the machine learning models used in the project.
  - `models/trained_model.pth`: Pre-trained model.
- `tmp/`: Temporary files.
- `QMDB/`: Contains the Quantum Machine Database and related tools.
- `Chem_ML.ipynb`: Main notebook for the project.

## Installation
```bash
git clone https://github.com/eric-xin/Chem_ML.git
cd Chem_ML
pip install -r requirements.txt
```

## Model Structure
*This is the structure of the currently used residue model*

![Model Structure](./assets/img/model_structure.png)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## To-Do
- [ ] Implement the model on Google Colab to use more memory.
- [ ] Add the Ionic Compound Attributes to the training features. *
- [ ] Increase the amount of data used in training.

\* Note: The Ionic Compound Attributes are implemented in the code but are not used in the training features, because the ionization energy data is missing for some elements at certain oxidation states. This causes the model to crash when training (NaN values). Currently thinking of using the average ionization energy of the elements to fill in the missing values.

## Contact
If you have any questions, feel free to contact me at [me@ericxin.eu](mailto:me@ericxin.eu)

## Citations
[1] L. Ward, “A general-purpose machine learning framework for predicting,” npj Computational Materials, 2016, doi: 10.1038/npjcompumats.2016.28.

## Acknowledgement
Some of the codes of this repo is from the [Magpie_python](https://github.com/ramv2/magpie_python). Their code is originally wrote in Python 2 and some of the functions are deprecated. I have updated the code to Python 3 and fixed some of the deprecated functions.