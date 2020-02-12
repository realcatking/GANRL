# GANRL
Source code for paper "A Unified Generative Adversarial Learning Framework for Improvement of Network Representation Learning Methods"

## Evironment Setting
- Python == 3.7.4

All required packages are defined in requirements.txt. To install all requirement, simply use the following commands:

	pip install -r requirements.txt

## Basic Usage
python main.py

## Parameter Setting (see file in config/ folder)
embedding\_size: Dimension of the embedding vector.  
n\_epochs: number of outer loops.  
learn\_rate\_gen, learn\_rate\_dis: The learning rate for generator and discriminator, respectively.  
n\_epochs\_gen, n\_epochs\_dis: umber of inner loops for generator and discriminator, respectively.  
batch\_size\_gen, batch\_size\_dis: batch size for generator and discriminator, respectively.  
n\_sample\_gen: number of samples for the generator.  
n\_sample\_dis: number of positive/negative samples for the discriminator.  

## Data
The [Cora](https://linqs.soe.ucsc.edu/data) dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. 


