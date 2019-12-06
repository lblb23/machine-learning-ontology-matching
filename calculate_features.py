import pandas as pd
import yaml

from utils_features import calculate_features

with open('config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SELECTED_DATASET = config['DATASET']

# Calculate features for training dataset
data = pd.read_csv(SELECTED_DATASET + '_train.csv')

data = calculate_features(data, 'Entity')
data = calculate_features(data, 'Parent')
data = calculate_features(data, 'Path')

data.to_csv(SELECTED_DATASET + '_train_features.csv', index=False)

# Calculate features for testing dataset
data = pd.read_csv(SELECTED_DATASET + '_test.csv')

data = calculate_features(data, 'Entity')
data = calculate_features(data, 'Parent')
data = calculate_features(data, 'Path')

data.to_csv(SELECTED_DATASET + '_test_features.csv', index=False)
