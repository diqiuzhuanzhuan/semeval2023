import pandas as pd
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.insert(0, PROJECT_ROOT)

from task4.configuration import config

def train_val_test_split():
    if not os.path.exists(config.validate_file['arguments']):
        arguments_val = config.validate_file['arguments']
        labels_val = config.validate_file['labels']
        level1_labels_val = config.validate_file['level1-labels']
        for file in [arguments_val, labels_val, level1_labels_val]:
            os.makedirs(os.path.dirname(file), exist_ok=True)
        
        arguments_df = pd.read_csv(config.train_file['arguments'], sep='\t')
        labels_df = pd.read_csv(config.train_file['labels'], sep='\t')
        level1_labels_df = pd.read_csv(config.train_file['level1-labels'], sep='\t')

        ID_KEY = "Argument ID"
        train_argument_ids = arguments_df.loc[:, ID_KEY].sample(int(len(arguments_df) * 0.8), random_state=42)

        arguments_df.loc[arguments_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.train_file['arguments'], index=False, sep='\t'
        )
        labels_df.loc[labels_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.train_file['labels'], index=False, sep='\t'
        )
        level1_labels_df.loc[level1_labels_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.train_file['level1-labels'], index=False, sep='\t'
        )

        arguments_df.loc[~arguments_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.validate_file['arguments'], index=False, sep='\t'
        )
        labels_df.loc[~labels_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.validate_file['labels'], index=False, sep='\t'
        )
        level1_labels_df.loc[~level1_labels_df.loc[:, ID_KEY].isin(train_argument_ids), :].to_csv(
            config.validate_file['level1-labels'], index=False, sep='\t'
        )

if __name__ == "__main__":
    train_val_test_split()
