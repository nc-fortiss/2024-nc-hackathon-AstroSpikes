import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split


class CreateTrainTestSplit:
    def __init__(self, cfg: DictConfig):
        self.data_root = str(os.path.join(cfg.root.dataset, cfg.data.source, cfg.data.transformation))
        self.label_root = str(os.path.join(cfg.root.dataset, cfg.data.source, 'labels'))
        self.all_files = os.listdir(self.label_root)
        self.output_dir = cfg.root.data_out

    def create_df(self):
        df_list = [pd.read_csv(os.path.join(self.label_root, label_file)) for label_file in self.all_files]
        df = pd.concat(df_list, ignore_index=True)
        df['filepath'] = df['filename'].apply(
            lambda x: os.path.join(self.data_root, x.split('_')[1].replace('.png', ''), x))
        df.sample(frac=1).reset_index(drop=True)
        data = {
            "filepath": df['filepath'].values,
            "Tx": df['Tx'].values.astype(np.float32),
            "Ty": df['Ty'].values.astype(np.float32),
            "Tz": df['Tz'].values.astype(np.float32),
            "Qx": df['Qx'].values.astype(np.float32),
            "Qy": df['Qy'].values.astype(np.float32),
            "Qz": df['Qz'].values.astype(np.float32),
            "Qw": df['Qw'].values.astype(np.float32),
        }
        return data

    def __call__(self):
        """
        Creates the combined DataFrame, splits it into train and test sets, and returns the results.
        """
        all_data = self.create_df()

        # Convert the dictionary to a DataFrame for easier splitting
        df = pd.DataFrame(all_data)

        # Perform the train-test split on the DataFrame.  Specify random_state for reproducibility.
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Convert DataFrames back to dictionaries, if necessary
        train_data = train_df.to_dict('list')
        test_data = test_df.to_dict('list')

        # Save to CSV files
        train_csv_path = os.path.join(self.output_dir, "train.csv")
        test_csv_path = os.path.join(self.output_dir, "val.csv")

        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        return train_data, test_data


if __name__ == '__main__':

    config_path = "configs/mobilenet.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    data_creation = CreateTrainTestSplit(cfg)
    data_creation()
