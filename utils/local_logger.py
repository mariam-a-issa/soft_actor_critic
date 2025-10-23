from pathlib import Path
from typing import Iterable
import os

import pandas as pd
import matplotlib.pyplot as plt

class LocalLogger:

    def __init__(self, save_path : Path):
        self._df = pd.DataFrame()
        self._df['step'] = pd.NA
        self.save_path = save_path

    def save_value(self, step : int, key : str, value : float) -> None:
        """Will save the given value

        Args:
            step (int): Current Steps in the environment
            key (str): Description of value to be saved
            value (float): Value to be savec
        """
        key = key.replace(' ', '_')

        if step not in self._df['step'].values:
            self._df.loc[len(self._df)] = [step]

        if key not in self._df.columns:
            self._df[key] = pd.NA

        self._df.loc[self._df['step'] == step, key] = value

    def save_run(self, keys : Iterable[str]) -> None:
        """Will save the logged data to the path as a csv and will save a new graph where the y value corresponds to the key

        Args:
            keys (Iterable[str]): The key (y value) for each graph to save
        """
        os.makedirs(self.save_path, exist_ok=True)
        self._df.to_csv(self.save_path / 'data.csv')

        for key in keys:
            key = key.replace(' ', '_')
            plt.figure()
            plt.plot(self._df['step'], self._df[key])
            plt.xlabel('Steps')
            plt.ylabel(key)
            plt.savefig(self.save_path / f'{key}_graph.png')
