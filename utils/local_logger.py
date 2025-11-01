from pathlib import Path
from typing import Iterable, Tuple
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
            new_row = pd.DataFrame([{'step': step}])
            self._df = pd.concat([self._df, new_row], ignore_index=True)

        if key not in self._df.columns:
            self._df[key] = pd.NA

        self._df.loc[self._df['step'] == step, key] = value
        self.save_run

    def save_run(self, keys : Iterable[Tuple[Iterable[Tuple[str, str]], str]]) -> None:
        """Will save the logged data to the path as a csv and will save a new graph where the y value corresponds to the key

        Args:
            keys (Iterable[(Iterable[[str], str)]): A key corresponds to y value in the graph. 
            Each tuple contains an iterable of keys and colors which represents one graph. 
            ([(graph_1_k, color_k), (graph_1_w, color_w)], [(graph_2_r, color_r), (graph_2_t, color_t)])
            Other element is the ylabel of the graph

        """
        os.makedirs(self.save_path, exist_ok=True)
        self._df.to_csv(self.save_path / 'data.csv')

        for key_l, y_label in keys:
            plt.figure()
            new_key_l = []
            for key, color in key_l:
                new_key_l.append(key)
                key = key.replace(' ', '_')
                mask = ~pd.isnull(self._df[key])
                plt.plot(self._df['step'][mask], self._df[key][mask], label=key, color=color)
                plt.xlabel('Steps')
                plt.ylabel(y_label)
            name = '_'.join(new_key_l).replace(' ', '_')
            plt.legend(loc='lower left')
            plt.savefig(self.save_path / f'{name}_graph.png')
