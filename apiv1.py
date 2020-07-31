import math

import numpy as np
import pandas as pd
import altair as alt

import dataset_utils


class LossDataEstimator:
    def __init__(self, algorithm, dataset, representation_fn=lambda x: x,
                 val_frac=0.1, n_seeds=5):
        self.algorithm = algorithm
        self.dataset = dataset
        self.val_frac = val_frac
        self.n_seeds = n_seeds

        # TODO: check the type of the dataset and make this work for
        # PyTorch Dataset or tensors

        # TODO: apply the representation function to each dataset and cache

        self.val_size = math.ceil(len(self.dataset) * self.val_frac)
        self.max_train_size = len(self.dataset) - self.val_size
        self.train_set = dataset_utils.DatasetSubset(
            self.dataset, stop=self.max_train_size)
        self.val_set = dataset_utils.DatasetSubset(
            self.dataset, start=self.max_train_size)

        self.train_set = dataset_utils.DatasetCache(self.train_set)
        self.val_set = dataset_utils.DatasetCache(self.val_set)

        self.results = pd.DataFrame(
            columns=["seed", "samples", "val_loss"])

    def compute_curve(self, n_points=10, sampling_type='log', points=None):
        """Computes the loss-data curve for the given algorithm and dataset.
        Arguments:
        - n_points: (int) the number of points at which the loss will be
            computed to estimate the curve
        - sampling_type: (str) how to distribute the n_points between 0 and
            len(dataset). valid options are 'log' (np.logspace) or 'linear'
            (np.linspace).
        - points: (list of ints) manually specify the exact points at which to
            estimate the loss.
        Returns: nothing.
        Effects: This LossDataEstimator instance will record the results of the
            experiments which are run, including them in the results dataframe
            and using them to compute representation quality measures.
        """
        if points is None:
            if sampling_type == 'log':
                points = np.logspace(1, np.log10(self.max_train_size), n_points)
            elif sampling_type == 'linear':
                points = np.logspace(10, self.max_train_size, n_points)
            else:
                raise ValueError((f"Argument sampling_type should be "
                                  f"'log' or 'linear', was {sampling_type}."))

        for point in points:
            for seed in range(self.n_seeds):
                shuffled_data = dataset_utils.DatasetShuffle(self.train_set)
                data_subset = dataset_utils.DatasetSubset(
                    shuffled_data, stop=int(point))
                predictor, evaluate = self.algorithm(data_subset, seed)
                val_loss = evaluate(predictor, self.val_set)
                self.results = self.results.append({
                    'seed': seed,
                    'samples': point,
                    'val_loss': val_loss,
                }, ignore_index=True)
                print(self.results)

        # TODO: parallelism, especially with JAX

    def _bound_esc(self, epsilon):
        """Finds an upper and lower bound for epsilon sample complexity.
        Looks through the results obtained so far.
        Finds the minimum n where loss is less than epsilon and the maximum n
        where loss is greater than epsilon.
        """
        r = self.results
        upper_bound = r[r['val_loss'] <= epsilon]['samples'].min()
        lower_bound = r[r['val_loss'] > epsilon]['samples'].max()
        if np.isnan(upper_bound):
            upper_bound = None
            lower_bound = r['samples'].max()
        elif np.isnan(lower_bound):
            lower_bound = None
            upper_bound = r['samples'].min()
        return (lower_bound, upper_bound)

    def refine_esc(self, epsilon, precision, parallelism=10):
        """Runs additional experiments to refine our estimate of the epsilon
        sample complexity. Performs experiments until the gap between an upper
        and lower bound is at most `precision`. This method is implemented as an
        iterative grid search.
        Arguments:
        - epsilon: (num) the tolerance specifying the maximum acceptable loss
            from running algorithm on dataset.
        - precision: (num) how tightly to bound eSC, in terms of
            upper_bound - lower_bound
        - parallelism: (int) the number of experiments to run in each round of
            grid search.
        """
        lower_bound, upper_bound = self._bound_esc(epsilon)
        while upper_bound - lower_bound > precision:
            points = np.linspace(lower_bound, upper_bound, parallelism)[1:-1]
            self.compute_curve(points=points)
            lower_bound, upper_bound = self._bound_esc(epsilon)
        return upper_bound

    def to_dataframe(self):
        return self.results


def render_curve(dataframe, save_path=None):
    # TODO: render an Altair plot of the dataframe. by default:
    #   - in a Jupyter environment, we should return something Jupyter displays
    #   - in a script, we should require a save path
    raise NotImplementedError


def render_table(dataframe):
    # TODO: return a literal string for a LaTeX table
    raise NotImplementedError
