import math
import functools

import numpy as np
import pandas as pd
import altair as alt

from torch.utils.data import DataLoader

import utils
import dataset_utils


class LossDataEstimator:
    def __init__(self, init_fn, train_step_fn, eval_fn, dataset,
                 representation_fn=lambda x: x,
                 val_frac=0.1, n_seeds=5,
                 train_steps=5e3, batch_size=128,
                 cache_data=True, use_vmap=True):
        """Create a LossDataEstimator.
        Arguments:
        - init_fn: (function int -> object)
            a function which maps from an integer random seed to an initial
            state for the training algorithm. this initial state will be fed to
            train_step_fn, and the output of train_step_fn will replace it at
            each step.
        - train_step_fn: (function (object, (ndarray, ndarray)) -> object)
            a function which performs one step of training.
            in particular, should map (state, batch) -> new_state where batch is
            a tuple (batch of inputs, batch of targets). state is defined
            recursively, being initialized by init_fn and replaced by
            train_step.
        - eval_fn: (function (object, (ndarray, ndarray)) -> float)
            a function which takes in a state as produced by init_fn or
            train_step_fn, plus a batch of data, and returns the _mean_ loss
            over points in that batch. should not mutate anything.
        - dataset
        """
        self.init_fn = init_fn
        self.train_step_fn = train_step_fn
        self.eval_fn = eval_fn
        self.dataset = dataset
        self.representation_fn = representation_fn
        self.val_frac = val_frac
        self.n_seeds = n_seeds
        self.train_steps = int(train_steps)
        self.batch_size = batch_size
        self.use_vmap = use_vmap

        # TODO: check the type of the dataset and make this work for
        # PyTorch Dataset or tensors

        # TODO: apply the representation function to each dataset and cache

        self.val_size = math.ceil(len(self.dataset) * self.val_frac)
        self.max_train_size = len(self.dataset) - self.val_size
        self.train_set = dataset_utils.DatasetSubset(
            self.dataset, stop=self.max_train_size)
        self.val_set = dataset_utils.DatasetSubset(
            self.dataset, start=self.max_train_size)

        if cache_data:
            self.train_set = dataset_utils.DatasetCache(self.train_set)
            self.val_set = dataset_utils.DatasetCache(self.val_set)

        self.results = pd.DataFrame(
            columns=["seed", "samples", "val_loss"])

    @profile
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
                # TODO: change back to 1
                points = np.logspace(3, np.log10(self.max_train_size), n_points)
            elif sampling_type == 'linear':
                points = np.linspace(10, self.max_train_size, n_points)
            else:
                raise ValueError((f"Argument sampling_type should be "
                                  f"'log' or 'linear', was {sampling_type}."))

        if self.use_vmap:
            # return self._compute_curve_vmap(points)
            return self._compute_curve_full_vmap(points)
        else:
            return self._compute_curve_sequential(points)

    def refine_esc(self, epsilon, precision, parallelism=10):
        """Runs experiments to refine an estimate of epsilon sample complexity.
        Performs experiments until the gap between an upper and lower bound is
        at most `precision`. This method is implemented as an iterative grid
        search.
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

    def _compute_curve_sequential(self, points):
        for point in points:
            for seed in range(self.n_seeds):
                shuffled_data = dataset_utils.DatasetShuffle(self.train_set)
                data_subset = dataset_utils.DatasetSubset(shuffled_data,
                                                          stop=int(point))
                state = self._train(seed, data_subset)
                val_loss = self._eval(state, self.val_set)
                self.results = self.results.append({
                    'seed': seed,
                    'samples': point,
                    'val_loss': val_loss,
                }, ignore_index=True)
                print(self.results)

    # @profile
    def _compute_curve_vmap(self, points):
        for point in points:
            shuffled_data = dataset_utils.DatasetShuffle(self.train_set)
            data_subset = dataset_utils.DatasetSubset(shuffled_data,
                                                      stop=int(point))
            seeds = list(range(self.n_seeds))
            states = self._train_vmap(seeds, data_subset)
            val_losses = self._eval_vmap(states, self.val_set)
            for (seed, val_loss) in zip(seeds, val_losses):
                self.results = self.results.append({
                    'seed': seed,
                    'samples': point,
                    'val_loss': val_loss,
                }, ignore_index=True)
            print(self.results)

    @profile
    def _compute_curve_full_vmap(self, points):
        seeds = list(range(self.n_seeds))
        jobs = [(point, seed) for point in points for seed in seeds]
        product_points = [j[0] for j in jobs]
        product_seeds = [j[1] for j in jobs]

        # loaders = []
        # for job in jobs:
        #     shuffled_dataset = dataset_utils.DatasetShuffle(self.train_set)
        #     subset_dataset = dataset_utils.DatasetSubset(shuffled_dataset,
        #                                                  stop=int(job[0]))
        #     loader = self._make_loader(subset_dataset, job[1])
        #     loaders.append(loader)

        # multi_iterator = utils.multi_loader_iterator(loaders)
        multi_iterator = utils.jax_multi_iterator(
            self.dataset, self.batch_size, product_seeds, product_points)

        # this is wrong, don't do it (only for profiling)
        # multi_iterator = utils.forever_iterator(loaders[0])
        states = self._train_full_vmap(multi_iterator, product_seeds)
        val_losses = self._eval_vmap(states, self.val_set)
        for (job, val_loss) in zip(jobs, val_losses):
            self.results = self.results.append({
                'seed': job[1],
                'samples': job[0],
                'val_loss': val_loss,
            }, ignore_index=True)
        print(self.results)

    @profile
    def _train_full_vmap(self, multi_iterator, seeds):
        import jax
        import jax.numpy as jnp
        vmap_train_step = jax.vmap(self.train_step_fn)
        states = jax.vmap(self.init_fn)(jnp.array(seeds))

        for step in range(self.train_steps):
            stacked_batch = next(multi_iterator)
            # stacked_batch = utils.batch_to_numpy(stacked_batch)
            states = vmap_train_step(states, stacked_batch)
        return states

    # @profile
    # def _compute_curve_full_vmap(self, points):
    #     seeds = list(range(self.n_seeds))
    #     jobs = [(point, seed) for point in points for seed in seeds]

    #     states = self._train_full_vmap(points, seeds)
    #     val_losses = self._eval_vmap(states, self.val_set)
    #     for (job, val_loss) in zip(jobs, val_losses):
    #         self.results = self.results.append({
    #             'seed': job[1],
    #             'samples': job[0],
    #             'val_loss': val_loss,
    #         }, ignore_index=True)
    #     print(self.results)


    # @profile
    # def _train_full_vmap(self, points, seeds):
    #     import jax
    #     import jax.numpy as jnp
    #     # n_jobs = len(points) * len(seeds)
    #     product_seeds = [seed for point in points for seed in seeds]
    #     vmap_train_step = jax.vmap(self.train_step_fn)
    #     states = jax.vmap(self.init_fn)(jnp.array(product_seeds))

    #     data_subsets = []
    #     for point in points:
    #         for seed in seeds:
    #             indices = np.arange(len(self.train_set))
    #             np.random.shuffle(indices)
    #             indices = indices[:int(point)]
    #             data_subsets.append(indices)

    #     data_sampler = utils.MultiSubsetSampler(data_subsets, self.batch_size)
    #     loader_iter = iter(DataLoader(self.train_set,
    #                                   batch_sampler=data_sampler))

    #     # loader = utils.MultiSubsetLoader(
    #     #     self.train_set, data_subsets, self.batch_size)
    #     # loader_iter = iter(loader)

    #     for step in range(self.train_steps):
    #         stacked_batch = next(loader_iter)
    #         # stacked_x = flat_x.reshape((n_jobs, -1, *flat_x.shape[2:]))
    #         # stacked_y = flat_y.reshape((n_jobs, -1, *flat_y.shape[2:]))
    #         stacked_batch = utils.batch_to_numpy(stacked_batch)
    #         states = vmap_train_step(states, stacked_batch)
    #     return states

    def _make_loader(self, dataset, seed):
        def _worker_init_fn(worker_id):
            import torch
            torch.manual_seed(seed * worker_id)
        return DataLoader(dataset,
                          batch_size=self.batch_size, shuffle=True,
                          worker_init_fn=_worker_init_fn,
                          num_workers=0, drop_last=self.use_vmap)
        # TODO: don't drop_last

    def _train(self, seed, dataset):
        """Performs training with the algorithm associated with this LDE.
        Runs `self.train_steps` batches' worth of updates to the model and
        returns the state.
        """
        loader = self._make_loader(dataset, seed)
        state = self.init_fn(seed)
        step = 0
        while step < self.train_steps:
            for batch in loader:
                batch = utils.batch_to_numpy(batch)
                state = self.train_step_fn(state, batch)
                step += 1
                if step >= self.train_steps:
                    break
        return state

    # @profile
    def _train_vmap(self, seeds, dataset):
        """Batches the train_step function across networks and trains models.
        To use this, train_step should be a pure JAX function.
        """
        import jax
        import jax.numpy as jnp
        # vmap_train_step = jax.vmap(self.train_step_fn)
        vmap_train_step = jax.vmap(self.train_step_fn, in_axes=(0, None))
        loader = self._make_loader(dataset, seeds[0])
        # jobs = [(seed, point) for seed in seeds for point in points]
        # states = [self.init_fn(seed) for (seed, point) in jobs]
        states = jax.vmap(self.init_fn)(jnp.array(seeds))
        # import ipdb; ipdb.set_trace()
        step = 0
        while step < self.train_steps:
            for batch in loader:
                # import ipdb; ipdb.set_trace()
                batch = utils.batch_to_numpy(batch)
                states = vmap_train_step(states, batch)
                step += 1
                if step >= self.train_steps:
                    break
        return states

    def _eval(self, state, dataset):
        """Evaluates the model specified by state on dataset.
        Computes the average loss by summing the total loss over all datapoints
        and dividing.
        """
        loss, examples = 0, 0
        loader = self._make_loader(dataset, seed=0)
        for batch in loader:
            # careful to deal with different-sized batches
            batch = utils.batch_to_numpy(batch)
            batch_examples = batch[0].shape[0]
            loss += self.eval_fn(state, batch) * batch_examples
            examples += batch_examples
        return loss / examples

    def _eval_vmap(self, states, dataset):
        """Evaluates
        """
        import jax
        vmap_eval = jax.vmap(self.eval_fn, in_axes=(0, None))

        # it can be hard to get the length of a state (if it's e.g. a Flax
        # Optimizer with nested parameters) so return however many losses we get
        losses = None
        examples = 0
        loader = self._make_loader(dataset, seed=0)
        for batch in loader:
            # careful to deal with different-sized batches
            batch = utils.batch_to_numpy(batch)
            batch_examples = batch[0].shape[0]

            if losses is None:
                losses = vmap_eval(states, batch) * batch_examples
            else:
                losses += vmap_eval(states, batch) * batch_examples
            examples += batch_examples
        return losses / examples

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



def render_curve(dataframe, save_path=None):
    # TODO: render an Altair plot of the dataframe. by default:
    #   - in a Jupyter environment, we should return something Jupyter displays
    #   - in a script, we should require a save path
    raise NotImplementedError


def render_table(dataframe):
    # TODO: return a literal string for a LaTeX table
    raise NotImplementedError
