import math
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from . import utils
from . import dataset_wrappers
from . import metrics


class LossDataEstimator:
    def __init__(self, init_fn, train_step_fn, eval_fn, dataset,
                 representation_fn=lambda x: x,
                 val_frac=0.1, n_seeds=5,
                 train_steps=4e3, batch_size=256,
                 cache_data=True, whiten=True,
                 use_vmap=True, verbose=False):
        """Create a LossDataEstimator.
        Arguments:
        - init_fn: (function int -> object)
            a function which maps from an integer random seed to an initial
            state for the training algorithm. this initial state will be fed to
            train_step_fn, and the output of train_step_fn will replace it at
            each step.
        - train_step_fn: (function (object, (ndarray, ndarray)) -> (object, num)
            a function which performs one step of training. in particular,
            should map (state, batch) -> (new_state, loss) where state is
            defined recursively, initialized by init_fn and replaced by
            train_step, and loss is a Python number.
        - eval_fn: (function (object, (ndarray, ndarray)) -> float)
            a function which takes in a state as produced by init_fn or
            train_step_fn, plus a batch of data, and returns the _mean_ loss
            over points in that batch. should not mutate anything.
        - dataset: a PyTorch Dataset or tuple (data_x, data_y).
        - representation_fn: (function ndarray -> ndarray)
            a function which takes in a batch of observations from the dataset,
            given as a numpy array, and gives back an ndarray of transformed
            observations.
        - val_frac: (float) the fraction of the data in [0, 1] to use for
            validation
        - n_seeds: (int) how many random seeds to use for estimating each point.
            the seed is used for randomly sampling a subset dataset and for
            initializing the algorithm.
        - train_steps: (number) how many batches of training to use with the
            algorithm. that is, how many times train_step_fn will be called on
            a batch of data.
        - batch_size: (int) the size of the batches used for training and eval
        - cache_data: (bool) whether to cache the entire dataset in memory.
            setting this to True will greatly improve performance by only
            computing the representation once for each point in the dataset
        - whiten: (bool) whether to normalize the dataset's Xs to have zero
            mean and unit variance
        - use_vmap: (bool) *only for JAX algorithms*. parallelize the training
            of <algorithm> by using JAX's vmap function. may cause CUDA out of
            memory errors; if this happens, call compute_curve with fewer
            points at a time or use a smaller probe
        - verbose: (bool) print out informative messages and results as we get
            them
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
        self.verbose = verbose
        if self.verbose:
            self.print = print
        else:
            self.print = utils.no_op

        if self.use_vmap and not cache_data:
            raise ValueError(("Setting use_vmap requires cache_data. "
                              "Either set cache_data=True or "
                              "turn off use_vmap."))

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        if not isinstance(self.dataset, Dataset):
            data_x, data_y = self.dataset
            data_x = torch.as_tensor(data_x)
            data_y = torch.as_tensor(data_y)
            self.dataset = torch.utils.data.TensorDataset(data_x, data_y)

        # Step 1: split into train and val
        self.val_size = math.ceil(len(self.dataset) * self.val_frac)
        self.max_train_size = len(self.dataset) - self.val_size
        self.train_set = dataset_wrappers.DatasetSubset(
            self.dataset, stop=self.max_train_size)
        self.val_set = dataset_wrappers.DatasetSubset(
            self.dataset, start=self.max_train_size)

        # Step 2: figure out when / if we're caching the data
        if use_vmap:
            self.print("Transforming and caching dataset.")
            # transform the whole training and put it in JAX
            self.train_set = utils.dataset_to_jax(
                self.train_set,
                batch_transforms=[self.representation_fn],
                batch_size=batch_size)
            self.val_set = utils.dataset_to_jax(
                self.val_set,
                batch_transforms=[self.representation_fn],
                batch_size=batch_size)
            # we've already used representation_fn to transform the data
            self.batch_transforms = []
        elif cache_data:
            self.print("Transforming and caching dataset.")
            # transform the data and cache it as a Pytorch dataset
            self.train_set = dataset_wrappers.DatasetTransformCache(
                self.train_set,
                batch_transforms=[self.representation_fn],
                batch_size=self.batch_size)
            self.val_set = dataset_wrappers.DatasetTransformCache(
                self.val_set,
                batch_transforms=[self.representation_fn],
                batch_size=self.batch_size)
            # we've already used representation_fn to transform
            self.batch_transforms = []
        else:
            # don't transform or cache the data yet
            # instead add representation_fn and transform one batch at a time
            self.batch_transforms = [self.representation_fn]

        # Step 3: whiten transformed data
        if whiten:
            # streams one batch at a time through batch_transforms
            mean, std = utils.compute_stats(
                self.train_set, self.batch_transforms, self.batch_size)
            self.print((f"Whitening with representation "
                        f"(mean, std): ({mean :.4f}, {std :.4f})"))
            whiten_transform = utils.make_whiten_transform(mean, std)
            self.batch_transforms.append(whiten_transform)

        self.results = pd.DataFrame(
            columns=["seed", "samples", "val_loss"])
        self.results['samples'] = self.results['samples'].astype(int)

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
        Returns: the current DataFrame containing the loss-data curve.
        Effects: This LossDataEstimator instance will record the results of the
            experiments which are run, including them in the results DataFrame
            and using them to compute representation quality measures.
        """
        if points is None:
            if sampling_type == 'log':
                points = np.logspace(1, np.log10(self.max_train_size), n_points)
            elif sampling_type == 'linear':
                points = np.linspace(10, self.max_train_size, n_points)
            else:
                raise ValueError((f"Argument sampling_type should be "
                                  f"'log' or 'linear', was {sampling_type}."))
        points = np.ceil(points)

        if self.use_vmap:
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
            number of training points required to reach loss `epsilon`. that
            is, the desired `upper_bound - lower_bound`
        - parallelism: (int) the number of experiments to run in each round of
            grid search.
        Returns: an upper bound on the epsilon sample complexity
        Effects: runs compute_curve multiple times and adds points to the
            loss-data curve
        """
        if len(self.results) == 0:
            self.compute_curve(n_points=parallelism)
        lower_bound, upper_bound = self._bound_esc(epsilon)
        if upper_bound is None:
            if self.results['samples'].max() < self.max_train_size:
                self.compute_curve(n_points=parallelism)
            lower_bound, upper_bound = self._bound_esc(epsilon)
        if upper_bound is None:
            return None
        while upper_bound - lower_bound > precision:
            points = np.linspace(lower_bound, upper_bound, parallelism+2)[1:-1]
            self.compute_curve(points=points)
            lower_bound, upper_bound = self._bound_esc(epsilon)
        return upper_bound

    def to_dataframe(self):
        """Return the current data for estimating the loss-data curve."""
        return self.results.copy()

    # Private methods

    def _compute_curve_sequential(self, points):
        for point in points:
            for seed in range(self.n_seeds):
                shuffled_data = dataset_wrappers.DatasetShuffle(self.train_set)
                data_subset = dataset_wrappers.DatasetSubset(shuffled_data,
                                                             stop=int(point))
                state = self._train(seed, data_subset)
                val_loss = self._eval(state, self.val_set)
                self.results = self.results.append({
                    'seed': seed,
                    'samples': point,
                    'val_loss': float(val_loss,)
                }, ignore_index=True)

                self.print(self.results)
        return self.to_dataframe()

    def _compute_curve_full_vmap(self, points):
        seeds = list(range(self.n_seeds))
        jobs = [(point, seed) for point in points for seed in seeds]
        product_points = [j[0] for j in jobs]
        product_seeds = [j[1] for j in jobs]

        multi_iterator = utils.jax_multi_iterator(
            self.train_set, self.batch_size, product_seeds, product_points)

        states = self._train_full_vmap(multi_iterator, product_seeds)
        val_losses = self._eval_vmap(states, self.val_set)
        for (job, val_loss) in zip(jobs, val_losses):
            self.results = self.results.append({
                'seed': job[1],
                'samples': job[0],
                'val_loss': float(val_loss),
            }, ignore_index=True)

        self.print(self.results)
        return self.to_dataframe()

    def _train_full_vmap(self, multi_iterator, seeds):
        import jax
        import jax.numpy as jnp
        vmap_train_step = jax.vmap(self.train_step_fn)
        states = jax.vmap(self.init_fn)(jnp.array(seeds))

        for step in range(self.train_steps):
            stacked_xs, stacked_ys = next(multi_iterator)
            stacked_xs = utils.apply_transforms(
                self.batch_transforms, stacked_xs)
            states, losses = vmap_train_step(states, (stacked_xs, stacked_ys))
        return states

    def _make_loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train(self, seed, dataset):
        """Performs training with the algorithm associated with this LDE.
        Runs `self.train_steps` batches' worth of updates to the model and
        returns the state.
        """
        torch.manual_seed(seed)
        loader = self._make_loader(dataset, shuffle=True)
        state = self.init_fn(seed)
        step = 0
        while step < self.train_steps:
            for batch in loader:
                xs, ys = utils.batch_to_numpy(batch)
                xs = utils.apply_transforms(
                    self.batch_transforms, xs)
                state, loss = self.train_step_fn(state, (xs, ys))
                step += 1
                if step >= self.train_steps:
                    break
        return state

    def _eval(self, state, dataset):
        """Evaluates the model specified by state on dataset.
        Computes the average loss by summing the total loss over all datapoints
        and dividing.
        """
        loss, examples = 0, 0
        loader = self._make_loader(dataset, shuffle=False)
        for batch in loader:
            # careful to deal with different-sized batches
            xs, ys = utils.batch_to_numpy(batch)
            xs = utils.apply_transforms(
                self.batch_transforms, xs)
            batch_examples = xs.shape[0]
            loss += self.eval_fn(state, (xs, ys)) * batch_examples
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
        for i in range(0, self.val_size, self.batch_size):
            xs = self.val_set[0][i: i + self.batch_size]
            ys = self.val_set[1][i: i + self.batch_size]

            # careful to deal with different-sized batches
            xs = utils.apply_transforms(
                self.batch_transforms, xs)
            batch_examples = xs.shape[0]

            if losses is None:
                losses = vmap_eval(states, (xs, ys)) * batch_examples
            else:
                losses += vmap_eval(states, (xs, ys)) * batch_examples
            examples += batch_examples
        return losses / examples

    def _bound_esc(self, epsilon):
        """Finds an upper and lower bound for epsilon sample complexity.
        Looks through the results obtained so far.
        Finds the minimum n where loss is less than epsilon and the maximum n
        where loss is greater than epsilon.
        """
        r = self.results
        r = r.groupby(["samples"]).mean().reset_index()
        upper_bound = r[r['val_loss'] <= epsilon]['samples'].min()
        lower_bound = r[r['val_loss'] > epsilon]['samples'].max()
        if np.isnan(upper_bound):
            upper_bound = None
            lower_bound = r['samples'].max()
        elif np.isnan(lower_bound):
            lower_bound = 0
            upper_bound = r['samples'].min()
        return (lower_bound, upper_bound)


def render_curve(df, ns=[], epsilons=[], save_path=None):
    """Render, and optionally save, a plot of the loss-data curve.
    Optionally takes arguments `ns` and `epsilons` to draw lines on the plot
    illustrating where metrics were calculated.
    Arguments:
    - df: (pd.DataFrame) the dataframe containing a loss-data curve as returned
        by LossDataEstimator.compute_curve or LossDataEstimator.to_dataframe.
    - ns: (list<num>) the list of training set sizes to use for computing
        metrics.
    - epsilons: (list<num>) the settings of epsilon used for computing SDL and
        eSC.
    - save_path: (str) optional: a path (ending in .pdf or .png) to save the
        chart. saving requires the
        [`altair-saver`](https://github.com/altair-viz/altair_saver/) package
        and its dependencies.
    Returns: an Altair chart. Note that this chart displays well in notebooks,
        so calling `render_curve(df)` without a save path will work well with
        Jupyter.
    """
    import altair as alt
    from . import altair_theme  # noqa: F401
    alt.data_transformers.disable_max_rows()

    if "name" not in df:
        print("Dataframe has no 'name' field. Using 'default'.")
        df['name'] = 'default'

    if len(ns) > 0:
        ns = _closest_valid_ns(df, ns)

    title = 'Loss-data curve'
    color_title = 'Representation'
    xscale = alt.Scale(type='log')
    yscale = alt.Scale(type='log')

    x_axis = alt.X('samples', scale=xscale, title='Dataset size')
    y_axis = alt.Y('mean(val_loss)', scale=yscale, title='Validation loss')

    line = alt.Chart(df, title=title).mark_line()
    line = line.encode(
        x=x_axis, y=y_axis,
        color=alt.Color('name:N', title=color_title, legend=None),
    )

    point = alt.Chart(df, title=title).mark_point(size=80, opacity=1)
    point = point.encode(
        x=x_axis, y=y_axis,
        color=alt.Color('name:N', title=color_title,),
        shape=alt.Shape('name:N', title=color_title),
        tooltip=['samples', 'name']
    )

    rules_df = pd.concat([
        pd.DataFrame({'x': ns}),
        pd.DataFrame({'y': epsilons})
    ], sort=False)

    rule_x = alt.Chart(rules_df).mark_rule(strokeDash=[4, 4]).encode(x='x')
    rule_y = alt.Chart(rules_df).mark_rule(strokeDash=[4, 4]).encode(y='y')

    chart = alt.layer(rule_x, rule_y, line, point).resolve_scale(
        color='independent',
        shape='independent'
    )
    if save_path is not None:
        import altair_saver
        altair_saver.save(chart, save_path)
    return chart


def compute_metrics(df, ns=None, epsilons=[1.0, 0.1, 0.01]):
    """Compute val loss, MDL, SDL, and eSC at the specified `ns` and `epsilons`.

    Arguments:
    - df: (pd.DataFrame) the dataframe containing a loss-data curve as returned
        by LossDataEstimator.compute_curve or LossDataEstimator.to_dataframe.
    - ns: (list<num>) the list of training set sizes to use for computing
        metrics. this will be rounded up to the nearest point where the loss
        has been computed. set this to [len(dataset)] to compute canonical
        results.
    - epsilons: (list<num>) the settings of epsilon used for computing SDL and
        eSC.
    """
    if "name" not in df:
        print("Dataframe has no 'name' field. Using 'default'.")
        df['name'] = 'default'
    df = df.groupby(['name', 'samples']).mean().reset_index()
    if ns is None:
        closest_ns = [max(df.samples)]
    else:
        closest_ns = _closest_valid_ns(df, ns)
    return metrics.compute_all(df, closest_ns, epsilons)


def render_latex(metrics_df, display=False, save_path=None):
    """Given a df of metrics from `compute_metrics`, renders a LaTeX table.

    Arguments:
    - metrics_df: (pd.DataFrame) a dataframe as returned by `compute_metrics`
    - display: (bool) *Jupyter only.* render an output widget containing the
        latex string. necessary because otherwise lots of things will be
        double-escaped.
    - save_path: (str) if specified, saves the text for the LaTeX table in a
        file.
    """
    metrics_df.index = metrics_df.index.str.replace(
        'eps', '$\\\\varepsilon$')
    metrics_df.index = metrics_df.index.str.replace(
        'eSC', '$\\\\varepsilon$SC')
    metrics_df = metrics_df.stack()
    metrics_df = metrics_df.swaplevel().sort_values('n', ascending=True)

    latex_str = metrics_df.to_latex(multicolumn_format='c',
                                    float_format="{:0.2f}".format,
                                    escape=False,
                                    column_format='llrrr')
    if save_path is not None:
        metrics_df.to_latex(save_path,
                            multicolumn_format='c',
                            float_format="{:0.2f}".format,
                            escape=False,
                            column_format='llrrr')
    if display:
        import ipywidgets as widgets
        import IPython.display
        out = widgets.Output(layout={'border': '1px solid black'})
        out.append_stdout(latex_str)
        IPython.display.display(out)


def _closest_valid_ns(df, ns):
    closest_ns = []
    available_ns = sorted(list(df.samples.unique()))
    last_match = 0
    for desired_n in sorted(ns):
        i = last_match
        while desired_n > available_ns[i] and i < len(available_ns) - 1:
            i += 1
        last_match = i
        closest_ns.append(available_ns[i])
    return closest_ns
