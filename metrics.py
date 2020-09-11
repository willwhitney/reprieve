import numpy as np
import pandas as pd
from functools import partial


def compute_all(df, ns, epsilons):
    df = auc_per_data(df, epsilons + [0]).reset_index(drop=True)
    df = sc_per_data(df, epsilons).reset_index(drop=True)

    auc_cols = {}
    sc_cols = {}
    for eps in epsilons:
        auc_cols[f'str_auc_agg@{eps}'.replace('.', '_')] = \
            f'SDL, eps={eps}'
        sc_cols[f'str_sc@{eps}'.replace('.', '_')] = \
            f'eSC, eps={eps}'
    auc_cols['str_auc_agg@0'] = 'MDL'
    output_df = df[df.samples.isin(ns)].groupby(['name', 'samples', *auc_cols.keys(), *sc_cols.keys()]).mean().reset_index()
    output_df = output_df[['samples', 'name', 'val_loss', *auc_cols.keys(), *sc_cols.keys()]]
    output_df = output_df.sort_values('samples')

    output_df = output_df.rename(columns={'name': 'Name', 'samples': 'n', 'val_loss': 'Val loss', **auc_cols, **sc_cols})
    auc_cols.pop('str_auc_agg@0')
    output_df = output_df.reindex(['Name', 'n', 'Val loss', 'MDL', *auc_cols.values(), *sc_cols.values()], axis=1)
    output_df = output_df.set_index(['Name', 'n'])
    output_df = output_df.transpose()
    output_df.reindex(
        ['Val loss', 'MDL', *auc_cols.values(), *sc_cols.values()])

    # output_df = output_df.transpose()
    return output_df


def pd_reduce(dataframe, output_column, fn):
    result = dataframe.copy()
    result[output_column] = np.nan
    for i, index in enumerate(result.index):
        prev = None if i == 0 else result.iloc[i - 1]
        curr = result.loc[index]
        result.loc[index, output_column] = fn(prev, curr, output_column)
    return result


def auc_segment(prev, curr, output_column, epsilon=0):
    last_samp = 0 if prev is None else prev.samples
    return (curr.samples - last_samp) * max(curr.val_loss - epsilon, 0)


def sum_reduction(prev, curr, output_column, input_column='auc_segment'):
    last_tot = 0 if prev is None else prev[output_column]
    return last_tot + curr[input_column]


def auc(dataframe, carry_column='auc_segment',
        output_column='auc_agg', epsilon=0):
    result = pd_reduce(dataframe, carry_column,
                       partial(auc_segment, epsilon=epsilon))
    result = pd_reduce(result, output_column,
                       partial(sum_reduction, input_column=carry_column))
    return result


def auc_per_data(df, epsilons):
    for epsilon in epsilons:
        colname = f'auc_agg@{epsilon}'.replace('.', '_')
        results = []

        for name in df.name.unique():
            subset = df[(df.name == name)]
            output_column = f'auc_agg@{epsilon}'.replace('.', '_')
            label_auc = auc(
                subset, output_column=output_column, epsilon=epsilon)
            results.append(label_auc)

        df = pd.concat(results)
        df[f'str_{colname}'] = df[colname].round(2).astype(str)
        if epsilon > 0:
            rows, cols = df['val_loss'] > epsilon, f'str_{colname}'
            df.loc[rows, cols] = "> " + df.loc[rows, cols]
            # bound_strs = df.loc[df['val_loss'] > epsilon, f'str_{colname}']
            # bound_strs = "> " + bound_strs
    return df


def sc_segment(prev, curr, output_column, epsilon=0):
    if prev is not None:
        prev_sc = prev[output_column]
    else:
        prev_sc = 1e20

    if curr.val_loss <= epsilon:
        curr_sc = curr.samples
    else:
        curr_sc = 1e20
    return min(prev_sc, curr_sc)


def sc(dataframe, carry_column='sc_segment', output_column='sc', epsilon=0):
    result = pd_reduce(dataframe, output_column,
                       partial(sc_segment, epsilon=epsilon))
    return result


def sc_per_data(df, epsilons):
    for epsilon in epsilons:
        colname = f'sc@{epsilon}'.replace('.', '_')
        results = []
        for name in df.name.unique():
            subset = df[(df.name == name)]
            results.append(sc(subset, output_column=colname, epsilon=epsilon))

        df = pd.concat(results)
        df[f'str_{colname}'] = df[colname].astype(int).astype(str)
        df.loc[df[colname] > 1e10,
               f'str_{colname}'] = "> " + df.loc[df[colname] > 1e10,
                                                 'samples'].astype(str)
    # note that this overwrites `df` many times! not having an outer concat is
    # by design
    return df
