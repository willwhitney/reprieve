import os
import pandas as pd

import torchvision

import reprieve
from reprieve.representations import mnist_vae
from reprieve.mnist_noisy_label import MNISTNoisyLabelDataset
from reprieve.algorithms import torch_mlp as alg


def main(args):
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((1, 28, 28), 10)
    dataset_mnist = torchvision.datasets.MNIST(
        './data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    raw_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_mnist,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=False, cache_data=args.cache_data,
        verbose=True)
    raw_results = raw_loss_data_estimator.compute_curve(n_points=args.points)

    vae_repr = mnist_vae.build_repr(8)
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((8,), 10)
    vae_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_mnist,
        representation_fn=vae_repr,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=False, cache_data=args.cache_data,
        verbose=True)
    vae_results = vae_loss_data_estimator.compute_curve(n_points=args.points)

    dataset_noisygt = MNISTNoisyLabelDataset(
        train=True, p_corrupt=0.05)
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((784,), 10)
    noisy_loss_data_estimator = reprieve.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_noisygt,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=False, cache_data=args.cache_data,
        verbose=True)
    noisy_results = noisy_loss_data_estimator.compute_curve(
        n_points=args.points)

    raw_results['name'] = 'Raw'
    vae_results['name'] = 'VAE'
    noisy_results['name'] = 'Noisy labels'

    outcome_df = pd.concat([
        raw_results,
        vae_results,
        noisy_results,
    ])

    os.makedirs('results', exist_ok=True)
    save_path = ('results/'
                 f'{args.name}'
                 f'_train{args.train_steps}'
                 f'_seed{args.seeds}'
                 f'_point{args.points}')

    ns = [1000, 60000]
    epsilons = [1, 0.1, 0.01]
    reprieve.render_curve(outcome_df, ns, epsilons,
                          save_path=save_path + '.pdf')
    metrics_df = reprieve.compute_metrics(outcome_df, ns, epsilons)
    print(metrics_df)
    reprieve.render_latex(metrics_df, save_path=save_path + '.tex')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--no_cache', dest='cache_data', action='store_false')
    parser.add_argument('--train_steps', type=float, default=4e3)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--points', type=int, default=10)
    args = parser.parse_args()

    import time
    start = time.time()
    main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
