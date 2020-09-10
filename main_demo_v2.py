import pandas as pd
import jax

import torchvision

import apiv2
import mnist_vae_repr
from mnist_noisygt_dataset import MNISTNoisyGTDataset
# from algorithms import mlp as alg
from algorithms import torch_mlp as alg


def main(args):
    dataset_mnist = torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    raw_loss_data_estimator = apiv2.LossDataEstimator(
        alg.init_fn, alg.train_step_fn, alg.eval_fn, dataset_mnist,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    raw_loss_data_estimator.compute_curve(n_points=10)

    dataset_noisygt = MNISTNoisyGTDataset(
        split='train', ntrain=60000, nval=0, ntest=0,
        p_corrupt=0.05)
    noisy_loss_data_estimator = apiv2.LossDataEstimator(
        alg.init_fn, alg.train_step_fn, alg.eval_fn, dataset_noisygt,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    noisy_loss_data_estimator.compute_curve(n_points=10)

    raw_results = raw_loss_data_estimator.to_dataframe()
    raw_results['name'] = 'Raw'
    noisy_results = noisy_loss_data_estimator.to_dataframe()
    noisy_results['name'] = 'Noisy labels'

    outcome_df = pd.concat([
        raw_results,
        noisy_results,
    ])

    save_path = ('results/'
                 f'{args.name}_vmap{args.use_vmap}'
                 f'_train{args.train_steps}'
                 f'_seed{args.seeds}.png')
    apiv2.render_curve(outcome_df, save_path=save_path)
    # return outcome_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_vmap', dest='use_vmap', action='store_false')
    parser.add_argument('--train_steps', type=float, default=4e3)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    import time
    start = time.time()
    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
    end = time.time()
    print(f"Time: {end - start :.3f} seconds")
