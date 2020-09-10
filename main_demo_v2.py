import pandas as pd
import jax

import torchvision

import apiv2
import representations.mnist_vae
from mnist_noisy_label import MNISTNoisyLabelDataset
from algorithms import mlp as alg
# from algorithms import torch_mlp as alg


def main(args):
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((1, 28, 28), 10)
    dataset_mnist = torchvision.datasets.MNIST(
        './data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    raw_loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_mnist,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    raw_loss_data_estimator.compute_curve(n_points=10)

    vae_repr = representations.mnist_vae.build_repr(8)
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((8,), 10)
    vae_loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_mnist,
        representation_fn=vae_repr,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    vae_loss_data_estimator.compute_curve(n_points=10)

    dataset_noisygt = MNISTNoisyLabelDataset(
        train=True, p_corrupt=0.05)
    init_fn, train_step_fn, eval_fn = alg.make_algorithm((784,), 10)
    noisy_loss_data_estimator = apiv2.LossDataEstimator(
        init_fn, train_step_fn, eval_fn, dataset_noisygt,
        train_steps=args.train_steps, n_seeds=args.seeds,
        use_vmap=args.use_vmap, verbose=True)
    noisy_loss_data_estimator.compute_curve(n_points=10)

    raw_results = raw_loss_data_estimator.to_dataframe()
    raw_results['name'] = 'Raw'
    vae_results = vae_loss_data_estimator.to_dataframe()
    vae_results['name'] = 'VAE'
    noisy_results = noisy_loss_data_estimator.to_dataframe()
    noisy_results['name'] = 'Noisy labels'

    outcome_df = pd.concat([
        raw_results,
        vae_results,
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
