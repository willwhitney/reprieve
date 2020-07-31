"""
# Scripts for fish shell to run sets of experiments:

# generates the results on noisy ground-truth labels shown in Figure 1
set SEEDS 0 2 4 6
for D in 0 1 2 3
    set SEED $SEEDS[(math "$D + 1")]
    env CUDA_VISIBLE_DEVICES=$D python main_online_mdl_efficient.py --data mnist_noisygt --ntrain 50000 --ntest 10000 --n_samples 2 --n_chunks 10 --first_seed $SEED &
end
wait

# generates the other MNIST results (raw pixels, CIFAR, VAE)
set REPS raw cifar_supervised mnist_vae
set DIMS 784 784 8
set SEEDS 0 1 2 3 4 5 6 7
for R in 0 1 2 4 5 6
    for D in 0 1 2 3
        set I (math "$R * 4 + $D + 1")
        set SEED $SEEDS[(math "$I % 8 + 1")]
        set DIM $DIMS[(math "$R % 3 + 1")]
        set REP $REPS[(math "$R % 3 + 1")]
        echo "Seed $SEED, Dim $DIM, Rep $REP"
        env CUDA_VISIBLE_DEVICES=$D python main_online_mdl_efficient.py --data mnist --ntrain 50000 --ntest 10000 --n_samples 1 --n_chunks 10 --repr $REP --repr_dim $DIM --first_seed $SEED &
    end
    wait
end
"""

import gc
import argparse
import sys
import math
import builtins
import numpy as np
import random
import torch
import pandas as pd

import jax
import jax.numpy as jnp
from jax.experimental import stax, optimizers
from jax import jit, grad
import jax.random as jr
import jax.lax as lax

import util
from mnist_noisygt_dataset import MNISTNoisyGTDataset

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--first_seed', type=int, default=0)
parser.add_argument('--ntrain', type=int, default=50000)
parser.add_argument('--ntest', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--repr', type=str, default="raw")
parser.add_argument('--n_chunks', type=int, default=10)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--data', type=str, default="mnist")
parser.add_argument('--debug', action='store_true')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--repr_dim', type=int, default=784)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--repr_level', type=int, default=3)
parser.add_argument('--name', default='')
args = parser.parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
args.batch_size = min(args.ntrain, args.batch_size)
# if len(args.name) > 0: args.name = args.name + '_'

torch.manual_seed(args.first_seed)
random.seed(args.first_seed)
np.random.seed(args.first_seed)


def stack_data(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.float())
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


if args.data == "mnist":
    train_loader, val_loader, test_loader = util.get_mnist_loaders(
        args.repr, DEVICE, args.repr_dim, args.repr_level,
        ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size)
elif args.data == "mnist_noisygt":
    p_corrupt = 0.05
    train_loader = torch.utils.data.DataLoader(MNISTNoisyGTDataset(
            split='train', ntrain=args.ntrain, nval=args.ntest, ntest=args.ntest, p_corrupt=p_corrupt),
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(MNISTNoisyGTDataset(
            split='val', ntrain=args.ntrain, nval=args.ntest, ntest=args.ntest, p_corrupt=p_corrupt),
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNISTNoisyGTDataset(
            split='test', ntrain=args.ntrain, nval=args.ntest, ntest=args.ntest, p_corrupt=p_corrupt),
        shuffle=True)
elif args.data == "svhn":
    train_loader, val_loader, test_loader = util.get_svhn_loaders(
        args.repr, DEVICE, ntrain=args.ntrain, ntest=args.ntest, batch_size=args.batch_size)

train_x, train_y = stack_data(train_loader)
train_x, train_y = train_x.numpy().astype(np.float32), train_y.numpy().astype(np.float32)
test_x, test_y = stack_data(test_loader)
test_x, test_y = test_x.numpy().astype(np.float32), test_y.numpy().astype(np.float32)

del train_loader
del val_loader
del test_loader
torch.cuda.empty_cache()

train_x, train_y, test_x, test_y = jnp.array(train_x), jnp.array(train_y), jnp.array(test_x), jnp.array(test_y)
train_x = train_x.reshape((train_x.shape[0], -1))
gc.collect()

input_dim = train_x[0].size
output_dim = int(train_y.max() + 1)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


train_targets = one_hot(train_y, output_dim)


def debug_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def eval_only(test_data, model):
    preds = model(test_data[0])
    max_preds = jnp.argmax(preds, axis=1)
    test_accuracy = jnp.mean(max_preds == test_data[1])
    test_loss = -jnp.mean(preds * one_hot(test_data[1], output_dim))
    return test_loss, test_accuracy

def make_model(seed, N):
    print = builtins.print
    if not args.debug:
        # if this has been jitted + vmapped, print won't do the right thing
        def print(*args, **kwargs):
            pass

    input_shape = train_x.shape[1]
    net_init, net_apply = stax.serial(
        stax.Flatten,
        stax.Dense(args.hidden_dim), stax.Relu,
        stax.Dense(args.hidden_dim), stax.Relu,
        stax.Dense(output_dim), stax.LogSoftmax,
    )

    rng = jr.PRNGKey(args.first_seed + seed)
    in_shape = (-1, input_shape)
    out_shape, net_params = net_init(rng, in_shape)
    net_opt_init, net_opt_update, net_get_params = optimizers.adam(
        step_size=1e-3)
    rng = jr.split(rng)[0]

    def loss_data(net_params, x_batch, targets_batch):
        preds = net_apply(net_params, x_batch)
        batch_loss = -jnp.mean(preds * targets_batch)
        return batch_loss
    loss_data_v_g = jax.value_and_grad(loss_data, argnums=(0, 1))

    n_batches = args.ntrain // args.batch_size
    def get_example(i):
        key = rng[0] + i
        index = jr.randint(jr.PRNGKey(key), shape=(1,),
                           minval=0, maxval=args.ntrain)
        index = index[0] % N
        x_i = lax.dynamic_index_in_dim(train_x, index, keepdims=False)
        target_i = lax.dynamic_index_in_dim(train_targets, index,
                                            keepdims=False)
        return x_i, target_i, index
    get_batch = jax.vmap(get_example)

    def full_step(state, indices):
        net_opt_state, update_count = state
        x_batch, targets_batch, indices = get_batch(indices)
        net_params = net_get_params(net_opt_state)

        loss_value, (net_g, lin_g) = loss_data_v_g(net_params, x_batch, targets_batch)
        acc = 0
        net_opt_state = net_opt_update(update_count, net_g, net_opt_state)
        update_count += 1
        return ((net_opt_state, update_count),
                (loss_value, acc))

    def train_epoch(carry, _):
        update_count, epochs_since_improve, (_, last_loss), net_opt_state = carry
        indices = jnp.arange(n_batches * args.batch_size)
        indices = indices.astype(jnp.int32).reshape((-1, args.batch_size))

        if args.debug and False:
            state, (losses, accuracies) = debug_scan(
                full_step,
                (net_opt_state, update_count),
                indices)
        else:
            state, (losses, accuracies) = jax.lax.scan(
                full_step,
                (net_opt_state, update_count),
                indices)
        (net_opt_state, update_count) = state
        net_params = net_get_params(net_opt_state)
        test_loss, test_accuracy = eval_only((test_x, test_y),
                                             jax.tree_util.Partial(
                                                 net_apply, net_params))
        print((f'Loss: {losses.mean()} \tTest loss: {test_loss} \t'
               f'Test accuracy: {test_accuracy}'))

        current_loss = jnp.mean(losses)

        # using a conditional will mess jax up, so do this hacky thing
        epochs_since_improve = (epochs_since_improve + 1) * \
            (current_loss < last_loss).astype(jnp.float32)
        carry = (update_count, epochs_since_improve,
                 (last_loss, current_loss), net_opt_state)
        return carry, current_loss

    net_opt_state = net_opt_init(net_params)

    carry = (0, 0., (1e10, 1e10), net_opt_state)
    if args.debug:
        carry, losses = debug_scan(train_epoch, carry,
                                   jnp.linspace(0, 99, args.epochs))
    else:
        carry, losses = jax.lax.scan(train_epoch, carry,
                                     jnp.linspace(0, 99, args.epochs))

    step_count, _, (_, last_loss), net_opt_state = carry

    net_params = net_get_params(net_opt_state)
    acc = 0
    final_predict = jax.tree_util.Partial(net_apply, net_params, )
    return final_predict, acc, losses[-1]


def full_experiment(train_sizes, seeds):
    if args.debug:
        model, train_accuracy, train_loss = make_model(0, args.ntrain)
        test_loss, test_accuracy = eval_only((test_x, test_y), model)
        sys.exit(0)

    make_models = jax.vmap(make_model)
    models, train_accuracies, train_losses = make_models(seeds, train_sizes)
    print("Built models.")

    eval_only_one = jax.partial(eval_only, (test_x, test_y))
    eval_only_all = jax.vmap(jit(eval_only_one))
    test_losses, test_accuracies = eval_only_all(models)
    print("Evaluated models.")
    results = np.stack([seeds, train_sizes, train_losses, train_accuracies,
                        test_losses, test_accuracies]).transpose()
    df = pd.DataFrame(results, columns=["seed", "samples", "train_loss",
                                        "train_accuracy", "test_loss",
                                        "test_accuracy"])
    return df


def compute_online_mdl(samples, test_losses):
    interval_widths = samples[1:] - samples[:-1]
    # if sending 10K points, you don't use the model trained on all 10K
    used_losses = test_losses[:-1]
    initial_loss = - jnp.log(1 / output_dim) * samples[0]
    return jnp.dot(interval_widths, used_losses) + initial_loss


if __name__ == '__main__':
    train_sizes = jnp.logspace(1, np.log10(args.ntrain),
                               args.n_chunks).astype(jnp.int32)
    train_sizes = jnp.repeat(train_sizes, args.n_samples)
    seeds = jnp.arange(args.n_chunks * args.n_samples).astype(jnp.int32)
    jobs = jnp.stack([train_sizes, seeds], axis=1)

    results = full_experiment(*jobs.transpose())

    for k, v in vars(args).items():
        results[k] = v
    name_pad = '_' if len(args.name) > 0 else ''
    filename = (f'results/realprobe_{args.name}{name_pad}{args.data}-'
                f'repr_{args.repr}_dim{args.repr_dim}_level{args.repr_level}-'
                f'seed{args.first_seed}.pkl')
    print(f"Writing results to {filename}")
    results.to_pickle(filename)

