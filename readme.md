# Evaluating representations by the complexity of learning low-loss predictors

This code contains the MNIST experiments for the paper "Evaluating representations by the complexity of learning low-loss predictors", submitted to NeurIPS 2020.

## Running experiments

To run experiments, use `main_online_mdl_efficient.py`. This file contains a [JAX](https://jax.readthedocs.io/en/latest/) implementation of the method which vectorizes the construction of the loss-data curve across dataset sizes and random seeds. This enables it to be highly efficient; on a modern GPU it should take less than a minute to construct a 10-point loss-data curve with a two-hidden-layer MLP probe.

Scripts for running all the MNIST experiments from the paper are below, and in a block comment at the top of the main file. While they are for the `fish` shell, they are quite trivial and can easily be ported to `bash` or run by hand.

```fish
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
```

## Computing and visualizing results

To compute MDL, SDL, and εSC, use `results.ipynb`. In the bottom two cells you can set the paths to the experimental results (which are printed by the experiment main file). Running those cells will compute the results, generate a chart of the loss-data curve, and output a table of the results.