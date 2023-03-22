def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--from-scratch', action='store_true')
    args = parser.parse_args()
    if args.big:
        from wiki103.config import big_config as config
        print("Using big config")
    else:
        from wiki103.config import small_config as config
        print("Using small config")

    config["no-train"] = args.no_train
    config["from-scratch"] = args.from_scratch
    return config


small_config = dict(
    # model parameters
    n_blocks=12,
    n_heads=16,
    n_tokens=512,
    n_embeddings=512,
    cutoffs=[10000, 30000],
    tie_embedding=True,
    # Training parameters
    batch_size=8,
    n_epochs=20,
    learning_rate=0.0003,
    weight_decay=0.1,
    accumulate_gradients=16,
)

big_config = dict(
    # model parameters
    n_blocks=16,
    n_heads=16,
    n_tokens=1024,  # 2048 + 1024,
    n_embeddings=1024,
    cutoffs=[20000, 60000],
    tie_embedding=True,
    # Training parameters
    batch_size=1,
    n_epochs=60,
    learning_rate=0.0003,
    weight_decay=0.1,
    accumulate_gradients=128,
)
