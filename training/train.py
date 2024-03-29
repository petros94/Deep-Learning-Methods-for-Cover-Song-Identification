import json
from training.train_cnn_angular import train_angular_loss
from training.train_cnn_triplet import train_triplet_loss
from training.train_embeddings_angular import train_embeddings_loss


def train(model, train_sets, valid_set, config_path, checkpoint_dir, results_dir, valid_set_full=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    if config["model"]["type"] == "embeddings":
        return train_embeddings_loss(
            model,
            train_sets,
            valid_set,
            config["train"]["n_epochs"],
            config["train"]["patience"],
            config["train"]["lr"],
            checkpoint_dir,
            results_dir
        )

    if config["loss"] == "triplet":
        return train_triplet_loss(
            model,
            train_sets,
            valid_set,
            config["train"]["n_epochs"],
            config["train"]["patience"],
            config["train"]["lr"],
            checkpoint_dir,
            results_dir,
        )
    elif config['loss'] == "angular":
        return train_angular_loss(
            model,
            train_sets,
            valid_set,
            config["train"]["n_epochs"],
            config["train"]["patience"],
            config["train"]["lr"],
            checkpoint_dir,
            results_dir,
            valid_set_full
        )
