datasets = {
    "GiMeFive": {
        "train": {
            "images": "../data/train/",
            "labels": "../data/train_labels.csv",
        },
        "validation": {
            "images": "../data/valid/",
            "labels": "../data/valid_labels.csv",
        },
        "test": {
            "images": "../data/test/",
            "labels": "../data/test_labels.csv",
        },
    },
}

experiments = {
    "baseline": {
        "apply_class_weight": False,
        "optimizer": "sgd_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    },
    "class_weights_SGD": {
        "apply_class_weight": True,
        "optimizer": "sgd_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    },
    "class_weights_ADAM": {
        "apply_class_weights": True,
        "optimizer": "adam_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    },
}
