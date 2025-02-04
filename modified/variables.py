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
            "learning_rate": 1e-3,  # 0.001
            "weight_decay": 1e-4,
            "momentum": 0.9,
        },
    },
    "class_weights_ADAM": {
        "apply_class_weight": True,
        "optimizer": "adam_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    "class_weights_ADAMW": {
        "apply_class_weight": True,
        "optimizer": "adam_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
    },
    "class_weights_ADAMW_1": {
        "apply_class_weight": True,
        "optimizer": "adam_optimizer",
        "dataset": "GiMeFive",
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dropout1": 0.3,
            "conv5_filters": 768,
            "fc1_input": 768,
            "fc1_output": 1024,
            "fc2_output": 512,
        },
    },
}

output_path = "outputs"

# Change the experiment only
experiment_name = "class_weights_ADAMW_1"

configuration = experiments[experiment_name]
dataset = datasets[configuration["dataset"]]
hyperparameters = configuration["hyperparameters"]
