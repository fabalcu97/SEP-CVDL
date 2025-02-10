def get_configuration(experiment_name):
    baseline_configuration = experiments["baseline"]
    configuration = experiments[experiment_name]

    new_configuration = {
        **baseline_configuration,
        **configuration,
        "hyperparameters": {
            **baseline_configuration["hyperparameters"],
            **(
                configuration["hyperparameters"]
                if "hyperparameters" in configuration
                else {}
            ),
        },
    }

    return new_configuration


output_path = "outputs"

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
        "apply_weighted_loss": False,
        "optimizer": "sgd_optimizer",
        "dataset": "GiMeFive",
        "use_scheduler": False,
        "use_label_smoothing": False,
        "hyperparameters": {
            "num_epochs": 80,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            # Model hyperparameters
            "dropout1": 0.2,
            "conv5_filters": 1024,  # fc1_input
            "fc1_input": 1024,
            "fc1_output": 2048,  # fc2_input
            "fc2_output": 1024,
        },
    },
    "class_weights_SGD": {
        "apply_weighted_loss": True,
        "optimizer": "sgd_optimizer",
    },
    "class_weights_ADAM": {
        "apply_weighted_loss": True,
        "optimizer": "adam_optimizer",
    },
    "class_weights_ADAMW": {
        "apply_weighted_loss": True,
        "optimizer": "adam_optimizer",
    },
    "class_weights_ADAMW_1": {
        "apply_weighted_loss": True,
        "optimizer": "adamw_optimizer",
        "hyperparameters": {
            "dropout1": 0.3,
            "conv5_filters": 768,
            "fc1_input": 768,
            "fc1_output": 1024,
            "fc2_output": 512,
        },
    },
    "SCHEDULER": {
        "apply_weighted_loss": True,
        "optimizer": "adamw_optimizer",
        "use_scheduler": True,
        "hyperparameters": {
            "dropout1": 0.3,
            "conv5_filters": 768,
            "fc1_input": 768,
            "fc1_output": 1024,
            "fc2_output": 512,
        },
    },
    "weighted_loss": {
        "apply_weighted_loss": True,
    },
    "weighted_loss_w_label_smoothing": {
        "apply_weighted_loss": True,
        "use_label_smoothing": True,
    },
    # weighted loss, label smoothing and hyperparameter tunning
    "wl_ls_hp_tunning": {
        "apply_weighted_loss": True,
        "use_label_smoothing": True,
        "hyperparameters": {
            "dropout1": 0.4,
            "conv5_filters": 768,
            "fc1_input": 768,
            "fc1_output": 512,
            "fc2_output": 256,
        },
    },
    "wl_ls_hp_tunning_2": {
        "apply_weighted_loss": True,
        "use_label_smoothing": True,
        "hyperparameters": {
            "dropout1": 0.4,
            "fc1_output": 512,
            "fc2_output": 256,
            # 2
            "conv5_filters": 1024,
            "fc1_input": 1024,
        },
    },
    "wl_ls_hp_tunning_3": {
        "apply_weighted_loss": True,
        "use_label_smoothing": True,
        "use_scheduler": True,
        "hyperparameters": {
            "dropout1": 0.4,
            "fc1_output": 512,
            "fc2_output": 256,
            # 2
            "conv5_filters": 1024,
            "fc1_input": 1024,
            # 3
            "learning_rate": 0.01,
            "weight_decay": 0.003,
            "momentum": 0.9,
        },
    },
    "wl_ls_hp_adw": {
        "apply_weighted_loss": True,
        "use_label_smoothing": True,
        "optimizer": "adamw_optimizer",
        "use_scheduler": True,
        "hyperparameters": {
            "dropout1": 0.4,
            "fc1_output": 512,
            "fc2_output": 256,
            "conv5_filters": 1024,
            "fc1_input": 1024,
        },
    },
    # increase batch size
    "batch_32_wl": {
        "apply_weighted_loss": True,
        "hyperparameters": {"batch_size": 32},
    },
    # activation(leaky relu)
    "leaky_relu": {},
    # batch norm after activation(leaky relu)
    "bn_after_activation": {},
    # weighted loss and lr
    "wl_lr": {
        "apply_weighted_loss": True,
    },
    # weighted loss and batch norm after activation(leaky relu)
    "wl_bn_after_lr": {},
}

# Change the experiment only
experiment_name = "wl_bn_after_lr"

configuration = get_configuration(experiment_name)
dataset = datasets[configuration["dataset"]]
hyperparameters = configuration["hyperparameters"]
