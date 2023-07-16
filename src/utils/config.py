from ray import tune

default_config = {
    'sign_size': 32,
    'cha_input': 16,
    'cha_hidden': 1024,
    'K': 3,
    'dropout_input': 0.1,
    'dropout_hidden': 0.1,
    'dropout_output': 0.1,
    'N': 7,
    'hidden_layer': 512,
    'dropout_size': 0.05
}

tune_config = {
    'sign_size': tune.choice([16, 32]),
    'cha_input': tune.choice([8, 16, 32]),
    'cha_hidden': tune.choice([256, 512]),
    'K': tune.choice([3]),
    'dropout_input': tune.choice([0.1]),
    'dropout_hidden': tune.choice([0.1]),
    'dropout_output': tune.choice([0.1]),
    'N': tune.choice([3]),
    'hidden_layer': tune.choice([256, 512]),
    'dropout_size': tune.choice([0.05])
}
