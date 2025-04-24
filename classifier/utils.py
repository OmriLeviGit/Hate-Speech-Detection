import copy

from sklearn.metrics import classification_report, confusion_matrix


def print_model_header(model_name, total_length=80):
    name_length = len(model_name)
    side_length = (total_length - name_length - 2) // 2

    extra_dash = 1 if (total_length - name_length - 2) % 2 != 0 else 0

    print("\n", "-" * side_length + f" {model_name} " + "-" * (side_length + extra_dash))


def print_model_results(best_model, best_score, y, y_pred, model_duration, total_duration):
    print("\nBest model:", best_model)
    print("Best cross-validation score:", best_score)
    print("\nClassification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(f"\nTraining time: {format_duration(model_duration)}")
    print(f"Total runtime: {format_duration(total_duration)}")


def format_duration(seconds):
    """ Convert duration from seconds to hh:mm:ss or mm:ss format """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    return f"{int(minutes)}:{int(seconds):02}"


def generate_model_configs(base_configs, learning_rates, l2_values):
    all_configs = []

    for base_config in base_configs:
        for lr in learning_rates:
            for l2 in l2_values:
                # Create a deep copy to avoid modifying the original
                config = copy.deepcopy(base_config)

                # Update the hyperparameters
                config["hyper_parameters"]["learning_rate"] = lr
                config["hyper_parameters"]["l2_regularization"] = l2

                # Update the model name to reflect the parameters
                config["model_name"] = f"{base_config['model_name']} (lr={lr:.3f}, l2={l2:.3f})"

                # Add to our list of configurations
                all_configs.append(config)

    return all_configs
