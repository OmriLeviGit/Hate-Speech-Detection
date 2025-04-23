import copy

import matplotlib.pyplot as plt
import numpy as np



def generate_model_configs(base_configs, learning_rates, l2_values):
    """
    Generate all combinations of model configurations with different learning rates and L2 values.

    Args:
        base_configs (list): List of base model configurations
        learning_rates (list): List of learning rate values to try
        l2_values (list): List of L2 regularization values to try

    Returns:
        list: All combinations of configurations
    """

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


def print_model_header(model_name, total_length=80):
    name_length = len(model_name)
    side_length = (total_length - name_length - 2) // 2

    extra_dash = 1 if (total_length - name_length - 2) % 2 != 0 else 0

    print("\n", "-" * side_length + f" {model_name} " + "-" * (side_length + extra_dash), "\n")


def visualize(metrics_tuples, save_fig=False):
    """
    Create a grid of model comparison plots with a single unified legend.

    Parameters:
    -----------
    metrics_tuples : list of tuples
        List of (model_config, metrics_dict) tuples with performance data
    save_fig : bool
        Whether to save the figure to disk
    """

    # Create a more compact figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Model Performance Comparison', fontsize=14)

    # Flatten axes for easier iteration
    axs = axs.flatten()

    # Plot titles
    titles = ['Training Loss', 'Training vs Validation Accuracy',
              'Accuracy Gap (Train-Val)', 'Final Evaluation Metrics']

    # Add subplot titles
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Epoch')

    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Accuracy')
    axs[2].set_ylabel('Gap')
    axs[3].set_ylabel('Score')

    # Add data to each subplot
    model_handles = []
    model_labels = []

    # Prepare for final metrics bar positions
    final_metrics = ['accuracy', 'precision', 'recall', 'f1']
    bar_width = 0.15
    num_models = len(metrics_tuples)

    for i, (model_config, metrics) in enumerate(metrics_tuples):
        # Extract model name for the legend
        model_name = model_config.get('model_name', 'Unknown model')

        # Get learning curves data
        curves = metrics.get('learning_curves', {})
        epochs = curves.get('epochs', [])

        # Plot training loss
        if 'train_losses' in curves:
            h, = axs[0].plot(epochs, curves['train_losses'], linewidth=1.5)
            model_handles.append(h)
            model_labels.append(model_name)
            model_color = h.get_color()  # Get the color to use it for both lines

        # Plot training and validation accuracy with same color but different line styles
        if 'train_accuracies' in curves and 'val_accuracies' in curves:
            axs[1].plot(epochs, curves['train_accuracies'], linewidth=1.5,
                        linestyle='--', color=model_color, alpha=0.7)
            axs[1].plot(epochs, curves['val_accuracies'], linewidth=1.5,
                        linestyle='-', color=model_color)

        # Plot accuracy gap
        if 'accuracy_gaps' in curves:
            axs[2].plot(epochs, curves['accuracy_gaps'], linewidth=1.5, color=model_color)

        # Plot final metrics (bar chart) - side by side
        if all(m in metrics for m in final_metrics):
            metric_values = [metrics[m] for m in final_metrics]
            x_pos = np.arange(len(final_metrics))

            # Offset each model's bars
            offset = (i - num_models / 2) * bar_width + bar_width / 2

            axs[3].bar(x_pos + offset, metric_values, width=bar_width,
                       alpha=0.8, color=model_color, label=model_name)

    # Set up the x-axis for the final metrics plot
    axs[3].set_xticks(np.arange(len(final_metrics)))
    axs[3].set_xticklabels(final_metrics, rotation=45)
    axs[3].set_xlabel('')

    # Create a single legend for all plots
    fig.legend(model_handles, model_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0), ncol=min(len(model_labels), 3))

    # Add a small legend for train/val line styles in the accuracy subplot
    train_line = plt.Line2D([0], [0], color='gray', linestyle='--', label='Training')
    val_line = plt.Line2D([0], [0], color='gray', linestyle='-', label='Validation')
    axs[1].legend(handles=[train_line, val_line], loc='upper right', fontsize=8)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_fig:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()


def create_summary_table(metrics_list, save_fig=False):
    # Prepare the data for the table
    data = []
    columns = ['Model', 'Accuracy', 'F1', 'LR', 'L2', 'Emoji', 'Classes', 'Train Time']

    for config, metrics in metrics_list:
        hyper_params = config.get("hyper_parameters", {})
        lr = hyper_params.get("learning_rate", "N/A")
        l2 = hyper_params.get("l2_regularization", "N/A")
        epochs = hyper_params.get("epochs", "N/A")

        base_name = config["model_name"].split("(")[0].strip()

        # Determine number of classes
        if config.get("combine_irrelevant", False):
            classes = "2 (combined)"
        elif "irrelevant" in str(config.get("distribution", {})):
            classes = "3"
        else:
            classes = "2"

        data.append([
            base_name,
            f"{metrics.get('accuracy', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}",
            f"{lr:.3f}",
            f"{l2:.3f}",
            f"{config.get('emoji_processing', 'None')}",
            classes,
            f"{metrics.get('total_training_time', 0):.1f}s"
        ])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 1))
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Set column widths
    table.auto_set_column_width(col=list(range(len(columns))))

    plt.title('Model Configuration Summary', fontsize=14)
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
