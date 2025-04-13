import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def visualize(metrics_tuples, save_fig=False):
    """
    Compare performance metrics from multiple models.

    Args:
        metrics_tuples: List of tuples, each containing (config, metrics) for one model
        save_fig: True if you want to save the figure as PNG
    """
    # Extract configs and metrics from tuples
    configs = [item[0] for item in metrics_tuples]
    model_metrics = [item[1] for item in metrics_tuples]
    model_names = [config["model_name"] for config in configs]

    num_models = len(model_metrics)

    # Set up the figure with subplots - adding one more for overfitting detection
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    # 1. Training loss comparison
    ax = axs[0, 0]
    ax.set_title('Training Loss Comparison', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    max_epochs = 0
    for i, metrics in enumerate(model_metrics):
        if 'learning_curves' in metrics and 'train_losses' in metrics['learning_curves']:
            epochs = metrics['learning_curves']['epochs']
            losses = metrics['learning_curves']['train_losses']
            ax.plot(epochs, losses, 'o-', label=model_names[i], color=colors[i], linewidth=2)
            max_epochs = max(max_epochs, len(epochs))

    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. Training vs Validation Accuracy
    ax = axs[0, 1]
    ax.set_title('Training vs Validation Accuracy', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    for i, metrics in enumerate(model_metrics):
        if 'learning_curves' in metrics:
            epochs = metrics['learning_curves']['epochs']

            # Plot training accuracy
            if 'train_accuracies' in metrics['learning_curves']:
                train_accs = metrics['learning_curves']['train_accuracies']
                ax.plot(epochs, train_accs, 'o--', label=f"{model_names[i]} (Train)",
                        color=colors[i], linewidth=2, alpha=0.7)

            # Plot validation accuracy
            if 'val_accuracies' in metrics['learning_curves']:
                val_accs = metrics['learning_curves']['val_accuracies']
                ax.plot(epochs, val_accs, 'o-', label=f"{model_names[i]} (Val)",
                        color=colors[i], linewidth=2)

    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. Accuracy Gap (Overfitting Detection)
    ax = axs[0, 2]
    ax.set_title('Accuracy Gap (Train-Val)', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')

    for i, metrics in enumerate(model_metrics):
        if 'learning_curves' in metrics and 'accuracy_gaps' in metrics['learning_curves']:
            epochs = metrics['learning_curves']['epochs']
            gaps = metrics['learning_curves']['accuracy_gaps']
            ax.plot(epochs, gaps, 'o-', label=model_names[i], color=colors[i], linewidth=2)

            # Add a horizontal line at y=0
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5,
                       label='Potential overfitting threshold' if i == 0 else "")

    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. Final metrics comparison (bar chart)
    ax = axs[1, 0]
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(metrics_names))
    width = 0.8 / num_models  # Width of bars, adjusted for number of models

    for i, metrics in enumerate(model_metrics):
        values = [metrics.get(metric, 0) for metric in metrics_names]
        offset = (i - num_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_names[i], color=colors[i])

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_title('Final Evaluation Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, axis='y')

    # 5. Time comparison (training and evaluation)
    ax = axs[1, 1]

    # Prepare data for time comparison
    times = []
    for metrics in model_metrics:
        time_data = {
            'Training': metrics.get('total_training_time', 0),
            'Evaluation': metrics.get('time', 0),
            'Total': metrics.get('total_training_time', 0) + metrics.get('time', 0)
        }
        times.append(time_data)

    # Set up bar positions
    time_categories = ['Training', 'Evaluation', 'Total']
    x = np.arange(len(time_categories))

    for i, time_data in enumerate(times):
        values = [time_data[cat] for cat in time_categories]
        offset = (i - num_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_names[i], color=colors[i])

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{value:.2f}s', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_title('Time Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(time_categories)
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True, axis='y')

    # 6. Empty subplot for balance
    axs[1, 2].axis('off')

    # Add a new section to display hyperparameters
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])  # Adjust for more space at bottom

    # Create a summary table with metrics and key hyperparameters
    summary_data = []
    for i, (config, metrics) in enumerate(metrics_tuples):
        hyper_params = config.get("hyper_parameters", {})
        lr = hyper_params.get("learning_rate", "N/A")
        epochs = hyper_params.get("epochs", "N/A")

        summary_data.append([
            model_names[i],
            f"{metrics.get('accuracy', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}",
            f"{lr}",
            f"{epochs}",
            f"{config.get('emoji_processing', 'None')}",
            f"{config.get('combine_irrelevant', False)}"
        ])

    table_columns = ['Model', 'Accuracy', 'F1 Score', 'LR', 'Epochs', 'Emoji', 'Combine']
    table = plt.table(
        cellText=summary_data,
        colLabels=table_columns,
        loc='bottom',
        bbox=[0.05, -0.35, 0.9, 0.2]  # [left, bottom, width, height]
    )

    # Adjust cell properties for better fit
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Smaller font
    table.scale(1, 1.5)  # Taller rows

    # Adjust column widths - make model column wider, others narrower
    col_widths = [0.3, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15]
    for i, width in enumerate(col_widths):
        for j in range(len(summary_data) + 1):  # +1 for header row
            cell = table[(j, i)]
            cell.set_width(width)

    plt.subplots_adjust(bottom=0.3)

    plt.show()

    if save_fig:
        fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
