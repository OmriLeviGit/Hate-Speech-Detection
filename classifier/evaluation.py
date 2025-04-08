# # Usage example
# import time
# import pandas as pd
# from typing import List, Dict, Any, Optional
#
# from classifier.BaseTextClassifier import BaseTextClassifier
#
#
# # Your BaseTextClassifier class as shown above
# # ...
#
# class ModelEvaluator:
#     """Evaluates multiple text classifiers and compares their performance."""
#
#     def __init__(self, classifiers: List[BaseTextClassifier], data_path: Optional[str] = None):
#         """
#         Initialize with a list of classifier instances and optional data path.
#
#         Args:
#             classifiers: List of classifier instances that inherit from BaseTextClassifier
#             data_path: Path to the dataset (if None, classifiers will use their sample data)
#         """
#         self.classifiers = classifiers
#         self.data_path = data_path
#         self.results = {}
#
#     def run_evaluation(self, test_size: float = 0.2, validation_size: float = 0.1,
#                        custom_params: Dict[str, Dict] = None) -> Dict[str, Dict]:
#         """
#         Evaluate all classifiers using the same data split.
#
#         Args:
#             test_size: Proportion of data to use for testing
#             validation_size: Proportion of data to use for validation
#             custom_params: Dictionary mapping classifier class names to training parameters
#
#         Returns:
#             Dictionary with evaluation results for all classifiers
#         """
#         if custom_params is None:
#             custom_params = {}
#
#         results = {}
#
#         # Load data once
#         data = None
#         for i, classifier in enumerate(self.classifiers):
#             classifier_name = classifier.__class__.__name__
#             print(f"Evaluating {classifier_name}...")
#
#             try:
#                 # Measure execution time
#                 start_time = time.time()
#
#                 # Load data (use the first classifier's data for all if not provided)
#                 if data is None:
#                     data = classifier.load_data(self.data_path)
#
#                 # Prepare datasets
#                 datasets = classifier.prepare_datasets(data, test_size, validation_size)
#
#                 # Preprocess data
#                 processed_datasets = classifier.preprocess_data(datasets)
#
#                 # Get custom parameters for this classifier if available
#                 params = custom_params.get(classifier_name, {})
#
#                 # Train the model
#                 classifier.train(processed_datasets, **params)
#
#                 # Evaluate the model
#                 metrics = classifier.evaluate(processed_datasets.get('test', None))
#
#                 # Record total time
#                 execution_time = time.time() - start_time
#
#                 # Store results
#                 results[classifier_name] = {
#                     'metrics': metrics,
#                     'execution_time': execution_time,
#                     'classifier': classifier  # Keep reference to the classifier
#                 }
#
#                 print(f"  Completed in {execution_time:.2f}s")
#                 print(f"  Metrics: {metrics}")
#
#             except Exception as e:
#                 print(f"  Error evaluating {classifier_name}: {str(e)}")
#                 results[classifier_name] = {'error': str(e)}
#
#         self.results = results
#         return results
#
#     def get_summary_table(self) -> pd.DataFrame:
#         """
#         Create a DataFrame summarizing the evaluation results.
#
#         Returns:
#             Pandas DataFrame with metrics for all classifiers
#         """
#         if not self.results:
#             raise ValueError("No results available. Run evaluation first.")
#
#         # Extract all unique metric names
#         all_metrics = set()
#         for result in self.results.values():
#             if 'metrics' in result:
#                 all_metrics.update(result['metrics'].keys())
#
#         # Prepare data for DataFrame
#         data = []
#         for classifier_name, result in self.results.items():
#             row = {'Classifier': classifier_name, 'Time (s)': result.get('execution_time', None)}
#
#             # Add metrics if available
#             if 'metrics' in result:
#                 for metric in all_metrics:
#                     row[metric] = result['metrics'].get(metric, None)
#
#             # Add error if present
#             if 'error' in result:
#                 row['Error'] = result['error']
#
#             data.append(row)
#
#         return pd.DataFrame(data)
#
#     def predict_with_best(self, text: str, metric: str = 'accuracy') -> Dict[str, Any]:
#         """
#         Make a prediction using the best performing classifier.
#
#         Args:
#             text: Text to classify
#             metric: Metric to use for determining the best classifier
#
#         Returns:
#             Dictionary with prediction and classifier information
#         """
#         if not self.results:
#             raise ValueError("No results available. Run evaluation first.")
#
#         # Find the best classifier based on the specified metric
#         best_classifier = None
#         best_score = -float('inf')
#
#         for classifier_name, result in self.results.items():
#             if 'metrics' in result and metric in result['metrics']:
#                 score = result['metrics'][metric]
#                 if score > best_score:
#                     best_score = score
#                     best_classifier = result['classifier']
#
#         if best_classifier is None:
#             raise ValueError(f"No classifier with metric '{metric}' found.")
#
#         # Make prediction with the best classifier
#         prediction = best_classifier.predict(text)
#
#         return {
#             'prediction': prediction,
#             'classifier': best_classifier.__class__.__name__,
#             'metric_score': best_score
#         }
#
#     def run_hyperparameter_search(self, classifier_class, param_grid,
#                                   test_size=0.2, validation_size=0.1):
#         """
#         Run grid search for best hyperparameters for a specific classifier.
#
#         Args:
#             classifier_class: The classifier class to instantiate
#             param_grid: Dictionary of hyperparameter names to possible values
#             test_size, validation_size: Data split proportions
#
#         Returns:
#             Dictionary with results for each hyperparameter combination
#         """
#         results = {}
#
#         # Generate all hyperparameter combinations
#         param_combinations = [dict(zip(param_grid.keys(), values))
#                               for values in itertools.product(*param_grid.values())]
#
#         # Load data once
#         base_classifier = classifier_class()
#         data = base_classifier.load_data(self.data_path)
#
#         for params in param_combinations:
#             # Create classifier with these hyperparameters
#             classifier = classifier_class()
#             classifier.set_hyperparameters(**params)
#
#             # Run evaluation steps (prepare, train, evaluate)
#             datasets = classifier.prepare_datasets(data, test_size, validation_size)
#             processed_datasets = classifier.preprocess_data(datasets)
#             classifier.train(processed_datasets)
#             metrics = classifier.evaluate(processed_datasets.get('test', None))
#
#             # Store results with hyperparameter info
#             param_key = "_".join([f"{k}={v}" for k, v in params.items()])
#             results[param_key] = {
#                 'params': params,
#                 'metrics': metrics
#             }
#
#         return results