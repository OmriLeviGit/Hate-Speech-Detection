# Hate-speech Detection on X (Twitter)

A machine learning system for detecting antisemitic content on X (formerly Twitter), consisting of data annotation tools, web app for inference, and model comparison frameworks.

This repository contains two main components:

### [Classifier App](./classifier/)
A web-based application for model inference and training, and an evaluation framework with a unified API for comparison across different model architectures.

**Features:**
- Web GUI for single/batch post inference capabilities
- Comparison framework for multiple ML algorithms across different architectures
- Support for various sampling methods and data balancing techniques
- Model retraining with new data uploads

**Tech Stack:** Python, Gradio, scikit-learn, Transformers, XGBoost, TensorFlow/PyTorch

### [Data Tagging Website](./tagging_website/)
A web application for manual annotation by multiple annotators to create labeled training datasets from scraped social media content.

**Features:**
- User-friendly annotation interface with multi-user support
- Quality control and inter-annotator agreement tracking
- Pro user support for tracking and analytics
- Export capabilities for labeled datasets

**Tech Stack:** React, FastAPI, PostgreSQL

## Getting Started

Each component can be run independently using Docker. See the individual README files for detailed setup instructions:

- [Classifier App Setup](./classifier/README.md)
- [Data Tagging Setup](./tagging_website/README.md)
