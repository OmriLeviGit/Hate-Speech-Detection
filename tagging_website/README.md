
# Data Tagging Website

A web application for manual annotation by multiple annotators to create labeled training datasets from social media content.

## Features

- User-friendly annotation interface with multi-user support
- Quality control and inter-annotator agreement tracking
- Pro user support for advanced tracking and analytics
- Export capabilities for labeled datasets

## Prerequisites

- [Docker](https://www.docker.com/get-started) (required)
- PostgreSQL (uses local volumes)

## Quick Start

Navigate to `/tagging-website` directory:

```bash
docker-compose up
```

Access the application at `http://localhost:3000`

## Tech Stack

- **Frontend**: React, Nginx
- **Backend**: FastAPI, SQLAlchemy, Python
- **Database**: PostgreSQL
- **Deployment**: Docker

## File Structure

Core file structure:


```
tagging_website/
├── clientside/                   # React frontend application
│   ├── src/                      # React components and logic
│   ├── public/                   # Static assets
│   ├── package.json              # Frontend dependencies
│   └── Dockerfile                # Frontend container config
├── serverside/                   # FastAPI backend application
│   ├── data/                     # Data storage and management
│   │   ├── data_of_interest/     # X users that will be scraped
│   │   ├── ready_to_load/        # Processed data ready to load to the database
│   │   ├── scraping_input/       # Raw input data from scraping sources
│   │   └── features.txt          # Feature definitions
│   ├── helper_functions/         # Utility functions
│   ├── auth.py                   # Authentication logic
│   ├── controller.py             # API route handlers
│   ├── db_service.py             # Database operations
│   ├── model.py                  # SQLAlchemy schemas
│   └── server.py                 # FastAPI application entry point
└── docker-compose.yml            # Multi-container orchestration
```

## Configuration

Copy `.env.example` to `.env.local` in the serverside directory and configure your environment variables before running.

The default configuration will work out of the box, but should be customized for production use due to security considerations.