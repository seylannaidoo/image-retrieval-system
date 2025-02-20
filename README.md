# Multi-Modal Image Retrieval System

A robust image retrieval system that uses natural language queries to find relevant images from a dataset. The system leverages CLIP (Contrastive Language-Image Pre-Training) for understanding both text and images, providing accurate semantic search capabilities.

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Test Status](https://img.shields.io/badge/tests-passing-brightgreen)

## Features

- 🔍 Natural language image search
- 🎯 High-accuracy semantic matching using CLIP
- 🖥️ Clean, accessible web interface
- ⚡ Fast vector similarity search using FAISS
- 📊 Configurable number of results
- ♿ Inclusive design with accessibility features

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Development Setup](#development-setup)
- [Assumptions and Design Decisions](#assumptions-and-design-decisions)
- [Architecture](#architecture)

## Requirements

- Python 3.9 or higher
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)
- 500MB+ free disk space for the test dataset
- Kaggle account and API credentials
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-retrieval-system.git
   cd image-retrieval-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
4. Set up Kaggle API access:
   1. Sign up for a Kaggle account at https://www.kaggle.com if you haven't already
   2. Go to your Kaggle account settings (click on your profile picture → Settings)
   3. Scroll down to "API" section and click "Create New API Token"
   4. This will download a `kaggle.json` file
   5. Create a `.kaggle` directory:
      ```bash
      # On Windows:
      mkdir %USERPROFILE%\.kaggle

      # On Linux/Mac:
      mkdir ~/.kaggle
      ```
   6. Move the downloaded `kaggle.json` to the `.kaggle` directory:
      ```bash
      # On Windows:
      move kaggle.json %USERPROFILE%\.kaggle\kaggle.json

      # On Linux/Mac:
      mv kaggle.json ~/.kaggle/kaggle.json
      ```
   7. Set proper permissions:
      ```bash
      # On Windows - no action needed
      
      # On Linux/Mac:
      chmod 600 ~/.kaggle/kaggle.json
      ```

5. Set up the configuration:
   ```bash
   cp config/settings.yaml.example config/settings.yaml
   ```
   Edit `config/settings.yaml` to match your environment if needed.

## Quick Start

1. Initialize the project:
   ```bash
   python main.py setup
   ```

2. Process the dataset:
   ```bash
   python main.py process
   ```
   This will:
   - Automatically download the test dataset from Kaggle
   - Process images and compute embeddings
   - Create the search index
   
   Note: If the automatic download fails, you can manually:
   1. Download the dataset from: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
   2. Extract the contents
   3. Copy all images from the `test_data_v2` folder to `data/raw/` in your project directory


3. Start the web interface:
   ```bash
   python main.py web
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

To run all steps at once:
```bash
python main.py all
```

## Project Structure

```
.
├── config/              # Configuration files
├── data/                # Dataset and processed files
│   ├── raw/             # Original images
│   ├── processed/       # Processed data
│   └── index/           # Search indices
├── scripts/             # Processing scripts
├── src/                 # Source code
│   └── models/          # ML models and encoders
├── tests/               # Test suite
└── web/                 # Web interface
    ├── static/          # Static assets
    └── templates/       # HTML templates
```

## Running Tests

Run the full test suite:
```bash
pytest
```

Run specific test categories:
```bash
pytest tests/test_clip_encoder.py  # Test CLIP encoder
pytest tests/test_vector_store.py  # Test vector store
pytest tests/test_data_processing.py  # Test data processing
```

Generate coverage report:
```bash
pytest --cov=src tests/
```

## Configuration

Key configuration options in `config/settings.yaml`:

```yaml
data_dir: "data"              # Base data directory
model:
  name: "openai/clip-vit-base-patch32"  # CLIP model variant
  image_size: 224             # Input image size
dataset:
  max_images: 500            # Maximum images to process
web:
  host: "localhost"          # Web interface host
  port: 5000                 # Web interface port
  max_results: 20            # Maximum search results
```

## Assumptions and Design Decisions

1. **Dataset**
   - Using the test_data_v2 folder from the AI vs. human-generated dataset
   - Limited to 500 images for demonstration purposes
   - Assumes JPEG/PNG format images

2. **Model Selection**
   - Using CLIP for its strong zero-shot image-text matching capabilities
   - Selected base model for balance of performance and resource usage
   - Embeddings are normalized for cosine similarity search

3. **Performance**
   - Uses FAISS for efficient similarity search
   - Batched processing for memory efficiency
   - Caches embeddings to avoid recomputation

4. **Web Interface**
   - Focused on accessibility and usability
   - Responsive design for various screen sizes
   - Progressive enhancement for better user experience

5. **Error Handling**
   - Graceful fallback for missing images
   - Clear error messages for users
   - Logging for debugging and monitoring

## Architecture

The system follows a modular architecture with these key components:

1. **Data Processing Pipeline**
   - Image loading and validation
   - CLIP embedding computation
   - Vector index creation

2. **Search System**
   - Text query embedding
   - FAISS similarity search
   - Result ranking and filtering

3. **Web Interface**
   - RESTful API endpoints
   - Responsive frontend
   - Accessibility features

For detailed architecture diagram, see [ARCHITECTURE.md](./ARCHITECTURE.md).