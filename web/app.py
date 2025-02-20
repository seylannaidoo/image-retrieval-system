import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import base64
import io
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.clip_encoder import CLIPEncoder
from src.models.vector_store import VectorStore
from config.config import config

app = Flask(__name__)


def load_models():
    try:
        encoder = CLIPEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
        vector_store = VectorStore.load(config.index_dir)
        print("Models loaded successfully.")
        return encoder, vector_store
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None



encoder, vector_store = load_models()


def encode_image_base64(image: Image.Image, max_size: int = 400) -> str:
    """Convert PIL image to base64 string, resizing if needed"""
    # Calculate new size maintaining aspect ratio
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route('/')
def home():
    if encoder is None or vector_store is None:
        return render_template('error.html',
                               message="Error: Models not loaded. Please check the logs.")
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    print("Received a /search request")
    if encoder is None or vector_store is None:
        return jsonify({'error': 'Models not loaded'}), 500

    try:
        query = request.form.get('query', '')
        top_k = min(int(request.form.get('top_k', 6)), config.max_results)

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Encode query and search
        text_embedding = encoder.encode_text([query])
        distances, metadata = vector_store.search(text_embedding, k=top_k)

        # Prepare results
        results = []
        for distance, meta in zip(distances, metadata):
            try:
                img_path = meta['path']
                img = Image.open(img_path).convert('RGB')
                similarity = float(1 - distance) * 100  # Convert to percentage

                results.append({
                    'image': encode_image_base64(img),
                    'similarity': f"{similarity:.1f}%",
                    'filename': meta['filename']
                })
            except Exception as e:
                print(f"Error processing result {meta['path']}: {e}")
                continue

        return jsonify({
            'results': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Check system status"""
    if encoder is None or vector_store is None:
        return jsonify({'status': 'error', 'message': 'Models not loaded'})
    return jsonify({
        'status': 'ok',
        'images_indexed': len(vector_store),
        'model': config.model_name
    })


if __name__ == '__main__':
    app.run(
        host=config.host,
        port=config.port,
        debug=True  # Set to True for development

    )