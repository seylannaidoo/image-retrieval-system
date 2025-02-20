import yaml
from pathlib import Path


class Config:
    _instance = None

    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Set up data paths
        self.data_dir = Path(self._config.get('data_dir', 'data'))
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.index_dir = self.data_dir / 'index'

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        # Model settings
        self.model_name = self._config.get('model', {}).get('name', 'openai/clip-vit-base-patch32')
        self.max_images = self._config.get('dataset', {}).get('max_images', 500)
        self.image_size = self._config.get('model', {}).get('image_size', 224)

        # Web interface settings
        self.host = self._config.get('web', {}).get('host', 'localhost')
        self.port = self._config.get('web', {}).get('port', 5000)
        self.max_results = self._config.get('web', {}).get('max_results', 20)

    @classmethod
    def get_instance(cls, config_path: str = "config/settings.yaml"):
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance


# Create default instance only if config file exists
default_config_path = "config/settings.yaml"
config = Config.get_instance(default_config_path) if Path(default_config_path).exists() else None