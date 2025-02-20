import argparse
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_project():
    """Initial project setup"""
    from config.config import config
    logger.info("Setting up project directories...")

    # Directories are created by config initialization
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Raw data directory: {config.raw_data_dir}")
    logger.info(f"Processed data directory: {config.processed_data_dir}")
    logger.info(f"Index directory: {config.index_dir}")

    logger.info("Project setup complete!")


def process_data():
    """Process dataset and create index"""
    from scripts.process_dataset import process_dataset
    logger.info("Processing dataset...")
    process_dataset()


def run_web():
    """Run the web interface"""
    from web.app import app
    from config.config import config
    logger.info(f"Starting web interface on {config.host}:{config.port}")
    app.run(host=config.host, port=config.port)


def main():
    parser = argparse.ArgumentParser(description="Image Retrieval System")
    parser.add_argument('command', choices=['setup', 'process', 'web', 'all'],
                        help='Command to run')

    args = parser.parse_args()

    try:
        if args.command == 'setup' or args.command == 'all':
            setup_project()

        if args.command == 'process' or args.command == 'all':
            process_data()

        if args.command == 'web' or args.command == 'all':
            run_web()

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()