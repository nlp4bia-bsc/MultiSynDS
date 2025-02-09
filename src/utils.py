import logging
import os
import yaml

logger = None  # Global logger variable

def setup_logger(path="logs/app.log"):
    global logger
    if logger is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        logger = logging.getLogger("custom_logger")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)

def log_info(message):
    if logger is None:
        raise ValueError("Logger is not set up. Call setup_logger(path) first.")
    logger.info(message)


# Add datetime to folder name
def path_with_datetime(path):
    from datetime import datetime
    import os
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(path, dt)

# Load YAML configuration

def load_config(config_path=None):
    if config_path is None:
        # Get the .py file that called this function
        import inspect
        caller = inspect.stack()[1].filename
        # Get the directory of the caller
        caller_dir = os.path.dirname(caller)
        # Assume the config file is in the same directory as the caller
        config_path = os.path.join(caller_dir, "config.yaml")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def log_config(config):
    global logger
    """Logs the current configuration settings"""
    logger.info("Configuration Settings:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")