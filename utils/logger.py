import logging
import logging.config
import yaml

def setup_logging(config_file="./utils/logging.yaml"):
    """Charge la configuration des logs depuis un fichier YAML."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

def get_logger(name=None):
    return logging.getLogger(name)

# logger.debug("Ceci est un log de debug")
# logger.info("Ceci est un log d'information")
# root_logger.warning("Ceci est un log de warning (root)")