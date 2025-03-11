import os
import logging

def setup_logger(args):
    # Create /logs folder if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), 'logs')
    
    # Construct the log file name based on input arguments
    log_file_name = args.log_name
    log_file_path = os.path.join(logs_dir, log_file_name)
    
    # Configure the logger
    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO)  # Set the logging level

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # Ensure previous handlers are not added again
    if not logger.handlers:
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
