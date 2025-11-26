import logging
import os
import sys


def create_logger(logfile, args):
    """Create logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Log arguments
    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    return logger

