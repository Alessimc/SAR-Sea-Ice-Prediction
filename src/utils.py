import logging
import sys
import os
import torch

def init_logging():
    # Suppress rasterio / GDAL warnings
    os.environ["CPL_LOG_LEVEL"] = "ERROR"
    os.environ["CPL_LOG"] = "OFF"

    logging.getLogger("rasterio").setLevel(logging.ERROR)
    logging.getLogger("rasterio.env").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    # Set up console logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

    return logger


def get_gpu_memory():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**3  # GB
        mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Mem: {mem_alloc:.2f}GB alloc, {mem_reserved:.2f}GB reserved"
    else:
        return "CPU"
