"""Logging setup."""

import os
import sys
import logging


def setup_logger(name: str, output_dir: str, rank: int = 0) -> logging.Logger:
    """Create a logger that writes to file and stdout (rank 0 only)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (rank 0 only)
    if rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler (always, for all ranks during debugging)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(output_dir, f"log_rank{rank}.txt"), mode="a"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
