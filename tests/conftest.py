import logging
import warnings

from tests.test_data import processed_data  # re-export for other modules


def _quiet_torch_loggers():
    """Silence noisy torch loggers that trigger stream errors under pytest."""
    logger_names = [
        "torch._dynamo",
        "torch._dynamo.utils",
        "torch.fx.experimental.symbolic_shapes",
    ]
    for name in logger_names:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)


_quiet_torch_loggers()

warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'loss' is an instance of `nn\.Module`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Attribute 'logging_metrics' is an instance of `nn\.Module`.*",
    category=UserWarning,
)
