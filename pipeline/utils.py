from dotenv import dotenv_values
from rich.table import Table
from rich.console import Console
from safetensors.torch import save_file, load_file
from typing import Dict
import logging
import torch
import os


def load_environment() -> None:
    """
    Load environment variables from environment, if absent, load from .env file.

    This is a convenience function for running Docker container of this repository.
    """

    dotenv_path = ".env"
    env_vars = dotenv_values(dotenv_path)

    for key, value in env_vars.items():
        if key not in os.environ and value is not None:
            os.environ[key] = value


def get_train_logger(run_id: str) -> logging.Logger:
    """
    Get a logger for training logs. 
    Logs are saved (DEBUG level) to a file in ./models/{run_id} directory and printed to the console (WARNING level).
    """

    log_file = f"./models/{run_id}/training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def health_check() -> None:
    """
    Perform a health check.
    """

    console = Console()

    console.print(f"PyTorch Version: {torch.__version__}")
    console.print(f"CUDA Version: {torch.version.cuda}")  # type: ignore
    console.print(f"CUDA Available: {str(torch.cuda.is_available())}")
    console.print("\n")

    if torch.cuda.is_available():
        gpu_table = Table("ID", "GPU")
        for i in range(torch.cuda.device_count()):
            gpu_table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))
        console.print(gpu_table)
    else:
        console.print("No GPUs available.")

    env_vars = ["HF_REPOSITORY", "HF_TOKEN"]
    env_table = Table("Environment Variable", "Value")

    for var in env_vars:
        value = os.getenv(var, "[bold red]NONE[/bold red]")
        env_table.add_row(var, value)

    console.print(env_table)


def save_model_to_safetensors(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model's state dict to a file using safetensors format.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The path to save the model to.

    Returns:
        None
    """

    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)


def load_model_from_safetensors(
    path: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model's state dict from a safetensors file.

    Args:
        path (str): The path to the safetensors file.
        device (str): The device to load the tensors onto. Defaults to CPU.

    Returns:
        Dict[str, torch.Tensor]: The state dict of the PyTorch model.
    """

    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a valid file.")

    state_dict = load_file(path, device=device)
    return state_dict