import tomli
import os

def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "../../pyproject.toml")
    pyproject_path = os.path.abspath(pyproject_path)
    if not os.path.exists(pyproject_path):
        raise FileNotFoundError(f"pyproject.toml not found at: {pyproject_path}")
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)["project"]["version"]

__version__ = get_version()

from langtrain import config, data, model, trainer, tokenizer, utils, nn