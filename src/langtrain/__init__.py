import tomli
import os

def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
        return pyproject_data["project"]["version"]

if "__version__" not in globals():
    __version__ = get_version()

from langtrain import config, data, model, trainer, tokenizer, utils