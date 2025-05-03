import tomli
import langtrain as lt

def test_version():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    v = pyproject["project"]["version"]
    assert v == lt.__version__