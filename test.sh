echo "Installing the package in editable mode..."
pip install -e .

echo "Running tests with pytest..."
pytest tests/

echo "Removing the package..."
rm -rf dist/*.tar.gz
rm -rf dist/*.whl
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf tests/pretrained_model
rm -rf .pytest_cache