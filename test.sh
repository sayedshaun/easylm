echo "Installing the package..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e .

echo "Running tests with pytest..."
pytest tests/

echo "Removing the package..."
pip uninstall -y langtrain
rm -rf tests/__pycache__
rm -rf tests/pretrained_model
rm -rf .pytest_cache
rm -rf tests/test_weights

# Remove all local build files
rm -rf build
rm -rf dist
rm -rf src/langtrain.egg-info