# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv
brew update
brew install pyenv

# Install Python 3.12.2 using pyenv
pyenv install 3.12.2

# Set Python 3.12.2 as the global version
pyenv global 3.12.2

# Verify the installation
python --version  # Should output Python 3.12.2

# Navigate to your project directory
cd /path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install necessary packages
pip install pandas seaborn matplotlib numpy networkx plotly

# Generate the requirements.txt file
pip freeze > requirements.txt

# Deactivate the virtual environment when done
deactivate
