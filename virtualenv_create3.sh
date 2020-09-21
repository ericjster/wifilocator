#!/bin/bash

# Note that virtualenv guides suggest updating pip, and using the system pip to
# bootstrap a user installation of pip.
#
# Here is a great guide to virtualenv, pyenv, pyenv-virtualenv,
# virtualenvwrapper, pyenv-virtualenvwrapper, pipenv.
# https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe
# One person says virtualenv is the community standard.
# One person says avoid virtualenv after Python 3.3+ and instead use venv from standard shipped library.

# Upgrade to latest pip, to bootstrap a user install of pip.
# At the time of writing: 
#    pip 19.1 from ~/Library/Python/3.7/lib/python/site-packages/pip (python 3.7)

usage_note() {
    echo ""
    echo "To start using the virtual environment, it needs to be activated:"
    echo "  source venv/bin/activate"
    echo "To deactiveate, run:"
    echo "  deactivate"
}

which python3 >/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing python3."
    exit 1
fi

which python3 | grep venv/bin/python3
if [ $? -eq 0 ]; then
    echo "Error: You need to call 'deactivate' first before running this script."
    exit 1
fi

if [ -f venv/bin/python2 ]; then
    echo "Error: Venv uses python2 and we want to build python3."
    exit 1
fi

set -e

# For python3, venv is the preferred way to create and manage virtual environments.
# Note that "-m venv" means run the "venv" package, and the last "venv" is the name of the directory to create.
# Note we cannot call virtualenv, because it can have an invalid shebang.

echo "Creating virtualenv 'venv'."
python3 -m venv venv

echo "Inside this script, using the virtual environment."
echo ""

# Only applies inside this shell script.
source venv/bin/activate

# Note we operate on the venv version of pip, so we don't want --user.
echo "pip install --upgrade pip"
which pip
pip --version
pip install --upgrade pip
which pip
pip --version

if [ -f requirements.txt ]; then
    echo ""
    echo "pip install -r requirements.txt"
    pip install -r requirements.txt
fi

usage_note