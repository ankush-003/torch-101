# Publishing Python Packages

`PyPI` is the Python Package Index, a repository of software for the Python programming language. PyPI helps you find and install software developed and shared by the Python community. You can publish your packages on PyPI so that other Python developers can install them using `pip`.

## 📦 Create a package

1. Create a new directory for your package:

```bash
mkdir mypackage
```

```bash
 mypackage
    | ├── mypackage
    | │   ├── __init__.py
    | │   └── mymodule.py
    | ├── LICENSE
    | ├── README.md
    | └── setup.py
```

2. Install the necessary tools:

```bash
pip install setuptools wheel twine==5.1.1
```

3. Create a `setup.py` file:

```python
from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
)
```

4. Create distribution archives:

```bash
python setup.py sdist bdist_wheel
```

## 🚀 Upload package to PyP

1. Create an account on [PyPI](https://pypi.org/account/register/).
2. Get your API token from [here](https://pypi.org/manage/account/).
3. Upload your package:

```bash
twine upload dist/*
```

it willl ask for your api token, paste it and hit enter.

> You can now install your package using `pip install mypackage`.