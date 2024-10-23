from setuptools import setup, find_packages

setup(
    name='nn_from_scratch',
    version='0.3',
    author='Ankush H V',
    license='MIT',
    description='Neural Networks from Scratch',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'h5py',
        'numpy',
        'torch',
        'gradio',
        'ipywidgets',
        'pillow',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'seaborn',
    ],
)