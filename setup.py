from setuptools import setup, find_packages

setup(
    name='COMP0197',
    version='0.1',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'torch==2.2',
        'torchvision==0.17',
        'torchmetrics',
        'datasets',
        'matplotlib'
    ],
    python_requires='>=3.10',
    packages=find_packages(),
)
