from setuptools import setup
from son_funcs import __version__
setup(
    name='py_son',
    version=__version__,

    url='https://github.com/Marwolfer/index.html',
    author='Marco Wolfer',

    py_modules=['py_son'],
    install_requires=[
        'matplotlib',
        "numpy",
        "scipy",
        "mingus",
        "pydub"]
)