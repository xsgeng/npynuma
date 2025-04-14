import numpy as np
from setuptools import Extension, setup

extensions = [
    Extension(
        name="npynuma.numa", 
        sources=["npynuma/numa.c"],
        libraries=['numa'],
        include_dirs=[np.get_include()]
    ),
]

setup(
    name="npynuma",
    ext_modules=extensions
)