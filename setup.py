import os

from os.path import join

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension(
    "cpp_balltree",
    [join("_cpp_balltree","ball_tree.pyx")],
    language = 'c++',
    depends=[join('_cpp_balltree', 'BallTree.h'),
             join('_cpp_balltree', 'BallTreePoint.h')],
    include_dirs=[numpy.get_include()]
    )

setup(cmdclass = {'build_ext': build_ext},
      name='ball_tree',
      version='1.0',
      ext_modules=[ext],
      )

ext2 = Extension(
    "npy_balltree",
    [join("_npy_balltree","ball_tree.pyx")],
    )

setup(cmdclass = {'build_ext': build_ext},
      name='ball_tree',
      version='1.0',
      ext_modules=[ext2],
      )

