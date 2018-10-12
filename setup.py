#!/usr/bin/env python

if __name__ == "__main__":
    from numpy.distutils.core import Extension, setup
    from os.path import join
    setup(
        name = 'pyshesd',
        version = '0.1',
        author = 'Jul Wong',
        author_email = 'zht.huang@gmail.com',
        packages=['pyshesd'],
        ext_modules=[
            Extension(name = 'pyshesd._stl', sources = [join('pyshesd', f) for f in ['f_stl.pyf', 'stl.f']])
        ],
        py_modules=['pyshesd.shesd']
    )
