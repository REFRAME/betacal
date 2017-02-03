from setuptools import setup

setup(name='betacal',
      version='0.2.2',
      description='Beta calibration',
      url='https://github.com/REFRAME/betacal',
      author='tmfilho',
      author_email='tmfilho@gmail.com',
      license='MIT',
      packages=['betacal'],
      install_requires=[
          'numpy',
          'sklearn',
      ],
      zip_safe=False)
