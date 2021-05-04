from setuptools import setup, find_packages

setup(name='analogreadout',
      version='0.1',
      description='Code for interfacing with instruments for data taking with MKIDs',
      url='https://github.com/zobristnicholas/analogreadout',
      author='Nicholas Zobrist',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=["pyvisa",
                        "PyDAQmx",
                        "pymeasure",
                        "numpy",
                        "scipy",
                        "yaml",
                        "future",
                        "scikit-image"],
      scripts=['sweep_gui.py', 'pulse_gui.py'],
      zip_safe=False)
