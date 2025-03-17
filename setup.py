from setuptools import setup, find_packages


setup(
    name='mtv4d',
    version='0.1.6',
    packages=find_packages(exclude=['*scripts*']),
    description='A 4d data sdk',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ioou',
    author_email='caacabr@163.com',
    url='https://github.com/ioou/mtv4d',
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "matplotlib",
        "pytest",
        "pyyaml",
    ],
)
