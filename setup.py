from setuptools import setup, find_packages

setup(
    name='rioss_prep',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[],
    author='Pedro Meirelles',
    author_email='goes.phmeirelles@gmail.com',
    description='RIOSS data setup package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PedroHMG/rioss_data_processing',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
)