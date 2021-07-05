from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='evaLEs',
    version='0.1',
    description='Evaluate the Lyapunov Spectrum of a Dynamical System',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='GPLv3',
    packages=find_packages(),
    author='Edoardo Gabrielli',
    author_email='dodogabrie97@live.it',
    keywords=['Lyapunov', 'LE', 'LyapunovExponents','LyapunovSpectrum'],
    url='https://github.com/dodogabrie/evaLEs',
)

install_requires = [
    'numba',
    'numpy', 
    'scipy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
