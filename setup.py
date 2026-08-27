from setuptools import setup, find_packages

setup(
    name="seatracer",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'seatracer': ['data/*', 'erg_data/*'],
    },
    install_requires=[
        "nicegui",
        "plotly",
        "pandas",
        "numpy",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "networkx",
        "patsy",
    ],
    author="Your Name",
    description="Rowing lineup and seat racing analysis tool",
)