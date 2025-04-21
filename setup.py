from setuptools import setup, find_packages

setup(
    name="seatracer",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'seatracer': ['data/*'],
    },
    install_requires=[
        "pandas",
        "numpy",
        "statsmodels",
        "scipy",
        "streamlit",
        # Add other dependencies
    ],
    author="Your Name",
    description="Rowing lineup and seat racing analysis tool",
)