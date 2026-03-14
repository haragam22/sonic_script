from setuptools import setup, find_packages

setup(
    name="white-box-composer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sounddevice",
        "soundfile",
        "lark",
    ],
    # --- THIS IS THE MAGIC PART ---
    entry_points={
        "console_scripts": [
            "sonic = compiler.cli:main",
        ],
    },
    include_package_data=True, # Important for including grammar.lark
)