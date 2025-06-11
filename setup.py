from setuptools import setup, find_packages

# Read in the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chipstb",
    version="2024.11.30",
    author="In Jun Park, Kamal Choudhary",
    author_email="kamal.choudhary@nist.gov",
    description="CHIPSTB: A tight binding analysis toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/chipstb",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "pydantic_settings",
        "pandas",  # Any other dependencies can be added here
        # "alignn",  # Uncomment if alignn becomes a required dependency
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run_chipstb=chipstb.run_chipstb:main",  # Allows `run_chipsff` to be called from the CLI
        ],
    },
    include_package_data=True,
)

