import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reprieve",
    version="0.0.1",
    author="Will Whitney",
    author_email="wfwhitney@gmail.com",
    description="A library for evalating representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/willwhitney/reprieve",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
