import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SELeCT",
    version="0.0.2",
    author="",
    author_email="",
    description="real-time lifelong machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['SELeCT'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
