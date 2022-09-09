from setuptools import find_packages, setup

def readme():
    """Get Readme."""
    with open("README.md", "r") as f:
        content = f.read()
    return content

setup(
    name="darklight",
    version="0.1",
    description="A library for accelerating knowledge distillation training",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://www.jaiyam.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=["numpy","torch"],
)