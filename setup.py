from setuptools import find_packages, setup

def readme():
    """Get Readme."""
    with open("README.md", "r") as f:
        content = f.read()
    return content

setup(
    name="darklight",
    version="0.4",
    description="A library for accelerating knowledge distillation training",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dataplayer12/darklight",
    author="Jaiyam Sharma",
    author_email="jaiyamsharma@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=["numpy", "torch", "pycuda"],
)
