from setuptools import find_packages, setup


setup(
    name="smartchunk",
    version="0.1.0",
    description="Self-describing text chunks with metadata for RAG applications",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="smartchunk contributors",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "semantic": ["sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.0.0"],
    },
    python_requires=">=3.9",
)
