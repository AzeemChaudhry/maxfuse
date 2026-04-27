from setuptools import setup, find_packages

setup(
    name="maxfuse",
    version="1.0.0",
    description="Multimodal Attention-based Cross-modal Fusion for Malware Classification",
    author="Afan Atif, Azeem Chaudhry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
