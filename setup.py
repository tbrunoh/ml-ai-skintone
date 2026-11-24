from setuptools import setup, find_packages

# Read the version from a file (to be auto-updated in GitHub Actions)
with open("VERSION", "r") as f:
    version = f.read().strip()

# Read the requirements from the Requirements.txt file
with open("Requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="mlaist",
    version=version,
    author="Nelson Kigen",
    url="https://github.com/tbrunoh/ml-ai-skintone",
    description="Skin tone classification on the Monk Skin Tone scale, part of Inclusive Skin AI",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    include_package_data=True
)
