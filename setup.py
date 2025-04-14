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
    author_email="nkigen@melaninluxe.ai",
    url="https://github.com/melaninluxe/ml-ai-skintone",
    description="Reusable PyTorch models for projects within Melanin Luxe AI organization",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    include_package_data=True
)
