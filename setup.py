from setuptools import setup, find_packages
import os

VERSION = {}
with open("lmpl/version.py") as version_file:
    exec(version_file.read(), VERSION)

# Load requirements.txt with a special case for allennlp so we can handle
# cross-library integration testing.
with open("requirements.txt") as requirements_file:
    import re

    def fix_url_dependencies(req: str) -> str:
        """Pip and setuptools disagree about how URL dependencies should be handled."""
        m = re.match(
            r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
        )
        if m is None:
            return req
        else:
            return f"{m.group('name')} @ {req}"

    install_requirements = []
    for line in requirements_file:
        line = line.strip()
        if line.startswith("#") or len(line) <= 0:
            continue
        install_requirements.append(line)

    install_requirements = [fix_url_dependencies(req) for req in install_requirements]

setup(
    name="LMPL",
    version=VERSION["VERSION"],
    description=("Language Model as Policy Learning Library"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning language modeling",
    url="https://github.com/kushalarora/lmpl",
    author="Kushal Arora",
    author_email="kushal.arora@mail.mcgill.ca",
    license="MIT",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"],),
    install_requires=[
      "allennlp>=1.0.0, <1.1.0",
      "allennlp-models>=1.0.0, <1.1.0",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
