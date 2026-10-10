from setuptools import find_packages, setup

setup(
    name="gb_reviews",
    packages=find_packages(exclude=["tests"]),
    version="0.1.0",
    description="Aspect-level review analysis of Gringo Bandito hot sauces.",
    author="Leonid Shpaner",
    license="MIT",
)
