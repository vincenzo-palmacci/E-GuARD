from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="eGuard",
    version="0.0.1",
    license="MIT",
    author="Vincenzo Palmacci, Yasmine Nahal",
    author_email="vincenzo.palmacci@univie.ac.at, yasmine.nahal@aalto.fi",
    description=" E-GuARD, a novel framework seeking to address data scarcity and imbalance by integrating self-distillation, active learning, and expert-guided molecular generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vincenzo-palmacci/E-GuARD.git",
    keywords=["ASSAY_INTERFERENCE", "SELF_DISTILLATION", "EXPERT_INPUT", "ACTIVE_LEARNING", "HINTL"],
    packages=find_packages(),
    # package_data={
    #    "eGuard": [
    #        "models/priors/random.prior.new",
    #        "scoring/chemspace/chembl.csv",
    #    ]
    # },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    # install_requires=[
    #    "PyTDC==1.0.7",
    #    "scipy==1.10.1",
    #    "torch==1.12.1",
    #    "fcd-torch==1.0.7",
    #    "click==8.1.7",
    #    "matplotlib==3.9.2",
    #    "jupyter==1.1.1",
    # ],
)
