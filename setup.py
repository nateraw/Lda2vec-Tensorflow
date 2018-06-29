from setuptools import find_packages
from distutils.core import setup

#install_requires = ["spacy==2.0.5","numpy==1.14.3","pandas==0.21.1","tensorflow-gpu==1.5.0","keras==2.1.2","sklearn"]

setup(name="lda2vec",
	  version="0.12.00",
	  description="Tools for interpreting natural language",
	  author="Nathan Raw",
	  author_email="nxr9266@rit.edu",
	  #install_requires=install_requires,
	  packages=find_packages("lda2vec"),
	  url="")
