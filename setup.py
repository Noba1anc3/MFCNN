#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["pip==21.1", "Click==7.0", "GPUtil==1.4.0", "Jinja2==2.10.1", "Keras==2.2.4", "Keras-Applications==1.0.8", "Keras-Preprocessing==1.1.0", "Markdown==3.1.1", "MarkupSafe==1.1.1", "Pillow==6.0.0", "Polygon3==3.0.8", "PyWavelets==1.0.3", "PyYAML==5.1", "Werkzeug==0.15.4", "absl-py==0.7.1", "argh==0.26.2", "arrow==0.13.2", "astor==0.8.0", "baidu-aip==2.2.13.0", "bert-serving-client==1.9.1", "bert-serving-server==1.9.1", "binaryornot==0.4.4", "boto==2.49.0", "boto3==1.9.169", "botocore==1.12.169", "bumpversion==0.5.3", "certifi==2019.3.9", "chardet==3.0.4", "cookiecutter==1.6.0", "coverage==4.5.1", "cycler==0.10.0", "decorator==4.4.0", "distconfig==0.1.0", "docutils==0.14", "flake8==3.5.0", "future==0.17.1", "gast==0.2.2", "gensim==3.7.3", "googleapis-common-protos==1.5.10", "grpcio==1.20.1", "grpcio-tools==1.20.1", "h5py==2.9.0", "idna==2.8", "imageio==2.5.0", "jinja2-time==0.2.0", "jmespath==0.9.4", "joblib==0.13.2", "kiwisolver==1.1.0", "logzero==1.5.0", "mock==3.0.5", "nltk==3.4.3", "numpy==1.16.3", "opencv-python==4.1.0.25", "pathlib==1.0.1", "pathtools==0.1.2", "poyo==0.4.2", "prometheus-client==0.6.0", "protobuf==3.7.1", "protoc-gen-swagger==0.1.0", "pyparsing==2.4.0", "python-consul==1.1.0", "python-dateutil==2.8.0", "pytoml==0.1.20", "pyzmq==18.0.1", "requests==2.22.0", "s3transfer==0.2.1", "scikit-image==0.14.3", "scikit-learn==0.21.2", "scipy==1.3.0", "setuptools==40.8.0", "six==1.12.0", "sklearn==0.0", "smart-open==1.8.4", "Sphinx==1.8.1", "tensorboard==1.13.1", "tensorflow==1.13.1", "tensorflow-estimator==1.13.0", "termcolor==1.1.0", "torch==1.1.0", "torchvision==0.3.0", "tox==3.5.2", "twine==1.12.1", "ujson==1.35", "urllib3==1.25.2", "videt-dar-tools==0.1.3", "videt-grpc-interceptor==0.1.0", "videt-idl==0.17.2", "videt-protos==2.0.1", "videt-py-conf==1.0.0", "vyper-config==0.3.3", "watchdog==0.9.0", "wheel==0.33.4", "whichcraft==0.5.2", "yacs==0.1.6", "matplotlib==3.1.0", "networkx==2.3"]

setup_requirements = []

test_requirements = []

setup(
    author="ZHANG XUANRUI",
    author_email='xuanrui.zhang@videt.cn',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Text Mountain for text detection GPC",
    entry_points={
        'console_scripts': [
            "videt-thai_customs=v_thai_customs.main:run",
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords="videt-thai-customs",
    name="videt-thai-customs",
    packages=find_packages(include=["v_thai_customs","v_thai_customs.*","v_thai_customs.*.*"]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url="https://git.videt.cn/zxrtt/videt-thai-customs",
    version='0.1.0',
    zip_safe=False,
)
