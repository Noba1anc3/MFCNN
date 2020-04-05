[tox]
envlist = py36

# [travis]
# python =
#     3.6: py36
#     3.5: py35
#     3.4: py34
#     2.7: py27

# [testenv:flake8]
# basepython = python
# deps = flake8
# commands = flake8 v_text_mountain_grpc

[testenv]
setenv =
    PYTHONPATH = {toxinidir}

commands = python setup.py test
