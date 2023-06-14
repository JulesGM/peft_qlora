#!/usr/bin/env python

from distutils.core import setup

setup(
    name="peft_qlora",
    version="0.1",
    description=(
        "This is just QLora from https://github.com/artidoro/qlora/ "
        "as it's own library, separated from their training script, to allow ",
        "for easier use in other projects."
    ),
    url="https://github.com/JulesGM/peft_qlora",
    packages=["peft_qlora"],
    author="Jules Gagnon-Marchand. See original QLora authors at https://github.com/artidoro/qlora/",
    author_email="jules.gagnonm.alt@gmail.com",
)