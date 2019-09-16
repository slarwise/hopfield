#!/bin/bash
latexmk -pdf -shell-escape -silent one_step_error.tex recognizing_digits.tex stochastic_hopfield_network.tex full_code.tex
latexmk -c
