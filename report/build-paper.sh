#!/bin/sh

# Copyright (C) 2014-2020 by Thomas Auzinger <thomas@auzinger.name>

# Replace the 'x' in the next line with the name of the thesis' main LaTeX document without the '.tex' extension
SOURCE=report

# Build the thesis document
pdflatex $SOURCE
bibtex $SOURCE
bibtex Online
pdflatex $SOURCE
pdflatex $SOURCE
makeindex -t $SOURCE.glg -s $SOURCE.ist -o $SOURCE.gls $SOURCE.glo
makeindex -t $SOURCE.alg -s $SOURCE.ist -o $SOURCE.acr $SOURCE.acn
makeindex -t $SOURCE.ilg -o $SOURCE.ind $SOURCE.idx
pdflatex $SOURCE
pdflatex $SOURCE

echo
echo
echo paper document compiled.
echo Delete build files.
rm -f "$SOURCE".a*
rm -f "$SOURCE".b*
rm -f "$SOURCE".g*
rm -f "$SOURCE".i*
rm -f "$SOURCE".l*
rm -f "$SOURCE".o*
rm -f "$SOURCE".to*
