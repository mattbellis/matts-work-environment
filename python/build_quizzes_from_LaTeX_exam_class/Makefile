#! gnumake

#LATEX = xelatex
LATEX = pdflatex

all: quiz.pdf solutions.pdf

quiz.pdf: *.tex Makefile
	$(LATEX) quiz
	$(LATEX) quiz

solutions.pdf: *.tex Makefile
	$(LATEX) solutions
	$(LATEX) solutions

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.nav
	rm -f *.out
	rm -f *.pdf
	rm -f *.snm
	rm -f *.toc
	rm -f *.vrb
	rm -f *.bbl
	rm -f *.blg
