# ── Compile the LaTeX paper ───────────────────────────────────────────────────
# Requirements: pdflatex, bibtex (part of any TeX distribution)
#   Windows: install MiKTeX or TeX Live
#   Conda:   conda install -c conda-forge texlive-core

PDF = main.pdf
SRC = main.tex

all: $(PDF)

$(PDF): $(SRC) references.bib sections/*.tex
	pdflatex  -interaction=nonstopmode $(SRC)
	bibtex    main
	pdflatex  -interaction=nonstopmode $(SRC)
	pdflatex  -interaction=nonstopmode $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view: $(PDF)
	start $(PDF)   # Windows
	# open $(PDF)  # Mac
	# xdg-open $(PDF) # Linux
