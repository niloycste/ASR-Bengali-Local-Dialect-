# ── Compile the LaTeX paper ───────────────────────────────────────────────────
# Requirements: lualatex, bibtex (part of any TeX distribution) + a Bengali font
#   (Noto Serif Bengali). The paper contains inline Bengali Unicode, so it must
#   be built with lualatex (or xelatex), NOT pdflatex.
#   Windows: install MiKTeX or TeX Live; install "Noto Serif Bengali".
#   Conda:   conda install -c conda-forge texlive-core

PDF = main.pdf
SRC = main.tex

all: $(PDF)

$(PDF): $(SRC) references.bib sections/*.tex
	lualatex  -interaction=nonstopmode $(SRC)
	bibtex    main
	lualatex  -interaction=nonstopmode $(SRC)
	lualatex  -interaction=nonstopmode $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view: $(PDF)
	start $(PDF)   # Windows
	# open $(PDF)  # Mac
	# xdg-open $(PDF) # Linux
