# Makefile 

# File lists:
RNW = $(wildcard *.Rnw)
R = $(patsubst %.Rnw,%.R,$(RNW))
TEX = $(patsubst %.Rnw,%.tex,$(RNW))

# Master file:
MASTER = msc_report

# Meta rules:
.Rnw.tex:
	Rscript -e "library(knitr); knitr::knit('$<','$@')"

.Rnw.R:
	Rscript -e "library(knitr); knitr::purl('$<',documentation=0L)"


# below changed from pdflatex to xelatex to allow for polish characters
.tex.pdf:
	latexmk -bibtex -synctex=1 -pdf  -use-make -halt-on-error \
	 -pdflatex="xelatex" -silent $(MASTER).tex  

# Conversion rules:
all: 	$(MASTER).pdf 

$(MASTER).pdf: $(TEX) $(MASTER).tex 

short: $(TEX)
	pdflatex $(MASTER).tex 	

clean:
	rm -f *~   *.out Rplots.pdf comment.cut \
              *.idx *.ilg *.brf *.blg *.spl  $(MASTER).dvi \
              *.backup *.toc *.fls  *fdb_latexmk *.synctex.gz *-concordance.tex 

cleanall: clean
	rm -f *.aux *.log *.ind *.ist
	rm -f figure/ch??_fig*.pdf
	rm -f chapter??.tex chapter??.R $(MASTER).bbl
	rm -rf cache/*.* cache/__packages .Rhistory

# Some technical details
.SUFFIXES: .Rnw .R .tex .pdf
.SILENT: *.pdf *.tex	
.PHONY: all short clean cleanall


# 2018 Reinhard Furrer, Florian Gerber

