

BASE := Version3_report_STA490_mike_kraehenbuehl

# R settings
R := R
ROPTS := --slave --no-save --no-restore

# Compile PDF from TEX, handling bibliography
${BASE}.pdf: ${BASE}.tex
	pdflatex -interaction=nonstopmode $<
	bibtex ${BASE}
	pdflatex -interaction=nonstopmode $<
	pdflatex -interaction=nonstopmode $<

# Generate TEX from RNW
${BASE}.tex: ${BASE}.Rnw
	${R} -e "library('knitr'); knit('$<')"

# Clean auxiliary files
clean:
	rm -f $(addprefix ${BASE}.,aux bbl blg log nav out snm toc vrb tex)

# Clean all generated files, including plots and caches
cleanall: clean
	rm -rf plots/* .cache/*

.PHONY: clean cleanall



# BASE := Version3_report_STA490_mike_kraehenbuehl

# R settings
#R := R
#ROPTS := --slave --no-save --no-restore

#${BASE}.pdf:	${BASE}.tex
#	texi2pdf --batch $<
# texi2pdf is part of the texinfo package: sudo apt-get install texinfo

#${BASE}.tex:	${BASE}.Rnw
#	@echo "library('knitr'); knit('$<')" | ${R} ${ROPTS}

#clean:
#	rm -f $(addprefix ${BASE}.,aux bbl blg log nav out snm toc vrb tex)

#cleanall: clean
#	rm -f plots/* .cache/*

#.PHONY: clean cleanall



