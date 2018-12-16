#!/bin/bash
set -e
cd ../tex
echo "\begin{verbatim}"  > directory-tree.tex
tree -I '*.log|*.adoc|*.out|*.pre|*.toc|*.aux|_minted*|*~|docs' --charset=ascii ..  >> directory-tree.tex
echo "\end{verbatim}"   >> directory-tree.tex

main_web_file="../webs/horseflies.web"
main_tex_file="horseflies.tex"

if [ $# -eq 0 ]; then 
        # no arguments passed, run in default 
        # mode, by weaving and tangling
	nuweb -r -v $main_web_file             
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	bibtex $main_tex_file                 
	asy *.asy                              
	nuweb -r -v $main_web_file            
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	touch ../horseflies.pdf # update time-stamp on symbolic link

elif [ $1=="--tangle-only" ] ; then 
        # Only extract the source code
        nuweb -t -v $main_web_file    
else 
	echo "Option not recognized."
fi 
