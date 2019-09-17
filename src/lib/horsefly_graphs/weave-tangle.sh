#!/bin/bash
set -e

main_web_file="horsefly_graphs.web"
main_tex_file="horsefly_graphs.tex"

if [ $# -eq 0 ]; then 
        # no arguments passed, run in default 
        # mode, by weaving and tangling
	nuweb -r -v $main_web_file             
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	#asy -vv *.asy                              
	bibtex $main_tex_file                 
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  
	nuweb -r -v $main_web_file            
	lualatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file  

	# Compile the source code. This will only be useful if I will be using 
	# C++/Haskell for writing some performance sensitive routines. 
	# EMPTY------------------------------- for now

elif [ $1=="--tangle-only" ] ; then 
        # Only extract the source code
        nuweb -t -v $main_web_file    
else 
	echo "Option not recognized."
fi 
