#!/bin/bash
read -n1 -p "Do you want to install project specific packages? [y,n]" doit 
case $doit in  
    y|Y) cat ./relevant_dependencies.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install ;;
    n|N) echo '' ; echo Nothing to install. Done. ;; 
    *) echo '' ; echo Wrong input. Please try again. ;; 
esac