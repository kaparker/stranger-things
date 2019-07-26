#!/bin/bash
for i in $( ls season*/*.txt ); do
    sed -n '1,/'----------'/ p' $i >> final_script.txt
done