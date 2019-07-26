#!/bin/bash
for i in $( ls season3/*.txt ); do
    sed -n '1,/'----------'/ p' $i >> season3_script.txt
done