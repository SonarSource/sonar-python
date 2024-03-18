#!/bin/bash
# Usage: run from repository root: source ./tools/extract-pytype-annotations/extract_its.sh

echo ""
echo "##########################################################"
echo "### Generating PyType information for checks ###"
echo "##########################################################"
echo ""
dir=python-checks/src/test/resources/checks
python tools/extract-pytype-annotations/extract_info.py $dir python-checks/src/test/resources/checks.json

echo ""
echo "#################################################"
echo "### Generating PyType information for sources ###"
echo "#################################################"
echo ""
for dir in its/sources/*/
do
    dir=${dir%*/}      # remove the trailing "/"
    dir=${dir##*/}     # remove everything before the final "/"
    python tools/extract-pytype-annotations/extract_info.py its/sources/$dir its/ruling/src/test/resources/types/$dir.json
    echo ""
done

echo ""
echo "##########################################################"
echo "### Generating PyType information for sources extended ###"
echo "##########################################################"
echo ""
for dir in its/sources_extended/[a-z]*/
do
    dir=${dir%*/}      # remove the trailing "/"
    dir=${dir##*/}     # remove everything before the final "/"
    python tools/extract-pytype-annotations/extract_info.py its/sources_extended/$dir its/ruling/src/test/resources/types_extended/$dir.json
    echo ""
done
