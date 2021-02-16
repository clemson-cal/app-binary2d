#!/bin/bash

pathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
    fi
}

target_dir=cdc_loader/target/release
cd cdc_loader; cargo build --release; cd ..

rslib=$(ls $target_dir | egrep '.so|.dylib')
pylib=$(pwd)/lib/cdc_loader.so
echo $pylib
mkdir -p lib
rm -f $pylib
ln -s $(pwd)/$target_dir/$rslib $pylib
pathadd $(pwd)/lib

echo "adding cdc_loader to your Python path:"
echo "PYTHONPATH=$PYTHONPATH"
