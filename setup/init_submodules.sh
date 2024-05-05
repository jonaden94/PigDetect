# this file does not need to be run by users of this repo!!! Just for initial repository setup (source setup/init_submodules.sh).

git submodule add https://github.com/open-mmlab/mmdetection mmdetection
cd mmdetection
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
cd ..

git submodule add https://github.com/open-mmlab/mmyolo mmyolo
cd mmyolo
git checkout 8c4d9dc503dc8e327bec8147e8dc97124052f693
cd ..