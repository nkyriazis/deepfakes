#!/bin/bash

# install deps
pip install -r requirements.txt

# clone the repo
git clone --depth=1 https://github.com/AliaksandrSiarohin/first-order-model.git

# download the weights from gdrive
fileid="1L8P-hpBhZi8Q_1vP2KlQ4N6dvlzpYBvZ"
filename="vox-adv-cpk.pth.tar"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

https://drive.google.com/file/d/1L8P-hpBhZi8Q_1vP2KlQ4N6dvlzpYBvZ/view?usp=sharing