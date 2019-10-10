FILE_ID=1XNIq7byciKgrn011jLBggd2g79jKX4uD
FILE_NAME=ECO_Lite_rgb_model_Kinetics.pth.tar
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
mv ${FILE_NAME} ../datasets/chapter9