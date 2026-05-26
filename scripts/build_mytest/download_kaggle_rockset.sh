mkdir -p ~/project/build_mytest/data

curl -L -o ~/project/build_mytest/data/rocks-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/neelgajare/rocks-dataset

curl -L -o ~/project/build_mytest/data/rock-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/salmaneunus/rock-classification

unzip ~/project/build_mytest/data/rocks-dataset.zip -d ~/project/build_mytest/data/rocks-dataset/
unzip ~/project/build_mytest/data/rock-classification.zip -d ~/project/build_mytest/data/rock-classification/

rm ~/project/build_mytest/data/rocks-dataset.zip
rm ~/project/build_mytest/data/rock-classification.zip
