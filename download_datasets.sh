echo "THESE DATASETS ARE HUGE GET READY"

mkdir data
cd data

wget http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170331.zip
wget http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170630.zip

wget https://nlp.stanford.edu/software/GloVe-1.2.zip

unzip clickbait17-train-170331.zip
unzip clickbait17-train-170630.zip

unzip GloVe-1.2.zip

rm clickbait17-train-170331.zip
rm clickbait17-train-170630.zip

rm GloVe-1.2.zip

mv clickbait17-train-170331 cb-small
mv clickbait17-validation-170630 cb-big

mv GloVe-1.2 glove

cd ../
