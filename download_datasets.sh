echo "THESE DATASETS ARE HUGE GET READY"

mkdir data
cd data

wget http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170331.zip
wget http://www.uni-weimar.de/medien/webis/corpora/corpus-webis-clickbait-17/clickbait17-train-170630.zip

wget http://nlp.stanford.edu/data/glove.6B.zip

unzip clickbait17-train-170331.zip
unzip clickbait17-train-170630.zip

unzip glove.6B.zip

rm clickbait17-train-170331.zip
rm clickbait17-train-170630.zip

rm glove.6B.zip

mv clickbait17-train-170331 cb-small
mv clickbait17-validation-170630 cb-big

mv glove.6B glove

cd ../
