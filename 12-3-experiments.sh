echo 'small-svm-text-45'
python sk_main.py --dataset small --model svm --choice text --freq_floor 45 --sess_name small-svm-text-45

echo 'small-svm-text-35'
python sk_main.py --dataset small --model svm --choice text --freq_floor 35 --sess_name small-svm-text-35

echo 'small-svm-text-25'
python sk_main.py --dataset small --model svm --choice text --freq_floor 25 --sess_name small-svm-text-25

echo 'big-svm-text-100'
python sk_main.py --dataset big --model svm --choice text --freq_floor 100 --sess_name big-svm-text-100

echo 'small-svm-post-3'
python sk_main.py --dataset small --model svm --choice post --freq_floor 3 --sess_name small-svm-post-3
