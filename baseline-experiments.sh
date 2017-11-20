python sk_main.py --dataset small --model svm --sess_name baseline-svm-small
python sk_main.py --dataset big --model svm --sess_name baseline-svm-big

python sk_main.py --dataset small --model kernelRR --sess_name baseline-kernelRR-small
python sk_main.py --dataset big --model kernelRR --sess_name baseline-kernelRR-big

python sk_main.py --dataset small --model bayesianRR --sess_name baseline-bayesianRR-small
python sk_main.py --dataset big --model bayesianRR --sess_name baseline-bayesianRR-big

python main.py --train --dataset small --model lr --cuda --sess_name baseline-lr-small
python main.py --train --dataset big --model lr --cuda --sess_name baseline-lr-big
