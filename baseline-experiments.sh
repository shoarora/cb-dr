python sk_main.py --dataset small --model svm --sess_name baseline-svm-small
echo "finished baseline-svm-small"
python sk_main.py --dataset small --model kernelRR --sess_name baseline-kernelRR-small
echo "finished baseline-kernelRR-small"
python sk_main.py --dataset small --model bayesRR --sess_name baseline-bayesianRR-small
echo "finished baseline-bayesianRR-small"
python main.py --train --dataset small --model lr --cuda --sess_name baseline-lr-small
echo "finished baseline-lr-small"

python sk_main.py --dataset big --model svm --sess_name baseline-svm-big
echo "finished baseline-svm-big"
python sk_main.py --dataset big --model kernelRR --sess_name baseline-kernelRR-big
echo "finished baseline-kernelRR-big"
python sk_main.py --dataset big --model bayesRR --sess_name baseline-bayesianRR-big
echo "finished baseline-bayesianRR-big"
python main.py --train --dataset big --model lr --cuda --sess_name baseline-lr-big
echo "finished baseline-lr-big"
