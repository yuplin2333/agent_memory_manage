export PYTHONPATH=$PWD

python agentdriver/evaluation/evaluation.py --metric $1 --success_threshold $2 --result_file $3
