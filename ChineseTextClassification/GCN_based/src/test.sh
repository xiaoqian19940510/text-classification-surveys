# run ML model
python ./run_ml.py

# run DL model
for file in `ls ../config/ |grep "\.json$"`
do
	path="../config/"$file
	python ./run.py --config $path
done