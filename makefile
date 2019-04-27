twins_test:
	python outer_loop.py --model $(model) --data Twins_0
	python outer_loop.py --model $(model) --data Twins_10
	python outer_loop.py --model $(model) --data Twins_20
	python outer_loop.py --model $(model) --data Twins_30
	python outer_loop.py --model $(model) --data Twins_40
	python outer_loop.py --model $(model) --data Twins_50

syn_test:
	python outer_loop.py --model $(model) --data Synthetic_1000
	python outer_loop.py --model $(model) --data Synthetic_3000
	python outer_loop.py --model $(model) --data Synthetic_5000
	python outer_loop.py --model $(model) --data Synthetic_10000
	python outer_loop.py --model $(model) --data Synthetic_30000
