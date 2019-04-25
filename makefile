twins_test:
	python outer_loop.py --model $(model) --data Twins_0
	python outer_loop.py --model $(model) --data Twins_10
	python outer_loop.py --model $(model) --data Twins_20
	python outer_loop.py --model $(model) --data Twins_30
	python outer_loop.py --model $(model) --data Twins_40
	python outer_loop.py --model $(model) --data Twins_50