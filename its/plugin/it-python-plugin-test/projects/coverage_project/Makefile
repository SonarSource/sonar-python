all:
	-nosetests --with-coverage --cover-erase --cover-branches --cover-package=src src/test_unit.py
	python-coverage xml -o ut-coverage.xml
	-nosetests --with-coverage --cover-erase --cover-branches --cover-package=src src/test_integration.py
	python-coverage xml -o it-coverage.xml
	# TODO: merge UT and IT to overall coverage
