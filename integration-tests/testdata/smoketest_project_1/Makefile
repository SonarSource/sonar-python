

all:
	coverage erase
	-coverage run --branch --source=src --omit='qtest_*' `which nosetests` -v --with-xunit
	coverage xml -i
	nosetests some/deeper/path --with-xunit --xunit-file=nosetest_not_relative_to_root.xml
