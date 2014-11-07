This is an integration test suite for the sonar-python plugin. It
provide a means to check that the plugin works (or works not) with a
particular SonarQuebe version/setup.


== Preconditions ==
Make sure the following preconditions are met, before running the test suite:

* Python is installed
* behave, the BDD-framework for Python, is installed
* request module is available ('pip install requests' may help)
* Optional: colorama module is installed ('pip install colorama')


== Usage ==
Either install the plugin and startup SonarQube manually and simply run

$ behave

from the project root folder or let the testsuite do the job by
telling it the path to your SQ installation:

$ SONARHOME=/path/to/SonarQuebe behave


== Features to test ==
- Import of the coverage data
- Detection of duplicated python code
- Pylint integration
- Rules from the common-repository
- Rules implemented in the plugin
- Multi-language support
- Multi-module support


== Why behave/python ==
Gherkin as a specification language is quite an obvious choice because
of its clarity. JBehave seems to be big and bloated. Besides, its just
more convenient to write the steps implementation in someting like
python. Hence behave.
