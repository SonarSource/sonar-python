This directory contains integration tests: these tests run the sonar-python plugin inside
a SonarQube instance and check how it behaves with some given projects and configurations.

There are 2 sets of integration tests:
* plugin: checks metrics, import of test results, coverage report, pylint report
* ruling: checks the results of rules against some real-world python code

To run integration tests, you will first need to build the sonar-python plugin:
* mvn clean package

To run the "ruling" tests, you also need to execute the following commands:
* git submodule init
* git submodule update

Then, you can run:
* cd its/plugin
* mvn test -Dsonar.runtimeVersion=LATEST_RELEASE

Or, for the "ruling" tests:
* cd its/ruling
* mvn test -Dsonar.runtimeVersion=LATEST_RELEASE
