# Code Quality and Security for Python [![Build Status](https://api.cirrus-ci.com/github/SonarSource/sonar-python.svg?branch=master)](https://cirrus-ci.com/github/SonarSource/sonar-python)  [![Quality Gate](https://next.sonarqube.com/sonarqube/api/project_badges/measure?project=org.sonarsource.python%3Apython&metric=alert_status)](https://next.sonarqube.com/sonarqube/dashboard?id=https://next.sonarqube.com/sonarqube/dashboard?id=org.sonarsource.python%3Apython)
#### Python analyzer for SonarQube, SonarCloud and SonarLint

## Useful links

* [Project homepage](https://www.sonarsource.com/products/codeanalyzers/sonarpython.html)
* [Issue tracking](http://jira.sonarsource.com/browse/SONARPY)
* [Available rules](https://rules.sonarsource.com/python)
* [SonarSource Community Forum](https://community.sonarsource.com) for feedback

## Building the project

Maven build is generating protobuf messages for Typeshed symbols from a python script (see [typeshed_serializer](https://github.com/SonarSource/sonar-python/tree/master/python-frontend/typeshed_serializer)).
In order for it to work properly it needs to have Python runtime and [Typeshed](https://github.com/python/typeshed) available.

### Prerequisites
- Run `git submodule update --init` to retrieve [Typeshed](https://github.com/python/typeshed) as a Git submodule
- Make sure to have Python 3.9 and [tox](https://tox.readthedocs.io/en/latest/) installed and available in PATH

### Profiles

- `mvn clean install` : execute full build, run tests for [typeshed_serializer](https://github.com/SonarSource/sonar-python/tree/master/python-frontend/typeshed_serializer)
- `mvn clean install -DskipTypeshed`: avoid running [typeshed_serializer](https://github.com/SonarSource/sonar-python/tree/master/python-frontend/typeshed_serializer) tests and build only Java maven modules

## License

Copyright 2011-2021 SonarSource.

Licensed under the [GNU Lesser General Public License, Version 3.0](http://www.gnu.org/licenses/lgpl.txt)
