# SonarPython [![Build Status](https://api.cirrus-ci.com/github/SonarSource/sonar-python.svg?branch=master)](https://cirrus-ci.com/github/SonarSource/sonar-python)  [![Quality Gate](https://next.sonarqube.com/sonarqube/api/project_badges/measure?project=org.sonarsource.python%3Apython&metric=alert_status)](https://next.sonarqube.com/sonarqube/dashboard?id=https://next.sonarqube.com/sonarqube/dashboard?id=org.sonarsource.python%3Apython)

SonarPython is a code analyzer for Python projects. 

## Useful links

* [Project homepage](https://www.sonarsource.com/products/codeanalyzers/sonarpython.html)
* [Issue tracking](http://jira.sonarsource.com/browse/SONARPY)
* [Available rules](https://rules.sonarsource.com/python)
* [SonarSource Community Forum](https://community.sonarsource.com) for feedback

## Building the project

SonarPython embeds [Typeshed](https://github.com/python/typeshed) as a Git submodule. Prior to building the project, you should therefore run `git submodule update --init` to retrieve the corresponding sources.

## License

Copyright 2011-2018 SonarSource.

Licensed under the [GNU Lesser General Public License, Version 3.0](http://www.gnu.org/licenses/lgpl.txt)
