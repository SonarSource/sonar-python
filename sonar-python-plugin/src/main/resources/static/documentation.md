---
title: Python
key: python
---

## Supported Versions

| Version           | Status                                               |
|-------------------|------------------------------------------------------|
| Python 3.0 - 3.11 | Fully Supported                                      |
| Python 2.x        | Fully Supported                                      |
| IPython 7.x       | Partially Supported (Jupyter Notebooks context only) |

## Language-Specific Properties

Discover and update the Python-specific [properties](/analysis/analysis-parameters/) in: <!-- sonarcloud -->Project <!-- /sonarcloud --> **[Administration > General Settings > Python](/#sonarqube-admin#/admin/settings?category=python)**.

## Handling Project Python Version

Python code is analyzed by default as compatible with python 2 and python 3. Some issues will be automatically silenced to avoid raising False Positives.
In order to get a more precise analysis you can specify the python versions your code supports via the `sonar.python.version` parameter.

Accepted format are a comma separated list of versions having the format "X.Y"

Examples: 
  * `sonar.python.version=2.7`
  * `sonar.python.version=3.8` 
  * `sonar.python.version=2.7, 3.7, 3.8, 3.9`

<!-- sonarqube -->
## Custom Rules

### Overview

The Python analyzer parses the source code, creates an Abstract Syntax Tree (AST) and then walks through the entire tree. A coding rule is a visitor that is able to visit nodes from this AST.

As soon as the coding rule visits a node, it can navigate its children and log issues if necessary.

### Writing a Plugin

Custom rules for Python can be added by writing a SonarQube Plugin and using Python analyzer APIs.
Here are the steps to follow:

#### Create a SonarQube Plugin

* create a standard SonarQube plugin project.
* attach this plugin to the SonarQube Python analyzer through the `pom.xml`:
  * add the dependency to the Python analyzer.
  * add the following line in the sonar-packaging-maven-plugin configuration.
  ```
  <requirePlugins>python:2.0-SNAPSHOT</requirePlugin>
  ```
* implement the following extension points:
  * [Plugin](https://javadocs.sonarsource.org/latest/org/sonar/api/Plugin.html)
  * [RulesDefinition](https://javadocs.sonarsource.org/latest/org/sonar/api/server/rule/RulesDefinition.html) and [PythonCustomRuleRepository](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/PythonCustomRuleRepository.java), which can be implemented by a single class, to declare your custom rules.
* declare the RulesDefinition as an extension in the Plugin extension point.

#### Implement a Rule

* create a class that will hold the implementation of the rule, it should:
  * extend `PythonCheckTree` or `PythonSubscriptionCheck`.
  * define the rule name, key, tags, etc. with Java annotations.
* declare this class in the `RulesDefinition`.

### Example Plugin

To get started a sample plugin can be found here: [python-custom-rules](https://github.com/SonarSource/sonar-custom-rules-examples/tree/master/python-custom-rules).

####  Implementation Details

**Using `PythonCheckTree`**

To explore a part of the AST, override a method from the PythonCheckTree. For example, if you want to explore "if statement" nodes, override [PythonCheckTree#visitIfStatement](https://github.com/SonarSource/sonar-python/blob/39b6126e9fdef42b93004cf6cc5818e861051334/python-frontend/src/main/java/org/sonar/plugins/python/api/tree/BaseTreeVisitor.java#L56) method that will be called each time an [ifStatement](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/tree/IfStatement.java) node is encountered in the AST.

![](/images/exclamation.svg) When overriding a visit method, you must call the super method in order to allow the visitor to visit the children of the node.

**Using `PythonSubscriptionCheck`**

To explore a part of the AST, override [`PythonSubscriptionCheck#initialize`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/SubscriptionCheck.java#L26) and call the [`SubscriptionCheck.Context#registerSyntaxNodeConsumer`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/SubscriptionCheck.java) with the [`Tree#Kind`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/tree/Tree.java#L42) of node you want to visit. For example, if you want to explore "if statement" you should register to the kind [`Tree#Kind#IF_STATEMENT`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/tree/Tree.java#L97) and then provide a lambda that will consume a [`SubscriptionContext`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/SubscriptionContext.java#L27) to act on such ndoes.

**Create Issues**

From the check, issue can be created by calling [`SubscriptionContext#addIssue`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/SubscriptionContext.java#L30) method or  [`PythonCheckTree#addIssue`](https://github.com/SonarSource/sonar-python/blob/master/python-frontend/src/main/java/org/sonar/plugins/python/api/PythonCheckTree.java#L36) method.

**Testing Checks**

To test custom checks you can use method [`PythonCheckVerifier#verify`](https://github.com/SonarSource/sonar-python/blob/master/python-checks-testkit/src/main/java/org/sonar/python/checks/utils/PythonCheckVerifier.java). Don't forget to add the testkit dependency to access this class from your project : 
  ```
    <dependency>
	  <groupId>org.sonarsource.python</groupId>
	  <artifactId>python-checks-testkit</artifactId>
	  <version>${project.version}</version>
	  <scope>test</scope>
    </dependency>
  ```

You should end each line having an issue with a comment in the following form:

```
# Noncompliant {{Message}}
```

Comment syntax is described [here](https://github.com/SonarSource/sonar-analyzer-commons/blob/master/test-commons/README.md).

<!-- /sonarqube -->

## Related Pages
* [Importing External Issues](/analysis/external-issues/) ([Pylint](https://pylint.pycqa.org/), [Bandit](https://github.com/PyCQA/bandit/blob/master/README.rst), [Flake8](https://flake8.pycqa.org/en/latest/))
* [Test Coverage & Execution](/analysis/coverage/) (the [Coverage Tool](https://coverage.readthedocs.io/en/stable/api.html) provided by [Ned Batchelder](https://nedbatchelder.com/), [Nose](https://nose.readthedocs.io/), [pytest](https://docs.pytest.org/))
