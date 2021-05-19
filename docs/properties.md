# Internal analysis properties of the Python analyzer

``sonar.python.performance.measure``: Boolean; if set to true, will enable performance monitoring of the analyzer (default: `false`).

``sonar.python.performance.measure.path``: Path where the performance monitoring report will be saved, relative to the work dir (default: `sonar-python-performance-measure.json`).

``sonar.internal.analysis.failFast``: Boolean; if set to true, exceptions will fail the analysis (default: `false`).

``sonar.python.sonarlint.maxlines``: Maximum number of lines in a project above which the project symbol table won't be computed in SonarLint context (default: `150000`).
