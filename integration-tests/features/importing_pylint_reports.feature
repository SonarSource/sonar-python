Feature: Importing Pylint reports

  As a SonarQube user
  I want to import the Pylint issues into SonarQube
  In order to have all static code checking results in one place,
     work with them, filter them etc. and derive metrics from them.


  Scenario Outline: Importing Pylint report(s)
    GIVEN the python project "pylint_project"
         AND only Pylint rules are active
    WHEN I run "sonar-runner -X -Dsonar.python.pylint.reportPath=<reportpath>"
    THEN the analysis finishes successfully
         AND the analysis log contains no error or warning messages
         AND the number of violations fed is <violations>

    Examples:
      | reportpath        | violations |
      | pylint-report.txt | 4          |


   Scenario: The reports are missing
     # TODO: this case one should be handled more user friendly
     # by putting an according line into the log

     GIVEN the python project "pylint_project"
     WHEN I run "sonar-runner -X -Dsonar.python.pylint.reportPath=missing"
     THEN the analysis finishes successfully
         AND the analysis log contains no error or warning messages
         AND the number of violations fed is 0


   Scenario Outline: The reports are containing trash only
     GIVEN the python project "pylint_project"
     WHEN I run "sonar-runner -X -Dsonar.python.pylint.reportPath=<reportpath>"
     THEN the analysis finishes successfully
         BUT the analysis log contains a line matching
              """
              .*DEBUG - Cannot parse the line: trash
              """
         AND the number of violations fed is <violations>

     Examples:
      | reportpath  | violations |
      | invalid.txt | 0          |


    Scenario: The report mentions an unknown rule
      GIVEN the python project "pylint_project"
         AND only Pylint rules are active
      WHEN I run "sonar-runner -X -Dsonar.python.pylint.reportPath=rule-unknown.txt"
      THEN the analysis finishes successfully
          BUT the analysis log contains a line matching
               """
               .*WARN.*Pylint rule 'C9999' is unknown in Sonar
               """
          AND the number of violations fed is 0
