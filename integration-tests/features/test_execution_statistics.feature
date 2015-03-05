Feature: Providing test execution numbers

  As a SonarQube user
  I want to import my Python tests execution numbers into SonarQube
  In order to be able to use diverse SonarQube features such as:
    - continuously monitor the number of skipped/failed/errorer tests,
      overall test number, test execution time and test success ratio
    - define Quality Gates on top of this metrics
    - have a means of finding the failed tests quickly without having
      to resort to reading of long and boring log files
    - ...

  Scenario: Importing a set of valid test reports in detailled mode
      GIVEN the python project "nosetests_project"

      WHEN I run "sonar-runner"

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric               | value |
              | tests                | 3.0   |
              | test_failures        | 1.0   |
              | test_errors          | 1.0   |
              | skipped_tests        | 1.0   |
              | test_success_density | 33.3  |
              | test_execution_time  | 1.0   |


  Scenario: Importing a set of valid test reports in simple mode
      GIVEN the python project "nosetests_project"

      WHEN I run "sonar-runner -Dsonar.python.xunit.skipDetails=true"

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric               | value |
              | tests                | 3.0   |
              | test_failures        | 1.0  |
              | test_errors          | 1.0   |
              | skipped_tests        | 1.0   |
              | test_success_density | 33.3  |
              | test_execution_time  | 1.0   |


  Scenario Outline: Test reports are missing or cannot be found
      GIVEN the python project "nosetests_project"

      WHEN I run <command>

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric               | value |
              | tests                | None  |
              | test_failures        | None  |
              | test_errors          | None  |
              | skipped_tests        | None  |
              | test_success_density | None  |
              | test_execution_time  | None  |

      Examples:
          | command                                                                                      |
          | "sonar-runner -Dsonar.python.xunit.reportPath=missing"                                       |
          | "sonar-runner -Dsonar.python.xunit.reportPath=missing -Dsonar.python.xunit.skipDetails=true" |


  Scenario: Test report is invalid
      GIVEN the python project "nosetests_project"

      WHEN I run "sonar-runner -Dsonar.python.xunit.reportPath=invalid_report.xml"

      THEN the analysis breaks
          AND the analysis log contains a line matching:
              """
              ERROR.*Cannot feed the data into sonar, details: .*
              """
