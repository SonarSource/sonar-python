Feature: Importing coverage data

  As a SonarQube user
  I want to import my coverage metric values into SonarQube
  In order to be able to use relevant SonarQube features


  Scenario: Importing coverage reports
      GIVEN the python project "coverage_project"

      WHEN I run sonar-runner with following options:
          """
          -Dsonar.python.coverage.reportPath=ut-coverage.xml
          -Dsonar.python.coverage.itReportPath=it-coverage.xml
          -Dsonar.python.coverage.overallReportPath=it-coverage.xml
          """

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric                   | value |
              | coverage                 | 80.0  |
              | line_coverage            | 75.0  |
              | branch_coverage          | 100   |
              | it_coverage              | 40    |
              | it_line_coverage         | 50    |
              | it_branch_coverage       | 0     |
              | overall_coverage         | 40    |
              | overall_line_coverage    | 50    |
              | overall_branch_coverage  | 0     |

  @wip
  Scenario Outline: Importing coverage reports zeroing coverage for untouched files
      GIVEN the python project "coverage_project"

      WHEN I run sonar-runner with following options:
          """
          -Dsonar.python.coverage.reportPath=<reportpath>
          -Dsonar.python.coverage.itReportPath=it-coverage.xml
          -Dsonar.python.coverage.overallReportPath=it-coverage.xml
          -Dsonar.python.coverage.forceZeroCoverage=True
          """

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric                  | value |
              | coverage                | 50    |
              | line_coverage           | 42.9  |
              | branch_coverage         | 100   |
              | it_coverage             | 25    |
              | it_line_coverage        | 28.6  |
              | it_branch_coverage      | 0     |
              | overall_coverage        | 25    |
              | overall_line_coverage   | 28.6  |
              | overall_branch_coverage | 0     |

     Examples:
      | reportpath                    |
      | ut-coverage.xml               |
      | ut-coverage-windowspaths.xml  |


  Scenario: Zeroing coverage measures without importing reports

      If we dont pass coverage reports *and* request zeroing untouched
      files at the same time, all coverage measures, except the branch
      ones, should be 'zero'. The branch coverage measures remain 'None',
      since its currently ignored by the 'force zero...'
      implementation

      GIVEN the python project "coverage_project"

      WHEN I run "sonar-runner -Dsonar.python.coverage.forceZeroCoverage=True"

      THEN the analysis finishes successfully
          AND the analysis log contains no error or warning messages
          AND the following metrics have following values:
              | metric                  | value |
              | coverage                | 0     |
              | line_coverage           | 0     |
              | branch_coverage         | None  |
              | it_coverage             | 0     |
              | it_line_coverage        | 0     |
              | it_branch_coverage      | None  |
              | overall_coverage        | 0     |
              | overall_line_coverage   | 0     |
              | overall_branch_coverage | None  |


   Scenario: Importing an empty coverage report

      GIVEN the python project "coverage_project"

      WHEN I run sonar-runner with following options:
          """
          -X
          -Dsonar.python.coverage.reportPath=empty.xml
          -Dsonar.python.coverage.itReportPath=empty.xml
          -Dsonar.python.coverage.overallReportPath=empty.xml
          """

      THEN the analysis finishes successfully
         BUT the analysis log contains a line matching
              """
              .*WARN.*The report '.*' seems to be empty, ignoring.
              """
          AND the following metrics have following values:
              | metric                  | value |
              | coverage                | None  |
              | it_coverage             | None  |
              | overall_coverage        | None  |
