Feature: Importing coverage data

  As a SonarQube user
  I want to import my coverage metric values into SonarQube
  In order to be able to use relevant SonarQube features

  Scenario: Importing a valid coverage report
      GIVEN the python project "coverage_project"

      WHEN I run "sonar-runner"

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
              | overall_line_coverage    | 50   |
              | overall_branch_coverage  | 0    |
