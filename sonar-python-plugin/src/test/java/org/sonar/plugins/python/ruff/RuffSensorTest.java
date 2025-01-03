/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.ruff;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import javax.annotation.Nullable;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.ExternalIssue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.rules.RuleType;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;
import org.sonar.api.utils.log.LoggerLevel;

class RuffSensorTest {

  private static final String RUFF_FILE = "python-project:ruff/file1.py";
  private static final String RUFF_JSON_REPORT = "ruff-json-format.json";
  private static final String RUFF_PROPERTY = "sonar.python.ruff.reportPaths";
  private static final String RUFF_REPORT_UNKNOWN_FILES = "unknown-file-path.json";

  private static final Path PROJECT_DIR = Paths.get("src", "test", "resources", "org", "sonar", "plugins", "python",
    "ruff");

  private static RuffSensor ruffSensor = new RuffSensor();

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void test_descriptor() {
    DefaultSensorDescriptor sensorDescriptor = new DefaultSensorDescriptor();
    ruffSensor.describe(sensorDescriptor);
    assertThat(sensorDescriptor.name()).isEqualTo("Import of Ruff issues");
    assertThat(sensorDescriptor.languages()).containsOnly("py");
    assertThat(sensorDescriptor.configurationPredicate()).isNotNull();
    assertNoErrorWarnLogs(logTester);

    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    context.settings().setProperty(RUFF_PROPERTY, "path/to/report");
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isTrue();

    context = SensorContextTester.create(baseDir);
    context.settings().setProperty("sonar.python.ruff.reportPath", "path/to/report");
    // No support of "reportPath" property for Ruff
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isFalse();
  }

  @Test
  void issues_with_json_format() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_JSON_REPORT);
    assertThat(externalIssues).hasSize(10);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_ruff:S107");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Possible hardcoded password assigned to function default: \"secret\"");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(5);
    assertThat(firstTextRange.start().lineOffset()).isEqualTo(16);
    assertThat(firstTextRange.end().line()).isEqualTo(5);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(23);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey()).hasToString("external_ruff:S605");
    assertThat(second.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(second.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondPrimaryLoc = second.primaryLocation();
    assertThat(secondPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(secondPrimaryLoc.message()).isEqualTo("Starting a process with a shell, possible injection detected");
    TextRange secondTextRange = secondPrimaryLoc.textRange();
    assertThat(secondTextRange).isNotNull();
    assertThat(secondTextRange.start().line()).isEqualTo(6);
    assertThat(secondTextRange.start().lineOffset()).isEqualTo(15);
    assertThat(secondTextRange.end().line()).isEqualTo(6);
    assertThat(secondTextRange.end().lineOffset()).isEqualTo(42);

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).isEmpty();

  }

  @Test
  void issues_primary_location_check() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_JSON_REPORT);
    assertThat(externalIssues).hasSize(10);

    ExternalIssue fourth = externalIssues.get(3);
    assertThat(fourth.ruleKey()).hasToString("external_ruff:F821");
    assertThat(fourth.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(fourth.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation fourthPrimaryLoc = fourth.primaryLocation();
    assertThat(fourthPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(fourthPrimaryLoc.message()).isEqualTo("Undefined name `random`");

    ExternalIssue secondToLast = externalIssues.get(8);
    assertThat(secondToLast.ruleKey()).hasToString("external_ruff:S110");
    assertThat(secondToLast.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(secondToLast.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondToLastPrimaryLoc = secondToLast.primaryLocation();
    assertThat(secondToLastPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(secondToLastPrimaryLoc.message()).isEqualTo("`try`-`except`-`pass` detected, consider logging the exception");

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).isEmpty();

  }

  @Test
  void issues_multiline_check() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_JSON_REPORT);
    assertThat(externalIssues).hasSize(10);

    ExternalIssue last = externalIssues.get(9);
    assertThat(last.ruleKey()).hasToString("external_ruff:C417");
    assertThat(last.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(last.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation lastPrimaryLoc = last.primaryLocation();
    assertThat(lastPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(lastPrimaryLoc.message()).isEqualTo("Unnecessary `map` usage (rewrite using a `list` comprehension)");

    TextRange lastTextRange = lastPrimaryLoc.textRange();
    assertThat(lastTextRange).isNotNull();
    assertThat(lastTextRange.start().line()).isEqualTo(25);
    assertThat(lastTextRange.start().lineOffset()).isEqualTo(33);
    assertThat(lastTextRange.end().line()).isEqualTo(27);
    assertThat(lastTextRange.end().lineOffset()).isEqualTo(5);

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).isEmpty();

  }

  @Test
  void report_on_first_line() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "report-on-first-line.json");
    assertThat(externalIssues).hasSize(1);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_ruff:D100");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo("python-project:ruff/file1.py");
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Missing docstring in public module");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isZero();
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(9);
  }

  @Test
  void report_on_empty_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "report-on-empty-file.json");
    assertThat(externalIssues).hasSize(1);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_ruff:D104");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo("python-project:ruff/__init__.py");
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Missing docstring in public package");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isZero();
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isZero();
  }

  @Test
  void unknown_json_file_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_REPORT_UNKNOWN_FILES);
    assertThat(externalIssues).hasSize(1);

    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.WARN)))
      .isEqualTo(
        "Failed to resolve 1 file path(s) in Ruff report. No issues imported related to file(s): unknown/file.py");
  }

  @Test
  void no_issues_without_report_paths_property() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, null);
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnLogs(logTester);
  }

  @Test
  void missing_rule_key_file_name_or_message() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "missing-fields.json");
    assertThat(externalIssues).hasSize(1);

    assertThat(logTester.logs(LoggerLevel.DEBUG)).hasSize(3);
    assertThat(logTester.logs(LoggerLevel.DEBUG).get(0))
      .startsWith("Missing information for ruleKey:'null',");
    assertThat(logTester.logs(LoggerLevel.DEBUG).get(1))
      .contains("filePath:'null'");
    assertThat(logTester.logs(LoggerLevel.DEBUG).get(2))
      .contains("message:'null'");
  }

  @Test
  void no_issues_with_invalid_report_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "invalid-path.json");
    assertThat(externalIssues).isEmpty();

    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("invalid-path.json' can't be read.");
  }

  @Test
  void no_issues_with_empty_or_invalid_ruff_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "empty-file.json");
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnLogs(logTester);

    externalIssues = executeSensorImporting(7, 9, "ruff-invalid-file.json");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("ruff-invalid-file.json' can't be read.");
  }

  @Test
  void unknown_rule() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "unknown-rule.json");
    assertThat(externalIssues).hasSize(1);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_ruff:ZZZ999");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);

    assertNoErrorWarnLogs(logTester);

  }

  @Test
  void incorrect_end_location() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "incorrect-end-location.json");
    assertThat(externalIssues).hasSize(1);
    assertNoErrorWarnLogs(logTester);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_ruff:S107");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Possible hardcoded password assigned to function default: \"secret\"");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(5);
    assertThat(firstTextRange.start().lineOffset()).isZero();
    assertThat(firstTextRange.end().line()).isEqualTo(5);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(25);

  }

  private static List<ExternalIssue> executeSensorImporting(int majorVersion, int minorVersion,
    @Nullable String fileName) throws IOException {
    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    try (Stream<Path> fileStream = Files.list(PROJECT_DIR)) {
      fileStream.forEach(file -> addFileToContext(context, baseDir, file));
      context.setRuntime(SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER,
        SonarEdition.DEVELOPER));
      if (fileName != null) {
        String path = PROJECT_DIR.resolve(fileName).toAbsolutePath().toString();
        context.settings().setProperty("sonar.python.ruff.reportPaths", path);
      }
      ruffSensor.execute(context);
      return new ArrayList<>(context.allExternalIssues());
    }
  }

  private static void addFileToContext(SensorContextTester context, Path projectDir, Path file) {
    try {
      String projectId = projectDir.getFileName().toString() + "-project";
      context.fileSystem().add(TestInputFileBuilder.create(projectId, projectDir.toFile(), file.toFile())
        .setCharset(UTF_8)
        .setLanguage(language(file))
        .setContents(new String(Files.readAllBytes(file), UTF_8))
        .setType(InputFile.Type.MAIN)
        .build());
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  private static String language(Path file) {
    String path = file.toString();
    return path.substring(path.lastIndexOf('.') + 1);
  }

  public static String onlyOneLogElement(List<String> elements) {
    assertThat(elements).hasSize(1);
    return elements.get(0);
  }

  public static void assertNoErrorWarnLogs(LogTesterJUnit5 logTester) {
    assertThat(logTester.logs(LoggerLevel.ERROR)).isEmpty();
    assertThat(logTester.logs(LoggerLevel.WARN)).isEmpty();
  }

}
