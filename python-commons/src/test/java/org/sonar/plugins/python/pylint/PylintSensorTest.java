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
package org.sonar.plugins.python.pylint;

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

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

class PylintSensorTest {

  private static final String PYLINT_FILE = "python-project:pylint/file1.py";
  private static final String PYLINT_REPORT_DEFAULT_FORMAT = "pylint_report_default_format.txt";
  private static final String PYLINT_REPORT_NO_COLUMN = "pylint_report_no_column.txt";
  private static final String DEFAULT_PROPERTY = "sonar.python.pylint.reportPaths";
  private static final String LEGACY_PROPERTY = "sonar.python.pylint.reportPath";

  private static final Path PROJECT_DIR = Paths.get("src", "test", "resources", "org", "sonar", "plugins", "python", "pylint");

  private static PylintSensor pylintSensor = new PylintSensor();

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void test_descriptor() {
    DefaultSensorDescriptor sensorDescriptor = new DefaultSensorDescriptor();
    pylintSensor.describe(sensorDescriptor);
    assertThat(sensorDescriptor.name()).isEqualTo("Import of Pylint issues");
    assertThat(sensorDescriptor.languages()).containsOnly("py");
    assertThat(sensorDescriptor.configurationPredicate()).isNotNull();
    assertNoErrorWarnLogs(logTester);

    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    context.settings().setProperty(DEFAULT_PROPERTY, "path/to/report");
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isTrue();

    context = SensorContextTester.create(baseDir);
    context.settings().setProperty(LEGACY_PROPERTY, "path/to/report");
    // Support of legacy "reportPath" property for Pylint
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isTrue();

    context = SensorContextTester.create(baseDir);
    context.settings().setProperty("random.key", "path/to/report");
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isFalse();
  }

  @Test
  void issues_default_format() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, PYLINT_REPORT_DEFAULT_FORMAT, false);
    assertThat(externalIssues).hasSize(10);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_pylint:C0114");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Missing module docstring (missing-module-docstring)");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isZero();
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(1);

    // Issue on column >= last column is reported on whole line instead
    ExternalIssue last = externalIssues.get(9);
    IssueLocation lastPrimaryLoc = last.primaryLocation();
    TextRange lastTextRange = lastPrimaryLoc.textRange();
    assertThat(lastTextRange).isNotNull();
    assertThat(lastTextRange.start().line()).isEqualTo(1);
    assertThat(lastTextRange.start().lineOffset()).isZero();
    assertThat(lastTextRange.end().line()).isEqualTo(1);
    assertThat(lastTextRange.end().lineOffset()).isEqualTo(13);

    assertNoErrorWarnLogs(logTester);
  }

  @Test
  void issues_no_column_info() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, PYLINT_REPORT_NO_COLUMN, false);
    assertThat(externalIssues).hasSize(3);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_pylint:C0111");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Missing module docstring");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isZero();
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(13);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey()).hasToString("external_pylint:C0103");
    assertThat(second.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(second.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondPrimaryLoc = second.primaryLocation();
    assertThat(secondPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(secondPrimaryLoc.message()).isEqualTo("Invalid argument name \"n\"");
    TextRange secondTextRange = secondPrimaryLoc.textRange();
    assertThat(secondTextRange).isNotNull();
    assertThat(secondTextRange.start().line()).isEqualTo(1);
    assertThat(secondTextRange.start().lineOffset()).isZero();
    assertThat(secondTextRange.end().line()).isEqualTo(1);
    assertThat(secondTextRange.end().lineOffset()).isEqualTo(13);

    ExternalIssue third = externalIssues.get(2);
    assertThat(third.ruleKey()).hasToString("external_pylint:C0111");
    assertThat(third.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(third.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation thirdPrimaryLoc = third.primaryLocation();
    assertThat(thirdPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(thirdPrimaryLoc.message()).isEqualTo("Missing function docstring");

    assertNoErrorWarnLogs(logTester);
    assertThat(onlyOneLogElement(logTester.logs(Level.DEBUG)))
      .isEqualTo("Cannot parse the line: ************* Module src.prod");
  }

  @Test
  void issues_other_formats() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "pylint_brackets.txt", false);
    assertThat(externalIssues).hasSize(10);
    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_pylint:E1300");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("message");

    externalIssues = executeSensorImporting(7, 9, "pylint_names_in_brackets.txt", false);
    assertThat(externalIssues).hasSize(1);
    first = externalIssues.get(0);
    assertThat(first.ruleKey()).hasToString("external_pylint:C0111");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(PYLINT_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Missing docstring");
  }

  @Test
  void no_issues_unknown_files() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "pylint_report_unknown_files.txt", false);
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.WARN))).hasToString("Failed to resolve 1 file path(s) in Pylint report. " +
      "No issues imported related to file(s): pylint/unknown_file.py");
  }

  @Test
  void no_issues_with_invalid_report_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "invalid-path.txt", false);
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("invalid-path.txt' can't be read.");
  }

  @Test
  void import_with_legacy_property() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "pylint_brackets.txt", true);
    assertThat(externalIssues).hasSize(10);
    assertThat(onlyOneLogElement(logTester.logs())).hasToString("The use of 'sonar.python.pylint.reportPath' is deprecated." +
      " Please use the 'sonar.python.pylint.reportPaths' property instead.");
  }

  private static List<ExternalIssue> executeSensorImporting(int majorVersion, int minorVersion, @Nullable String fileName, boolean useLegacyKey) throws IOException {
    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    try (Stream<Path> fileStream = Files.list(PROJECT_DIR)) {
      fileStream.forEach(file -> addFileToContext(context, baseDir, file));
      context.setRuntime(SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER, SonarEdition.DEVELOPER));
      if (fileName != null) {
        String path = PROJECT_DIR.resolve(fileName).toAbsolutePath().toString();
        if (useLegacyKey) {
          context.settings().setProperty(LEGACY_PROPERTY, path);
        } else {
          context.settings().setProperty(DEFAULT_PROPERTY, path);
        }
      }
      pylintSensor.execute(context);
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
    assertThat(logTester.logs(Level.ERROR)).isEmpty();
    assertThat(logTester.logs(Level.WARN)).isEmpty();
  }

}
