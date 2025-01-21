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
package org.sonar.plugins.python.flake8;

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

class Flake8SensorTest {

  private static final String FLAKE8_FILE = "python-project:flake8/file1.py";
  private static final String FLAKE8_F401 = "external_flake8:F401";
  private static final String FLAKE_8_REPORT = "flake8-report.txt";
  private static final String FLAKE_8_PROPERTY = "sonar.python.flake8.reportPaths";
  private static final String FLAKE_8_REPORT_UNKNOWN_FILES = "flake8-report-unknown-files.txt";

  private static final Path PROJECT_DIR = Paths.get("src", "test", "resources", "org", "sonar", "plugins", "python", "flake8");

  private static Flake8Sensor flake8Sensor = new Flake8Sensor();

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void test_descriptor() {
    DefaultSensorDescriptor sensorDescriptor = new DefaultSensorDescriptor();
    flake8Sensor.describe(sensorDescriptor);
    assertThat(sensorDescriptor.name()).isEqualTo("Import of Flake8 issues");
    assertThat(sensorDescriptor.languages()).containsOnly("py");
    assertThat(sensorDescriptor.configurationPredicate()).isNotNull();
    assertNoErrorWarnDebugLogs(logTester);

    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    context.settings().setProperty(FLAKE_8_PROPERTY, "path/to/report");
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isTrue();

    context = SensorContextTester.create(baseDir);
    context.settings().setProperty("sonar.python.flake8.reportPath", "path/to/report");
    // No support of "reportPath" property for Flake8
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isFalse();
  }

  @Test
  void issues_with_sonarqube_79() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, FLAKE_8_REPORT);
    assertThat(externalIssues).hasSize(3);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey().toString()).isEqualTo(FLAKE8_F401);
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(FLAKE8_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("'os' imported but unused");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isEqualTo(0);
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(1);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey().toString()).isEqualTo("external_flake8:E302");
    assertThat(second.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(second.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondPrimaryLoc = second.primaryLocation();
    assertThat(secondPrimaryLoc.inputComponent().key()).isEqualTo(FLAKE8_FILE);
    assertThat(secondPrimaryLoc.message()).isEqualTo("expected 2 blank lines, found 1");
    TextRange secondTextRange = secondPrimaryLoc.textRange();
    assertThat(secondTextRange).isNotNull();
    assertThat(secondTextRange.start().line()).isEqualTo(3);
    assertThat(secondTextRange.start().lineOffset()).isEqualTo(0);
    assertThat(secondTextRange.end().line()).isEqualTo(3);
    assertThat(secondTextRange.end().lineOffset()).isEqualTo(1);

    ExternalIssue third = externalIssues.get(2);
    assertThat(third.ruleKey().toString()).isEqualTo("external_flake8:C901");
    assertThat(third.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(third.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation thirdPrimaryLoc = third.primaryLocation();
    assertThat(thirdPrimaryLoc.inputComponent().key()).isEqualTo(FLAKE8_FILE);
    assertThat(thirdPrimaryLoc.message()).isEqualTo("'bar' is too complex (6)");

    assertNoErrorWarnDebugLogs(logTester);
  }

  @Test
  void issues_with_sonarqube_79_unknown_files() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, FLAKE_8_REPORT_UNKNOWN_FILES);
    assertThat(externalIssues).hasSize(2);

    assertThat(onlyOneLogElement(logTester.logs(Level.WARN)))
      .isEqualTo("Failed to resolve 2 file path(s) in Flake8 report. No issues imported related to file(s): tests/subject/unknown1.py;tests/subject/unknown2.py");
  }

  @Test
  void no_issues_without_report_paths_property() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, null);
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnDebugLogs(logTester);
  }

  @Test
  void no_issues_with_invalid_report_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "invalid-path.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("invalid-path.txt' can't be read.");
  }

  @Test
  void no_issues_with_empty_or_invalid_flake8_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "empty-file.txt");
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnDebugLogs(logTester);

    externalIssues = executeSensorImporting(7, 9, "flake8-invalid-file.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.DEBUG))).isEqualTo("Cannot parse the line: invalid line");
  }

  @Test
  void issues_with_pylint_format() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "flake8-pylint-format.txt");
    assertThat(externalIssues).hasSize(2);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey().toString()).isEqualTo(FLAKE8_F401);
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(FLAKE8_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("'os' imported but unused");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(1);
    assertThat(firstTextRange.start().lineOffset()).isEqualTo(0);
    assertThat(firstTextRange.end().line()).isEqualTo(1);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(9);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey().toString()).isEqualTo("external_flake8:E302");
    assertThat(second.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(second.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondPrimaryLoc = second.primaryLocation();
    assertThat(secondPrimaryLoc.inputComponent().key()).isEqualTo(FLAKE8_FILE);
    assertThat(secondPrimaryLoc.message()).isEqualTo("expected 2 blank lines, found 1");
    TextRange secondTextRange = secondPrimaryLoc.textRange();
    assertThat(secondTextRange).isNotNull();
    assertThat(secondTextRange.start().line()).isEqualTo(3);
    assertThat(secondTextRange.start().lineOffset()).isEqualTo(0);

    assertNoErrorWarnDebugLogs(logTester);
  }

  private static List<ExternalIssue> executeSensorImporting(int majorVersion, int minorVersion, @Nullable String fileName) throws IOException {
    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    try (Stream<Path> fileStream = Files.list(PROJECT_DIR)) {
      fileStream.forEach(file -> addFileToContext(context, baseDir, file));
      context.setRuntime(SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER, SonarEdition.DEVELOPER));
      if (fileName != null) {
        String path = PROJECT_DIR.resolve(fileName).toAbsolutePath().toString();
        context.settings().setProperty("sonar.python.flake8.reportPaths", path);
      }
      flake8Sensor.execute(context);
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

  public static void assertNoErrorWarnDebugLogs(LogTesterJUnit5 logTester) {
    assertThat(logTester.logs(Level.ERROR)).isEmpty();
    assertThat(logTester.logs(Level.WARN)).isEmpty();
    assertThat(logTester.logs(Level.DEBUG)).isEmpty();
  }

}
