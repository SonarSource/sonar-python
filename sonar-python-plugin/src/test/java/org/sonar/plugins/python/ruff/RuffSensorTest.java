/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.plugins.python.ruff;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.junit.Rule;
import org.junit.Test;
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
import org.sonar.api.utils.Version;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

public class RuffSensorTest {

  private static final String RUFF_FILE = "python-project:ruff/file1.py";
  private static final String RUFF_REPORT = "ruff-report.txt";
  private static final String RUFF_JSON_REPORT = "ruff-json-format.json";
  private static final String RUFF_PROPERTY = "sonar.python.ruff.reportPaths";
  private static final String RUFF_REPORT_UNKNOWN_FILES = "ruff-report-unknown-files.txt";

  private static final Path PROJECT_DIR = Paths.get("src", "test", "resources", "org", "sonar", "plugins", "python", "ruff");

  private static RuffSensor ruffSensor = new RuffSensor();

  @Rule
  public LogTester logTester = new LogTester();

  @Test
  public void test_descriptor() {
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
  public void issues_with_sonarqube_79() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_REPORT);
    assertThat(externalIssues).hasSize(9);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey().toString()).isEqualTo("external_ruff:S107");
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
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(17);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey().toString()).isEqualTo("external_ruff:S605");
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
    assertThat(secondTextRange.end().lineOffset()).isEqualTo(16);

    ExternalIssue third = externalIssues.get(2);
    assertThat(third.ruleKey().toString()).isEqualTo("external_ruff:UP031");
    assertThat(third.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(third.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation thirdPrimaryLoc = third.primaryLocation();
    assertThat(thirdPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(thirdPrimaryLoc.message()).isEqualTo("[*] Use format specifiers instead of percent format");

    ExternalIssue last = externalIssues.get(8);
    assertThat(last.ruleKey().toString()).isEqualTo("external_ruff:S110");
    assertThat(last.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(last.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation fourthPrimaryLoc = last.primaryLocation();
    assertThat(fourthPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(fourthPrimaryLoc.message()).isEqualTo("`try`-`except`-`pass` detected, consider logging the exception");

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).containsAll(new ArrayList<>(Arrays.asList(
      "Cannot parse the line: Found 9 errors.",
      "Cannot parse the line: [*] 1 potentially fixable with the --fix option."
    )));
  }

  @Test
  public void issues_with_json_format() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_JSON_REPORT);
    assertThat(externalIssues).hasSize(9);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey().toString()).isEqualTo("external_ruff:S107");
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
    assertThat(second.ruleKey().toString()).isEqualTo("external_ruff:S605");
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


    ExternalIssue fourth = externalIssues.get(3);
    assertThat(fourth.ruleKey().toString()).isEqualTo("external_ruff:F821");
    assertThat(fourth.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(fourth.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation fourthPrimaryLoc = fourth.primaryLocation();
    assertThat(fourthPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(fourthPrimaryLoc.message()).isEqualTo("Undefined name `random`");

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).isEmpty();

  }

  @Test
  public void issues_with_pylint_format() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "ruff-pylint-format.txt");
    assertThat(externalIssues).hasSize(9);

    ExternalIssue first = externalIssues.get(0);
    assertThat(first.ruleKey().toString()).isEqualTo("external_ruff:S107");
    assertThat(first.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(first.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation firstPrimaryLoc = first.primaryLocation();
    assertThat(firstPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(firstPrimaryLoc.message())
      .isEqualTo("Possible hardcoded password assigned to function default: \"secret\"");
    TextRange firstTextRange = firstPrimaryLoc.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(5);
    assertThat(firstTextRange.start().lineOffset()).isEqualTo(0);
    assertThat(firstTextRange.end().line()).isEqualTo(5);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(25);

    ExternalIssue second = externalIssues.get(1);
    assertThat(second.ruleKey().toString()).isEqualTo("external_ruff:S605");
    assertThat(second.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(second.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation secondPrimaryLoc = second.primaryLocation();
    assertThat(secondPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(secondPrimaryLoc.message()).isEqualTo("Starting a process with a shell, possible injection detected");
    TextRange secondTextRange = secondPrimaryLoc.textRange();
    assertThat(secondTextRange).isNotNull();
    assertThat(secondTextRange.start().line()).isEqualTo(6);
    assertThat(secondTextRange.start().lineOffset()).isEqualTo(0);
    assertThat(secondTextRange.end().line()).isEqualTo(6);
    assertThat(secondTextRange.end().lineOffset()).isEqualTo(43);

    ExternalIssue third = externalIssues.get(2);
    assertThat(third.ruleKey().toString()).isEqualTo("external_ruff:UP031");
    assertThat(third.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(third.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation thirdPrimaryLoc = third.primaryLocation();
    assertThat(thirdPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(thirdPrimaryLoc.message()).isEqualTo("Use format specifiers instead of percent format");

    ExternalIssue last = externalIssues.get(8);
    assertThat(last.ruleKey().toString()).isEqualTo("external_ruff:S110");
    assertThat(last.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(last.severity()).isEqualTo(Severity.MAJOR);
    IssueLocation fourthPrimaryLoc = last.primaryLocation();
    assertThat(fourthPrimaryLoc.inputComponent().key()).isEqualTo(RUFF_FILE);
    assertThat(fourthPrimaryLoc.message()).isEqualTo("`try`-`except`-`pass` detected, consider logging the exception");

    assertNoErrorWarnLogs(logTester);
    assertThat(logTester.logs(LoggerLevel.DEBUG)).isEmpty();
  }

  @Test
  public void issues_with_sonarqube_79_unknown_files() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, RUFF_REPORT_UNKNOWN_FILES);
    assertThat(externalIssues).hasSize(2);

    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.WARN)))
      .isEqualTo("Failed to resolve 2 file path(s) in Ruff report. No issues imported related to file(s): tests/subject/unknown1.py;tests/subject/unknown2.py");
  }

  @Test
  public void no_issues_without_report_paths_property() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, null);
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnLogs(logTester);
  }

  @Test
  public void no_issues_with_invalid_report_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "invalid-path.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("invalid-path.txt' can't be read.");
  }

  @Test
  public void no_issues_with_empty_or_invalid_ruff_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(7, 9, "empty-file.txt");
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnLogs(logTester);

    externalIssues = executeSensorImporting(7, 9, "ruff-invalid-file.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(LoggerLevel.DEBUG))).isEqualTo("Cannot parse the line: invalid line");
  }


  private static List<ExternalIssue> executeSensorImporting(int majorVersion, int minorVersion, @Nullable String fileName) throws IOException {
    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    try (Stream<Path> fileStream = Files.list(PROJECT_DIR)) {
      fileStream.forEach(file -> addFileToContext(context, baseDir, file));
      context.setRuntime(SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER, SonarEdition.DEVELOPER));
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

  public static void assertNoErrorWarnLogs(LogTester logTester) {
    assertThat(logTester.logs(LoggerLevel.ERROR)).isEmpty();
    assertThat(logTester.logs(LoggerLevel.WARN)).isEmpty();
  }

}
