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
package org.sonar.plugins.python.mypy;

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
import org.sonar.api.SonarRuntime;
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

class MypySensorTest {

  private static final Path PROJECT_DIR = Paths.get("src", "test", "resources", "org", "sonar", "plugins", "python", "mypy");
  private static final String MYPY_FILE = "python-project:mypy/type_hints_noncompliant.py";
  private static final String UNKNOWN_FILE_REPORT = "mypy_unknown_file_output.txt";

  private final static MypySensor mypySensor = new MypySensor();
  private static final SonarRuntime SONAR_RUNTIME = SonarRuntimeImpl.forSonarQube(Version.create(9, 9), SonarQubeSide.SERVER, SonarEdition.DEVELOPER);

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void descriptor() {
    DefaultSensorDescriptor sensorDescriptor = new DefaultSensorDescriptor();
    mypySensor.describe(sensorDescriptor);
    assertThat(sensorDescriptor.name()).isEqualTo("Import of Mypy issues");
    assertThat(sensorDescriptor.languages()).containsOnly("py");
    assertThat(sensorDescriptor.configurationPredicate()).isNotNull();
    assertNoErrorWarnDebugLogs(logTester);

    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    context.settings().setProperty("sonar.python.mypy.reportPaths", "path/to/report");
    assertThat(sensorDescriptor.configurationPredicate().test(context.config())).isTrue();
  }

  @Test
  void issues_with_sonarqube_79() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("mypy_output.txt");
    assertThat(externalIssues).hasSize(5);

    assertIssue(externalIssues.get(0),
      "arg-type",
      "Argument 1 to \"greet_all\" has incompatible type \"List[int]\"; expected \"List[str]\"",
      11, 0, 11, 15);
    assertIssue(externalIssues.get(1),
      "no-untyped-def",
      "Function is missing a type annotation",
      13, 0, 13, 21);
    assertIssue(externalIssues.get(2),
      "import",
      "Cannot find implementation or library stub for module named \"unknown\"",
      16, 0, 16, 27);
    assertIssue(externalIssues.get(3),
      "no-untyped-call",
      "Call to untyped function \"no_type_hints\" in typed context",
      19, 0, 19, 27);
    assertIssue(externalIssues.get(4),
      "unknown_mypy_rule",
      "Unused \"type: ignore\" comment",
      24, 0, 24, 49);
  }

  @Test
  void issues_with_sonarqube_79_column_numbers() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("mypy_output_show_column_numbers.txt");
    assertThat(externalIssues).hasSize(5);

    assertIssue(externalIssues.get(0),
      "arg-type",
      "Argument 1 to \"greet_all\" has incompatible type \"List[int]\"; expected \"List[str]\"",
      11, 10, 11, 11);
    assertIssue(externalIssues.get(1),
      "no-untyped-def",
      "Function is missing a type annotation",
      13, 0, 13, 1);
    assertIssue(externalIssues.get(2),
      "import",
      "Cannot find implementation or library stub for module named \"unknown\"",
      16, 0, 16, 1);
    assertIssue(externalIssues.get(3),
      "no-untyped-call",
      "Call to untyped function \"no_type_hints\" in typed context",
      19, 10, 19, 11);
    assertIssue(externalIssues.get(4),
      "unknown_mypy_rule",
      "Unused \"type: ignore\" comment",
      24, 0, 24, 49);
  }

  @Test
  void issues_with_sonarqube_79_error_end() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("mypy_output_show_error_end.txt");
    assertThat(externalIssues).hasSize(5);

    assertIssue(externalIssues.get(0),
      "arg-type",
      "Argument 1 to \"greet_all\" has incompatible type \"List[int]\"; expected \"List[str]\"",
      11, 10, 11, 11);
    assertIssue(externalIssues.get(1),
      "no-untyped-def",
      "Function is missing a type annotation",
      13, 0, 13, 1);
    assertIssue(externalIssues.get(2),
      "import",
      "Cannot find implementation or library stub for module named \"unknown\"",
      16, 0, 16, 1);
    assertIssue(externalIssues.get(3),
      "no-untyped-call",
      "Call to untyped function \"no_type_hints\" in typed context",
      19, 10, 19, 11);
    assertIssue(externalIssues.get(4),
      "unknown_mypy_rule",
      "Unused \"type: ignore\" comment",
      24, 0, 24, 49);
  }

  @Test
  void unknown_file_in_report() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(UNKNOWN_FILE_REPORT);
    assertThat(externalIssues).hasSize(5);

    assertThat(onlyOneLogElement(logTester.logs(Level.WARN)))
      .isEqualTo("Failed to resolve 1 file path(s) in Mypy report. No issues imported related to file(s): mypy/unknown.py");
  }

  @Test
  void no_issues_without_report_paths_property() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting(null);
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnDebugLogs(logTester);
  }

  @Test
  void no_issues_with_invalid_report_path() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("invalid-path.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.ERROR)))
      .startsWith("No issues information will be saved as the report file '")
      .contains("invalid-path.txt' can't be read.");
  }

  @Test
  void empty_mypy_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("empty.txt");
    assertThat(externalIssues).isEmpty();
    assertNoErrorWarnDebugLogs(logTester);
  }

  @Test
  void invalid_mypy_file() throws IOException {
    List<ExternalIssue> externalIssues = executeSensorImporting("invalid_format.txt");
    assertThat(externalIssues).isEmpty();
    assertThat(onlyOneLogElement(logTester.logs(Level.DEBUG))).isEqualTo("Cannot parse the line: this is not a mypy output");
  }

  private static void assertIssue(ExternalIssue issue, String key, String message, int startLine, int startColumn, int endLine, int endColumn) {
    IssueLocation location = issue.primaryLocation();
    assertThat(issue.type()).isEqualTo(RuleType.CODE_SMELL);
    assertThat(issue.severity()).isEqualTo(Severity.MAJOR);
    assertThat(issue.ruleId()).isEqualTo(key);
    assertThat(issue.ruleKey()).hasToString(String.format("external_mypy:%s", key));

    assertThat(location.inputComponent().key()).isEqualTo(MYPY_FILE);
    assertThat(location.message())
      .isEqualTo(message);

    TextRange firstTextRange = location.textRange();
    assertThat(firstTextRange).isNotNull();
    assertThat(firstTextRange.start().line()).isEqualTo(startLine);
    assertThat(firstTextRange.start().lineOffset()).isEqualTo(startColumn);
    assertThat(firstTextRange.end().line()).isEqualTo(endLine);
    assertThat(firstTextRange.end().lineOffset()).isEqualTo(endColumn);
  }

  private static List<ExternalIssue> executeSensorImporting(@Nullable String fileName) throws IOException {
    Path baseDir = PROJECT_DIR.getParent();
    SensorContextTester context = SensorContextTester.create(baseDir);
    try (Stream<Path> fileStream = Files.list(PROJECT_DIR)) {
      fileStream.forEach(file -> addFileToContext(context, baseDir, file));
      context.setRuntime(SONAR_RUNTIME);
      if (fileName != null) {
        String path = PROJECT_DIR.resolve(fileName).toAbsolutePath().toString();
        context.settings().setProperty("sonar.python.mypy.reportPaths", path);
      }
      mypySensor.execute(context);
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
