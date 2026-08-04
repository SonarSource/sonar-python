/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import com.sonarsource.scanner.engine.sensor.test.fixtures.SensorContextTester;
import com.sonarsource.scanner.engine.sensor.test.fixtures.TestInputFileBuilder;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.indexer.PythonIndexer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class IssuesRepositoryTest {

  @TempDir
  File baseDir;

  private SensorContextTester context;
  private PythonInputFile inputFile;

  @BeforeEach
  void setUp() throws Exception {
    File source = new File(baseDir, "sample.py");
    Files.writeString(source.toPath(), "x = 1\ny = 2\n");
    context = SensorContextTester.create(baseDir);
    inputFile = new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", "sample.py")
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setLanguage(Python.KEY)
      .setType(InputFile.Type.MAIN)
      .initMetadata(TestUtils.fileContent(source, StandardCharsets.UTF_8))
      .build());
    context.fileSystem().add(inputFile.wrappedFile());
  }

  @Test
  void reports_typed_data_flows_with_description() {
    context = spy(context);
    MockSonarLintIssue mockIssue = new MockSonarLintIssue(context);
    when(context.newIssue()).thenReturn(mockIssue);

    PythonChecks checks = mock(PythonChecks.class);
    when(checks.ruleKey(any())).thenReturn(RuleKey.of("python", "S9999"));
    PythonIndexer indexer = mock(PythonIndexer.class);

    PythonCheck check = new PythonVisitorCheck() {
    };
    PythonCheck.PreciseIssue preciseIssue = new PythonCheck.PreciseIssue(check,
      IssueLocation.preciseLocation(new LocationInFile(null, 1, 0, 1, 1), "primary"));
    preciseIssue.addFlow("Actual value first",
      List.of(IssueLocation.preciseLocation(new LocationInFile(null, 1, 0, 1, 1), "Actual value first.")));
    preciseIssue.addFlow("Expected value first",
      List.of(IssueLocation.preciseLocation(new LocationInFile(null, 2, 0, 2, 1), "Expected value first.")));

    IssuesRepository repository = new IssuesRepository(context, checks, indexer, true, new ReentrantLock());
    repository.save(inputFile, List.of(preciseIssue));

    assertThat(mockIssue.getSaved()).isTrue();
    assertThat(mockIssue.flows()).hasSize(2);
    Issue.Flow firstFlow = mockIssue.flows().get(0);
    assertThat(firstFlow.type()).isEqualTo(NewIssue.FlowType.DATA);
    assertThat(firstFlow.description()).isEqualTo("Actual value first");
    assertThat(firstFlow.locations()).hasSize(1);
    assertThat(firstFlow.locations().get(0).message()).isEqualTo("Actual value first.");

    Issue.Flow secondFlow = mockIssue.flows().get(1);
    assertThat(secondFlow.type()).isEqualTo(NewIssue.FlowType.DATA);
    assertThat(secondFlow.description()).isEqualTo("Expected value first");
  }

  @Test
  void reports_flow_locations_from_other_files() throws Exception {
    File otherSource = new File(baseDir, "other.py");
    Files.writeString(otherSource.toPath(), "z = 3\n");
    PythonInputFile otherFile = new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", "other.py")
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setLanguage(Python.KEY)
      .setType(InputFile.Type.MAIN)
      .initMetadata(TestUtils.fileContent(otherSource, StandardCharsets.UTF_8))
      .build());
    context.fileSystem().add(otherFile.wrappedFile());

    context = spy(context);
    MockSonarLintIssue mockIssue = new MockSonarLintIssue(context);
    when(context.newIssue()).thenReturn(mockIssue);

    PythonChecks checks = mock(PythonChecks.class);
    when(checks.ruleKey(any())).thenReturn(RuleKey.of("python", "S9999"));
    PythonIndexer indexer = mock(PythonIndexer.class);

    PythonCheck check = new PythonVisitorCheck() {
    };
    PythonCheck.PreciseIssue preciseIssue = new PythonCheck.PreciseIssue(check,
      IssueLocation.preciseLocation(new LocationInFile(null, 1, 0, 1, 1), "primary"));
    preciseIssue.addFlow("Cross-file flow",
      List.of(IssueLocation.preciseLocation(
        new LocationInFile(otherSource.getAbsolutePath(), 1, 0, 1, 1),
        "On other file")));

    IssuesRepository repository = new IssuesRepository(context, checks, indexer, true, new ReentrantLock());
    repository.save(inputFile, List.of(preciseIssue));

    assertThat(mockIssue.getSaved()).isTrue();
    assertThat(mockIssue.flows()).hasSize(1);
    assertThat(mockIssue.flows().get(0).description()).isEqualTo("Cross-file flow");
    assertThat(mockIssue.flows().get(0).locations()).hasSize(1);
  }

  @Test
  void skips_flow_locations_when_file_cannot_be_resolved() {
    context = spy(context);
    MockSonarLintIssue mockIssue = new MockSonarLintIssue(context);
    when(context.newIssue()).thenReturn(mockIssue);

    PythonChecks checks = mock(PythonChecks.class);
    when(checks.ruleKey(any())).thenReturn(RuleKey.of("python", "S9999"));
    PythonIndexer indexer = mock(PythonIndexer.class);

    PythonCheck check = new PythonVisitorCheck() {
    };
    PythonCheck.PreciseIssue preciseIssue = new PythonCheck.PreciseIssue(check,
      IssueLocation.preciseLocation(new LocationInFile(null, 1, 0, 1, 1), "primary"));
    preciseIssue.addFlow("Unresolved flow",
      List.of(IssueLocation.preciseLocation(
        new LocationInFile("/does/not/exist.py", 1, 0, 1, 1),
        "Missing file")));

    IssuesRepository repository = new IssuesRepository(context, checks, indexer, true, new ReentrantLock());
    repository.save(inputFile, List.of(preciseIssue));

    assertThat(mockIssue.getSaved()).isTrue();
    assertThat(mockIssue.flows()).isEmpty();
  }
}
