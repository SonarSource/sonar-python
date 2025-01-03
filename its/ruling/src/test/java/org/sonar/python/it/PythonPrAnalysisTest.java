/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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
package org.sonar.python.it;

import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.container.Edition;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.it.RulingHelper.getMeasure;
import static org.sonar.python.it.RulingHelper.getOrchestrator;

class PythonPrAnalysisTest {

  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = getOrchestrator(Edition.ENTERPRISE);

  private static final String PR_ANALYSIS_PROJECT_KEY = "prAnalysis";
  private static final String INCREMENTAL_ANALYSIS_PROFILE = "incrementalPrAnalysis";
  private static final String PR_KEY = "1";
  private static final String PR_BRANCH_NAME = "incremental";

  @TempDir
  public Path temporaryFolder;

  private String scenario;
  private int expectedTotalFiles;
  private int expectedRecomputed;
  private int expectedPartiallySkipped;
  private int expectedFullySkipped;
  private List<String> deletedFiles;
  private Integer expectedDuplicatedLines;

  public void init(
    String scenario,
    int expectedTotalFiles,
    int expectedRecomputed,
    int expectedFullySkipped,
    int expectedPartiallySkipped,
    List<String> deletedFiles,
    Integer expectedDuplication
  ) {
    this.scenario = scenario;
    this.expectedTotalFiles = expectedTotalFiles;
    this.expectedRecomputed = expectedRecomputed;
    this.expectedFullySkipped = expectedFullySkipped;
    this.expectedPartiallySkipped = expectedPartiallySkipped;
    this.deletedFiles = deletedFiles;
    this.expectedDuplicatedLines = expectedDuplication;
  }

  @BeforeAll
  static void prepare_quality_profile() throws IOException {
    ORCHESTRATOR.getServer().provisionProject(PR_ANALYSIS_PROJECT_KEY, PR_ANALYSIS_PROJECT_KEY);

    // Create and load quality profile
    String profile = RulingHelper.profile(INCREMENTAL_ANALYSIS_PROFILE, "py", "python", List.of("S5713"));
    RulingHelper.loadProfile(ORCHESTRATOR, profile);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PR_ANALYSIS_PROJECT_KEY, "py", INCREMENTAL_ANALYSIS_PROFILE);
    String iPythonProfile = RulingHelper.profile(INCREMENTAL_ANALYSIS_PROFILE, "ipynb", "ipython", List.of("S5713"));
    RulingHelper.loadProfile(ORCHESTRATOR, iPythonProfile);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PR_ANALYSIS_PROJECT_KEY, "ipynb", INCREMENTAL_ANALYSIS_PROFILE);
  }

//  @Parameters(name = "{index}: {0}")
  public static Stream<Arguments> data() {
    return Stream.of(
      Arguments.of("newFile", 10, 1, 9, 9, Collections.emptyList(), null),
      Arguments.of("changeInImportedModule", 9, 1, 7, 8, Collections.emptyList(), null),
      Arguments.of("changeInParent", 9, 1, 6, 8, Collections.emptyList(), null),
      Arguments.of("changeInPackageInit", 9, 1, 7, 8, Collections.emptyList(), null),
      Arguments.of("changeInRelativeImport", 9, 2, 4, 7, Collections.emptyList(), null),
      Arguments.of("deletedFile", 8, 0, 7, 8, List.of("submodule.py"), null),
      Arguments.of("duplication", 10, 1, 9, 9, Collections.emptyList(), 55)
    );
  }


  @ParameterizedTest
  @MethodSource("data")
  void pr_analysis_logs(String scenario,
    int expectedTotalFiles,
    int expectedRecomputed,
    int expectedFullySkipped,
    int expectedPartiallySkipped,
    List<String> deletedFiles,
    Integer expectedDuplication) throws IOException {
    init(scenario, expectedTotalFiles, expectedRecomputed, expectedFullySkipped, expectedPartiallySkipped, deletedFiles, expectedDuplication);
    File tempDirectory = temporaryFolder.toFile();
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();

    // Analyze base commit
    analyzeAndAssertBaseCommit(tempDirectory, litsDifferencesFile);

    // Analyze the changed branch
    setUpChanges(tempDirectory, scenario);
    SonarScanner build = prepareScanner(tempDirectory, PR_ANALYSIS_PROJECT_KEY, scenario, litsDifferencesFile)
      .setProperty("sonar.pullrequest.key", PR_KEY)
      .setProperty("sonar.pullrequest.branch", PR_BRANCH_NAME);

    BuildResult result = ORCHESTRATOR.executeBuild(build);
    assertPrAnalysisLogs(result);
    assertMeasures();
  }

  @ParameterizedTest
  @MethodSource("data")
  void pr_analysis_issues(String scenario,
    int expectedTotalFiles,
    int expectedRecomputed,
    int expectedFullySkipped,
    int expectedPartiallySkipped,
    List<String> deletedFiles,
    Integer expectedDuplication) throws IOException {
    init(scenario, expectedTotalFiles, expectedRecomputed, expectedFullySkipped, expectedPartiallySkipped, deletedFiles, expectedDuplication);
    File tempDirectory = temporaryFolder.toFile();
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();

    // Analyze base commit
    analyzeAndAssertBaseCommit(tempDirectory, litsDifferencesFile);

    // Analyze the changed branch

    // By default, when performing branch analysis, the incremental analysis is disabled.
    // Still, while testing, we want to run branch analysis, to take full advantage of LITS by comparing the total issues that are raised
    // (if we set up a PR Analysis, LITS will fail comparing all the expected issues).
    // Thus, in the test we perform branch analysis, and we manually enable incremental analysis for testing purposes.
    setUpChanges(tempDirectory, scenario);
    SonarScanner build = prepareScanner(tempDirectory, PR_ANALYSIS_PROJECT_KEY, scenario, litsDifferencesFile).setProperty("sonar.python.skipUnchanged", "true");

    BuildResult result = ORCHESTRATOR.executeBuild(build);

    // Check expected issues
    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();
    assertPrAnalysisLogs(result);
  }

  private void assertPrAnalysisLogs(BuildResult result) {
    String expectedRecomputedLog = String.format("Cached information of global symbols will be used for %d out of %d main files. Global symbols will be recomputed for the remaining files.",
      expectedTotalFiles - expectedRecomputed, expectedTotalFiles);

    String expectedFullySkippedLog = String.format("Fully optimized analysis can be performed for %d out of %d files.",
      expectedFullySkipped, expectedTotalFiles);

    String expectedPartiallySkippedLog = String.format("Partially optimized analysis can be performed for %d out of %d files.",
      expectedPartiallySkipped, expectedTotalFiles);

    String expectedFinalLog = String.format("The Python analyzer was able to leverage cached data from previous analyses for %d out of %d files. These files were not parsed.",
      expectedPartiallySkipped, expectedTotalFiles);

    assertThat(result.getLogs())
      .contains(expectedRecomputedLog)
      .contains(expectedPartiallySkippedLog)
      .contains(expectedFullySkippedLog)
      .contains(expectedFinalLog);
  }

  private void assertMeasures() {
    if (expectedDuplicatedLines != null) {
      var duplicatedLines = getMeasure(ORCHESTRATOR, PR_KEY, PR_ANALYSIS_PROJECT_KEY + ":index_duplicate.py", "duplicated_lines");
      assertThat(duplicatedLines)
        .isNotNull();
      assertThat(Integer.parseInt(duplicatedLines.getValue()))
        .isEqualTo(expectedDuplicatedLines);
    }
  }

  private void analyzeAndAssertBaseCommit(File tempFile, File litsDifferencesFile) throws IOException {
    FileUtils.copyDirectory(new File("../sources_pr_analysis", "baseCommit"), tempFile);

    SonarScanner build = prepareScanner(tempFile, PR_ANALYSIS_PROJECT_KEY, "baseCommit", litsDifferencesFile);
    ORCHESTRATOR.executeBuild(build);

    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();
  }

  private void setUpChanges(File tempDirectory, String scenario) throws IOException {
    Arrays.stream(tempDirectory.listFiles(f -> deletedFiles.contains(f.getName()))).forEach(File::delete);
    FileUtils.copyDirectory(new File("../sources_pr_analysis", scenario), tempDirectory);
  }

  private SonarScanner prepareScanner(File path, String projectKey, String scenario, File litsDifferencesFile) throws IOException{
    return SonarScanner.create(path)
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectVersion("1")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("sonar.lits.dump.old", FileLocation.of("src/test/resources/expected_pr_analysis/" + scenario).getFile().getAbsolutePath())
      .setProperty("sonar.lits.dump.new", FileLocation.of("target/actual").getFile().getAbsolutePath())
      .setProperty("sonar.lits.differences", litsDifferencesFile.getAbsolutePath())
      .setProperty("sonar.internal.analysis.failFast", "true")
      .setEnvironmentVariable("SONAR_RUNNER_OPTS", "-Xmx2000m");
  }


}
