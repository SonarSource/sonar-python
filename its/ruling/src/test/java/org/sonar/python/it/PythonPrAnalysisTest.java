/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2022 SonarSource SA
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
package org.sonar.python.it;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.container.Edition;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.it.RulingHelper.getOrchestrator;

@RunWith(Parameterized.class)
public class PythonPrAnalysisTest {

  @ClassRule
  public static final Orchestrator ORCHESTRATOR = getOrchestrator(Edition.DEVELOPER);

  private static final String PR_ANALYSIS_PROJECT_KEY = "prAnalysis";
  private static final String INCREMENTAL_ANALYSIS_PROFILE = "incrementalPrAnalysis";

  @Rule
  public TemporaryFolder temporaryFolder = new TemporaryFolder();

  private final String scenario;
  private final int expectedTotalFiles;
  private final int expectedRecomputed;
  private final int expectedSkipped;
  private final List<String> deletedFiles;

  public PythonPrAnalysisTest(String scenario, int expectedTotalFiles, int expectedRecomputed, int expectedSkipped, List<String> deletedFiles) {
    this.scenario = scenario;
    this.expectedTotalFiles = expectedTotalFiles;
    this.expectedRecomputed = expectedRecomputed;
    this.expectedSkipped = expectedSkipped;
    this.deletedFiles = deletedFiles;
  }

  @BeforeClass
  public static void prepare_quality_profile() throws IOException {
    ORCHESTRATOR.getServer().provisionProject(PR_ANALYSIS_PROJECT_KEY, PR_ANALYSIS_PROJECT_KEY);

    // Create and load quality profile
    String profile = RulingHelper.profile(INCREMENTAL_ANALYSIS_PROFILE, "py", "python", List.of("S5713"));
    RulingHelper.loadProfile(ORCHESTRATOR, profile);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PR_ANALYSIS_PROJECT_KEY, "py", INCREMENTAL_ANALYSIS_PROFILE);
  }

  @Parameters(name = "{index}: {0}")
  public static Collection<Object[]> data() {
    return List.of(new Object[][] {
      // {<scenario>, <total files>, <recomputed>, <skipped>, <deleted>}
      {"newFile", 10, 1, 9, Collections.emptyList()},
      {"changeInImportedModule", 9, 1, 7, Collections.emptyList()},
      {"changeInParent", 9, 1, 6, Collections.emptyList()},
      {"changeInPackageInit", 9, 1, 7, Collections.emptyList()},
      {"changeInRelativeImport", 9, 2, 4, Collections.emptyList()},
      {"deletedFile", 8, 0, 7, List.of("submodule.py")}}
    );
  }

  @Test
  public void pr_analysis_logs() throws IOException {
    File tempDirectory = temporaryFolder.newFolder();
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();

    // Analyze base commit
    analyzeAndAssertBaseCommit(tempDirectory, litsDifferencesFile);

    // Analyze the changed branch
    setUpChanges(tempDirectory, scenario);
    SonarScanner build = prepareScanner(tempDirectory, PR_ANALYSIS_PROJECT_KEY, scenario, litsDifferencesFile)
      .setProperty("sonar.pullrequest.key", "1")
      .setProperty("sonar.pullrequest.branch", "incremental");

    BuildResult result = ORCHESTRATOR.executeBuild(build);
    assertPrAnalysisLogs(result);
  }

  @Test
  public void pr_analysis_issues() throws IOException {
    File tempDirectory = temporaryFolder.newFolder();
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

    String expectedRegularAnalysisLog = String.format("Optimized analysis can be performed for %d out of %d files.",
      expectedSkipped, expectedTotalFiles);

    String expectedFinalLog = String.format("The Python analyzer was able to leverage cached data from previous analyses for %d out of %d files. These files were not parsed.",
      expectedSkipped, expectedTotalFiles);

    assertThat(result.getLogs())
      .contains(expectedRecomputedLog)
      .contains(expectedRegularAnalysisLog)
      .contains(expectedFinalLog);
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
      .setLanguage("py")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("sonar.lits.dump.old", FileLocation.of("src/test/resources/expected_pr_analysis/" + scenario).getFile().getAbsolutePath())
      .setProperty("sonar.lits.dump.new", FileLocation.of("target/actual").getFile().getAbsolutePath())
      .setProperty("sonar.cpd.exclusions", "**/*")
      .setProperty("sonar.lits.differences", litsDifferencesFile.getAbsolutePath())
      .setProperty("sonar.internal.analysis.failFast", "true")
      .setEnvironmentVariable("SONAR_RUNNER_OPTS", "-Xmx2000m");
  }


}
