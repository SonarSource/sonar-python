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

  public PythonPrAnalysisTest(String scenario) {
    this.scenario = scenario;
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
  public static Object[] data() {
    return new String[] {"newFile", "changeInImportedModule", "changeInParent", "changeInPackageInit", "changeInRelativeImport"};
  }

  @Test
  public void pr_analysis() throws IOException {
    File tempFile = temporaryFolder.newFolder();

    // Analyze base commit
    FileUtils.copyDirectory(new File("../sources_pr_analysis", "baseCommit"), tempFile);

    File litsDifferencesFile = FileLocation.of("target/differences").getFile();
    SonarScanner build = prepareScanner(tempFile, PR_ANALYSIS_PROJECT_KEY, "baseCommit", litsDifferencesFile);
    ORCHESTRATOR.executeBuild(build);

    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();

    // Analyze the changed branch
    executePrAnalysisOnBranch(scenario, tempFile, litsDifferencesFile);
  }

  private BuildResult executePrAnalysisOnBranch(String scenario, File tempDirectory, File litsDifferencesFile) throws IOException {
    FileUtils.copyDirectory(new File("../sources_pr_analysis", scenario), tempDirectory);

    SonarScanner build = prepareScanner(tempDirectory, PR_ANALYSIS_PROJECT_KEY, scenario, litsDifferencesFile)
      .setProperty("sonar.pullrequest.key", "1")
      .setProperty("sonar.pullrequest.branch", "incremental");

    BuildResult result = ORCHESTRATOR.executeBuild(build);

    // Check expected issues
    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();

    // TODO: Check logs

    return result;
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
