/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2023 SonarSource SA
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
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.junit4.OrchestratorRule;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.it.RulingHelper.bugRuleKeys;
import static org.sonar.python.it.RulingHelper.getOrchestrator;

// Ruling test for bug rules, to ensure they are properly tested without slowing down the CI
public class PythonExtendedRulingTest {

  @ClassRule
  public static final OrchestratorRule ORCHESTRATOR = getOrchestrator();

  @BeforeClass
  public static void prepare_quality_profile() throws IOException {
    List<String> ruleKeys = bugRuleKeys();
    String pythonProfile = RulingHelper.profile("customProfile", "py", "python", ruleKeys);
    RulingHelper.loadProfile(ORCHESTRATOR, pythonProfile);
  }

  @Test
  public void test_airflow() throws IOException {
    SonarScanner build = buildWithCommonProperties("airflow");
    build.setProperty("sonar.sources", "airflow");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  public void test_archery() throws IOException {
    executeBuild(buildWithCommonProperties("Archery"));
  }

  @Test
  public void test_autokeras() throws IOException {
    executeBuild(buildWithCommonProperties("autokeras"));
  }

  @Test
  public void test_black() throws IOException {
    SonarScanner build = buildWithCommonProperties("black");
    build.setProperty("sonar.sources", "src");
    build.setProperty("sonar.tests", "tests");
    build.setProperty("sonar.test.exclusions", "tests/data/async_as_identifier.py");
    executeBuild(build);
  }

  @Test
  public void test_calibre() throws IOException {
    SonarScanner build = buildWithCommonProperties("calibre");
    build.setProperty("sonar.sources", "src");
    executeBuild(build);
  }

  @Test
  public void test_chalice() throws IOException {
    SonarScanner build = buildWithCommonProperties("chalice");
    build.setProperty("sonar.sources", "chalice");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  public void test_django_shop() throws IOException {
    SonarScanner build = buildWithCommonProperties("django-shop");
    build.setProperty("sonar.sources", "shop");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  public void test_indico() throws IOException {
    SonarScanner build = buildWithCommonProperties("indico");
    build.setProperty("sonar.sources", "indico");
    executeBuild(build);
  }

  @Test
  public void test_LibCST() throws IOException {
    SonarScanner build = buildWithCommonProperties("LibCST");
    build.setProperty("sonar.sources", "libcst");
    build.setProperty("sonar.tests", "libcst/tests");
    build.setProperty("sonar.test.inclusions", "**/");
    executeBuild(build);
  }

  @Test
  public void test_nltk() throws IOException {
    SonarScanner build = buildWithCommonProperties("nltk");
    build.setProperty("sonar.sources", ".");
    build.setProperty("sonar.exclusions", "**/test/**/*");
    executeBuild(build);
  }

  @Test
  public void test_saleor() throws IOException {
    SonarScanner build = buildWithCommonProperties("saleor");
    build.setProperty("sonar.sources", "saleor");
    executeBuild(build);
  }

  @Test
  public void test_salt() throws IOException {
    SonarScanner build = buildWithCommonProperties("salt");
    build.setProperty("sonar.sources", "salt");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  public void test_scikit_learn() throws IOException {
    SonarScanner build = buildWithCommonProperties("scikit-learn");
    build.setProperty("sonar.sources", "sklearn");
    executeBuild(build);
  }

  @Test
  public void test_timesketch() throws IOException {
    SonarScanner build = buildWithCommonProperties("timesketch");
    build.setProperty("sonar.sources", "timesketch");
    build.setProperty("sonar.test.inclusions", "**/*_test.py");
    executeBuild(build);
  }

  public SonarScanner buildWithCommonProperties(String projectKey) {
    return buildWithCommonProperties(projectKey, projectKey);
  }

  public SonarScanner buildWithCommonProperties(String projectKey, String projectName) {
    ORCHESTRATOR.getServer().provisionProject(projectKey, projectKey);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(projectKey, "py", "customProfile");
    return SonarScanner.create(FileLocation.of(String.format("../sources_extended/%s", projectName)).getFile())
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectVersion("1")
      .setLanguage("py")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("sonar.lits.dump.old", FileLocation.of(String.format("src/test/resources/expected_extended/%s", projectKey)).getFile().getAbsolutePath())
      .setProperty("sonar.lits.dump.new", FileLocation.of(String.format("target/actual_extended/%s", projectKey)).getFile().getAbsolutePath())
      .setProperty("sonar.cpd.exclusions", "**/*")
      .setProperty("sonar.internal.analysis.failFast", "true")
      .setEnvironmentVariable("SONAR_RUNNER_OPTS", "-Xmx2000m");
  }

  void executeBuild(SonarScanner build) throws IOException {
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();
    build.setProperty("sonar.lits.differences", litsDifferencesFile.getAbsolutePath());
    ORCHESTRATOR.executeBuild(build);
    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();
  }
}
