/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.version.Version;
import com.sonar.orchestrator.build.SonarScannerInstaller;
import com.sonar.orchestrator.config.Configuration;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.sonarsource.analyzer.commons.ProfileGenerator;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.it.RulingHelper.getOrchestrator;

@Execution(ExecutionMode.CONCURRENT)
@ExtendWith(TestDurationMeasureExtension.class)
class PythonRulingTest {


  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = getOrchestrator();
  public static final Configuration CONFIGURATION = ORCHESTRATOR.getConfiguration();

  private static final String PROFILE_NAME = "rules";

  @BeforeAll
  static void prepare_quality_profile() {
    ProfileGenerator.RulesConfiguration parameters = new ProfileGenerator.RulesConfiguration()
      .add("CommentRegularExpression", "message", "The regular expression matches this comment")
      .add("S1451", "headerFormat", "# Copyright 2004 by Harry Zuzan. All rights reserved.");
    String serverUrl = ORCHESTRATOR.getServer().getUrl();
    File profileFile = ProfileGenerator.generateProfile(serverUrl, "py", "python", parameters, Collections.emptySet());
    ORCHESTRATOR.getServer().restoreProfile(FileLocation.of(profileFile));
    File iPythonProfileFile = ProfileGenerator.generateProfile(serverUrl, "ipynb", "ipython", parameters, Collections.emptySet());
    ORCHESTRATOR.getServer().restoreProfile(FileLocation.of(iPythonProfileFile));
  }

  @BeforeAll
  static void install_sonar_scanner() {
    new SonarScannerInstaller(CONFIGURATION.locators()).install(Version.create(SonarScanner.DEFAULT_SCANNER_VERSION), CONFIGURATION.fileSystem().workspace());
  }

  @Test
  void test_airflow() throws IOException {
    SonarScanner build = buildWithCommonProperties("airflow");
    build.setProperty("sonar.sources", "airflow");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  void test_archery() throws IOException {
    executeBuild(buildWithCommonProperties("Archery"));
  }

  @Test
  void test_autokeras() throws IOException {
    executeBuild(buildWithCommonProperties("autokeras"));
  }

  @Test
  void test_biopython() throws IOException {
    executeBuild(buildWithCommonProperties("biopython"));
  }

  @Test
  void test_black() throws IOException {
    SonarScanner build = buildWithCommonProperties("black");
    build.setProperty("sonar.sources", "src");
    build.setProperty("sonar.tests", "tests");
    build.setProperty("sonar.test.exclusions", "tests/data/async_as_identifier.py");
    executeBuild(build);
  }

  @Test
  void test_buildbot() throws IOException {
    SonarScanner build = buildWithCommonProperties("buildbot","buildbot-0.8.6p1");
    build.setProperty("sonar.sources", "buildbot");
    build.setProperty("sonar.tests", "contrib");
    executeBuild(build);
  }

  @Test
  void test_buildbot_slave() throws IOException {
    SonarScanner build = buildWithCommonProperties("buildbot-slave", "buildbot-slave-0.8.6p1");
    build.setProperty("sonar.sources", "buildslave");
    build.setProperty("sonar.tests", "contrib");
    executeBuild(build);
  }

  @Test
  void test_calibre() throws IOException {
    SonarScanner build = buildWithCommonProperties("calibre");
    build.setProperty("sonar.sources", "src");
    executeBuild(build);
  }

  @Test
  void test_celery() throws IOException {
    SonarScanner build = buildWithCommonProperties("celery");
    build.setProperty("sonar.sources", "celery");
    build.setProperty("sonar.tests", "t");
    executeBuild(build);
  }

  @Test
  void test_chalice() throws IOException {
    SonarScanner build = buildWithCommonProperties("chalice");
    build.setProperty("sonar.sources", "chalice");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  void test_django() throws IOException {
    SonarScanner build = buildWithCommonProperties("django", "django-2.2.3");
    build.setProperty("sonar.sources", "django");
    executeBuild(build);
  }

  @Test
  void test_django_cms() throws IOException {
    SonarScanner build = buildWithCommonProperties("django-cms", "django-cms-3.7.1");
    build.setProperty("sonar.sources", "cms");
    build.setProperty("sonar.test", "cms/tests");
    executeBuild(build);
  }

  @Test
  void test_django_shop() throws IOException {
    SonarScanner build = buildWithCommonProperties("django-shop");
    build.setProperty("sonar.sources", "shop");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  void test_docker_compose() throws IOException {
    SonarScanner build = buildWithCommonProperties("docker-compose", "docker-compose-1.24.1");
    build.setProperty("sonar.sources", "compose");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  void test_indico() throws IOException {
    SonarScanner build = buildWithCommonProperties("indico");
    build.setProperty("sonar.sources", "indico");
    executeBuild(build);
  }

  @Test
  void test_keras_tutorials() throws IOException {
    executeBuild(buildWithCommonProperties("keras-tutorials"));
  }

  @Test
  void test_LibCST() throws IOException {
    SonarScanner build = buildWithCommonProperties("LibCST");
    build.setProperty("sonar.sources", "libcst");
    build.setProperty("sonar.tests", "libcst/tests");
    build.setProperty("sonar.test.inclusions", "**/");
    executeBuild(build);
  }

  @Test
  void test_mypy() throws IOException {
    SonarScanner build = buildWithCommonProperties("mypy", "mypy-0.782");
    build.setProperty("sonar.sources", "mypy,mypyc");
    build.setProperty("sonar.exclusions", "**/test/**/*");
    build.setProperty("sonar.tests", "mypy/test,mypyc/test");
    executeBuild(build);
  }

  @Test
  void test_nltk() throws IOException {
    SonarScanner build = buildWithCommonProperties("nltk");
    build.setProperty("sonar.sources", ".");
    build.setProperty("sonar.exclusions", "**/test/**/*");
    executeBuild(build);
  }

  @Test
  void test_numpy() throws IOException {
    SonarScanner build = buildWithCommonProperties("numpy", "numpy-1.16.4");
    build.setProperty("sonar.sources", "numpy");
    build.setProperty("sonar.exclusions", "**/tests/**/*");
    build.setProperty("sonar.tests", "numpy/tests");
    executeBuild(build);
  }

  @Test
  void test_pecos() throws IOException {
    SonarScanner build = buildWithCommonProperties("pecos");
    build.setProperty("sonar.sources", "pecos");
    build.setProperty("sonar.tests", "test");
    executeBuild(build);
  }

  @Test
  void test_saleor() throws IOException {
    SonarScanner build = buildWithCommonProperties("saleor");
    build.setProperty("sonar.sources", "saleor");
    executeBuild(build);
  }

  @Test
  void test_salt() throws IOException {
    SonarScanner build = buildWithCommonProperties("salt");
    // salt is not actually a Python 3.12 project. This is to ensure analysis is performed correctly when the parameter is set.
    build.setProperty("sonar.python.version", "3.12");
    build.setProperty("sonar.sources", "salt");
    build.setProperty("sonar.tests", "tests");
    executeBuild(build);
  }

  @Test
  void test_scikit_learn() throws IOException {
    SonarScanner build = buildWithCommonProperties("scikit-learn");
    build.setProperty("sonar.sources", "sklearn");
    executeBuild(build);
  }

  @Test
  void test_specific_rules() throws IOException {
    // this tests is a hodgepodge of tests which are designed for specific rules
    executeBuild(buildWithCommonProperties("specific-rules"));
  }

  @Test
  void test_tensorflow() throws IOException {
    SonarScanner build = buildWithCommonProperties("tensorflow");
    build.setProperty("sonar.sources", "python");
    executeBuild(build);
  }

  @Test
  void test_timesketch() throws IOException {
    SonarScanner build = buildWithCommonProperties("timesketch");
    build.setProperty("sonar.sources", "timesketch");
    build.setProperty("sonar.test.inclusions", "**/*_test.py");
    executeBuild(build);
  }

  @Test
  void test_tornado() throws IOException {
    SonarScanner build = buildWithCommonProperties("tornado", "tornado-2.3");
    build.setProperty("sonar.sources", "tornado");
    build.setProperty("sonar.exclusions", "**/test/**/*");
    build.setProperty("sonar.tests", "tornado/test");
    executeBuild(build);
  }

  @Test
  void test_twisted() throws IOException {
    SonarScanner build = buildWithCommonProperties("twisted", "twisted-12.1.0");
    build.setProperty("sonar.sources", "twisted");
    build.setProperty("sonar.exclusions", "**/test/**/*");
    build.setProperty("sonar.tests", "twisted/test");
    executeBuild(build);
  }


  public SonarScanner buildWithCommonProperties(String projectKey) {
    return buildWithCommonProperties(projectKey, projectKey);
  }

  public SonarScanner buildWithCommonProperties(String projectKey, String projectName) {
    ORCHESTRATOR.getServer().provisionProject(projectKey, projectKey);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(projectKey, "py", PROFILE_NAME);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(projectKey, "ipynb", PROFILE_NAME);
    return SonarScanner.create(FileLocation.of(String.format("../sources_ruling/%s", projectName)).getFile())
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectVersion("1")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("sonar.lits.dump.old", FileLocation.of(String.format("src/test/resources/expected_ruling/%s", projectKey)).getFile().getAbsolutePath())
      .setProperty("sonar.lits.dump.new", FileLocation.of(String.format("target/actual_ruling/%s", projectKey)).getFile().getAbsolutePath())
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
