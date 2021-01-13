/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2021 SonarSource SA
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
import com.sonar.orchestrator.locator.FileLocation;
import com.sonar.orchestrator.locator.MavenLocation;
import java.io.File;
import java.nio.file.Files;
import java.util.Collections;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonarsource.analyzer.commons.ProfileGenerator;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

public class PythonRulingTest {

  private static final String SQ_VERSION_PROPERTY = "sonar.runtimeVersion";
  private static final String DEFAULT_SQ_VERSION = "LATEST_RELEASE";
  public static final String PROJECT_KEY = "project";

  @ClassRule
  public static final Orchestrator ORCHESTRATOR = Orchestrator.builderEnv()
    .setSonarVersion(System.getProperty(SQ_VERSION_PROPERTY, DEFAULT_SQ_VERSION))
    .addPlugin(FileLocation.byWildcardMavenFilename(new File("../../sonar-python-plugin/target"), "sonar-python-plugin-*.jar"))
    .addPlugin(MavenLocation.of("org.sonarsource.sonar-lits-plugin", "sonar-lits-plugin", "0.8.0.1209"))
    .build();

  @BeforeClass
  public static void prepare_quality_profile() {
    ProfileGenerator.RulesConfiguration parameters = new ProfileGenerator.RulesConfiguration()
      .add("CommentRegularExpression", "message", "The regular expression matches this comment");
    String serverUrl = ORCHESTRATOR.getServer().getUrl();
    File profileFile = ProfileGenerator.generateProfile(serverUrl, "py", "python", parameters, Collections.emptySet());
    ORCHESTRATOR.getServer().restoreProfile(FileLocation.of(profileFile));
  }

  @Test
  public void test() throws Exception {
    ORCHESTRATOR.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "rules");
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();
    SonarScanner build = SonarScanner.create(FileLocation.of("../sources").getFile())
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1")
      .setLanguage("py")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("dump.old", FileLocation.of("src/test/resources/expected").getFile().getAbsolutePath())
      .setProperty("dump.new", FileLocation.of("target/actual").getFile().getAbsolutePath())
      .setProperty("sonar.cpd.exclusions", "**/*")
      .setProperty("lits.differences", litsDifferencesFile.getAbsolutePath())
      .setProperty("sonar.internal.analysis.failFast", "true")
      .setEnvironmentVariable("SONAR_RUNNER_OPTS", "-Xmx2000m");
    ORCHESTRATOR.executeBuild(build);

    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();
  }

}
