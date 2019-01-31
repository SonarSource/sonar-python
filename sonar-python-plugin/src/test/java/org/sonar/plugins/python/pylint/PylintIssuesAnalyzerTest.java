/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import org.junit.Test;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PylintIssuesAnalyzerTest {

  private static final Logger LOG = Loggers.get(PylintIssuesAnalyzerTest.class);

  @Test
  public void shouldParseCorrectly() {
    String resourceName = "/org/sonar/plugins/python/pylint/sample_pylint_output.txt";
    String pathName = getClass().getResource(resourceName).getPath();
    String pylintConfigPath = null;
    String pylintPath = null;
    List<String> lines = readFile(pathName);
    List<Issue> issues = analyzer(pylintPath, pylintConfigPath).parseOutput(lines);
    assertThat(issues.size()).isEqualTo(21);
  }

  @Test
  public void shouldParseCorrectlyNewFormat() {
    String resourceName = "/org/sonar/plugins/python/pylint/sample_pylint_output_new_format.txt";
    String pathName = getClass().getResource(resourceName).getPath();
    String pylintConfigPath = null;
    String pylintPath = null;
    List<String> lines = readFile(pathName);
    List<Issue> issues = analyzer(pylintPath, pylintConfigPath).parseOutput(lines);
    assertThat(issues.size()).isEqualTo(1);
    assertThat(issues.get(0).getRuleId()).isEqualTo("C0111");
  }

  @Test
  public void shouldParseCorrectlyOutputWithWindowsPaths() {
    String resourceName = "/org/sonar/plugins/python/pylint/sample_pylint_output_with_win_paths.txt";
    String pathName = getClass().getResource(resourceName).getPath();
    String pylintConfigPath = null;
    String pylintPath = null;
    List<String> lines = readFile(pathName);
    List<Issue> issues = analyzer(pylintPath, pylintConfigPath).parseOutput(lines);
    assertThat(issues.size()).isEqualTo(1);
  }

  @Test
  public void shouldMapIssuesIdsCorrectly() {
    String resourceOld = "/org/sonar/plugins/python/pylint/sample_pylint_output_oldids.txt";
    String resourceNew = "/org/sonar/plugins/python/pylint/sample_pylint_output_newids.txt";
    String pathNameOld = getClass().getResource(resourceOld).getPath();
    String pathNameNew = getClass().getResource(resourceNew).getPath();
    String pylintConfigPath = null;
    String pylintPath = null;
    List<String> linesOld = readFile(pathNameOld);
    List<String> linesNew = readFile(pathNameNew);
    List<Issue> issuesOld = analyzer(pylintPath, pylintConfigPath).parseOutput(linesOld);
    List<Issue> issuesNew = analyzer(pylintPath, pylintConfigPath).parseOutput(linesNew);
    assertThat(getIds(issuesOld)).isEqualTo(getIds(issuesNew));
  }

  @Test
  public void shouldWorkWithValidCustomConfig() {
    String resourceName = "/org/sonar/plugins/python/pylint/pylintrc_sample";
    String pylintConfigPath = getClass().getResource(resourceName).getPath();
    String pylintPath = null;
    analyzer(pylintPath, pylintConfigPath);
  }

  @Test(expected = IllegalStateException.class)
  public void shouldFailIfGivenInvalidConfig() {
    String pylintConfigPath = "xx_path_that_doesnt_exist_xx";
    String pylintPath = null;
    analyzer(pylintPath, pylintConfigPath);
  }

  @Test
  public void shouldInstantiateWhenGivenValidParams() {
    String pylintrcResource = "/org/sonar/plugins/python/pylint/pylintrc_sample";
    String pylintrcPath = getClass().getResource(pylintrcResource).getPath();
    String executableResource = "/org/sonar/plugins/python/pylint/executable";
    String executablePath = getClass().getResource(executableResource).getPath();
    final String[] validParameters =
      {
        null, null,
        executablePath, null,
        null, pylintrcPath,
        executablePath, pylintrcPath
      };

    int numberOfParams = validParameters.length;
    for(int i = 0; i<numberOfParams-1; i+=2){
      try{
        analyzer(validParameters[i], validParameters[i + 1]);
      } catch (IllegalStateException se) {
        assert(false);
      }
    }
  }


  @Test
  public void shouldFailIfGivenInvalidParams() {
    final String NOT_EXISTING_PATH = "notexistingpath";
    final String[] invalidParameters =
      {
        null, NOT_EXISTING_PATH,
        NOT_EXISTING_PATH, null,
        NOT_EXISTING_PATH, NOT_EXISTING_PATH
      };

    int exceptionCount = 0;
    int numberOfParams = invalidParameters.length;
    for(int i = 0; i<numberOfParams-1; i+=2){
      try{
        analyzer(invalidParameters[i], invalidParameters[i + 1]);
        assert(false);
      } catch (IllegalStateException se) {
        exceptionCount++;
      }
    }
    assertThat(exceptionCount).isEqualTo(3);
  }

  private static List<String> readFile(String path) {
    try {
      return Files.readAllLines(Paths.get(path), UTF_8);
    } catch (IOException e) {
      LOG.error("Cannot read the file '{}'", path);
      return Collections.emptyList();
    }
  }

  private static List<String> getIds(List<Issue> issues){
    List<String> ids = new LinkedList<>();
    for(Issue issue: issues) {
      ids.add(issue.getRuleId());
    }
    return ids;
  }

  private static PylintIssuesAnalyzer analyzer(String pylintPath, String pylintConfigPath) {
    PylintArguments arguments = mock(PylintArguments.class);
    return new PylintIssuesAnalyzer(pylintPath, pylintConfigPath, arguments);
  }

}
