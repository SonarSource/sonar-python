/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.junit.Test;
import org.sonar.api.utils.SonarException;

public class PythonViolationsAnalyzerTest {
  @Test
  public void shouldParseCorrectly() {
    String resourceName = "/org/sonar/plugins/python/complexity/sample_pylint_output.xml";
    String pathName = getClass().getResource(resourceName).getPath();
    String pylintConfigPath = null;
    String pylintPath = null;
    List<String> lines = readFile(pathName);
    List<Issue> issues = new PythonViolationsAnalyzer(pylintPath, pylintConfigPath).parseOutput(lines);
    assertEquals(issues.size(), 21);
  }

  // @Test
  // public void shouldWorkWithValidCustomConfig() {
  //   String resourceName = "/org/sonar/plugins/python/complexity/pylintrc_sample";
  //   String pylintConfigPath = getClass().getResource(resourceName).getPath();
  //   new PythonViolationsAnalyzer(pylintConfigPath);
  // }

  // @Test(expected = SonarException.class)
  // public void shouldFailIfGivenInvalidConfig() {
  //   String pylintConfigPath = "xx_path_that_doesnt_exist_xx";
  //   new PythonViolationsAnalyzer(pylintConfigPath);
  // }

  @Test
  public void shouldInstantiateWhenGivenValidParams() {
    String pylintrcResource = "/org/sonar/plugins/python/complexity/pylintrc_sample";
    String pylintrcPath = getClass().getResource(pylintrcResource).getPath();
    String executableResource = "/org/sonar/plugins/python/complexity/executable";
    String executablePath = getClass().getResource(executableResource).getPath();
    final String[] VALID_PARAMETERS =
      {
	null, null,
	executablePath, null,
	null, pylintrcPath,
	executablePath, pylintrcPath
      };

    int numberOfParams = VALID_PARAMETERS.length;
    for(int i = 0; i<numberOfParams-1; i+=2){
      try{
	new PythonViolationsAnalyzer(VALID_PARAMETERS[i], VALID_PARAMETERS[i+1]);
      } catch (SonarException se) {
	assert(false);
      }
    }
  }


  @Test
  public void shouldFailIfGivenInvalidParams() {
    final String NOT_EXISTING_PATH = "notexistingpath";
    final String[] INVALID_PARAMETERS =
      {
	null, NOT_EXISTING_PATH,
	NOT_EXISTING_PATH, null,
	NOT_EXISTING_PATH, NOT_EXISTING_PATH
      };

    int numberOfParams = INVALID_PARAMETERS.length;
    for(int i = 0; i<numberOfParams-1; i+=2){
      try{
	new PythonViolationsAnalyzer(INVALID_PARAMETERS[i], INVALID_PARAMETERS[i+1]);
	assert(false);
      } catch (SonarException se) {}
    }
  }

  private List<String> readFile(String path) {
    List<String> lines = new LinkedList<String>();

    try {
      BufferedReader reader = new BufferedReader(new FileReader(path));
      String s = null;

      while ((s = reader.readLine()) != null) {
        lines.add(s);
      }
    } catch (IOException e) {
      System.err.println("Cannot read the file '" + path + "'");
    }

    return lines;
  }
}
