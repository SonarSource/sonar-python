/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.google.common.base.Charsets;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class PylintIssuesAnalyzerIT {

  @Rule
  public TemporaryFolder tempFolder = new TemporaryFolder();

  @Test
  public void issuesTest() throws Exception {
    String pylintrcResource = "/org/sonar/plugins/python/pylint/pylintrc_sample";
    String codeChunksResource = "/org/sonar/plugins/python/code_chunks_2.py";
    String pylintConfigPath = getClass().getResource(pylintrcResource).getPath();
    String codeChunksPathName = getClass().getResource(codeChunksResource).getPath();
    String pylintPath = null;
    File out = tempFolder.newFile();

    List<Issue> issues = new PylintIssuesAnalyzer(pylintPath, pylintConfigPath).analyze(codeChunksPathName, Charsets.UTF_8, out);
    assertThat(issues).isNotEmpty();
  }

}
