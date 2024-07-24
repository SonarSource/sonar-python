/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.internal.apachecommons.lang.StringUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.plugins.python.TestUtils.createInputFile;

class IpynbNotebookParserTest {
  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python").getAbsoluteFile();

  @Test
  void testParseNotebook() {
    var inputFile = createInputFile(baseDir, "notebook.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var result = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(result.locationMap().keySet()).hasSize(20);
    assertThat(StringUtils.countMatches(result.aggregatedSource(), IpynbNotebookParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER))
      .isEqualTo(7);
    assertThat(result.locationMap()).extracting(map -> map.get(17)).isEqualTo(new IpynbNotebookParser.IPythonLocation(64, 27, Map.of(6, 21, 20, 37, -1, 3)));

    // The wrapped file changes the lines of the notebook
    assertThat(result.locationMap()).extracting(map -> map.get(22)).isEqualTo(new IpynbNotebookParser.IPythonLocation(84, 15, Map.of(6, 21, 15, 32, -1, 3)));
    assertThat(result.locationMap()).extracting(map -> map.get(23)).isEqualTo(new IpynbNotebookParser.IPythonLocation(84, 37, Map.of(6, 21, 23, 40, -1, 3)));

    assertThat(result.locationMap()).extracting(map -> map.get(25))
      .isEqualTo(new IpynbNotebookParser.IPythonLocation(91, 15, Map.of(4, 19, 39, 62, 41, 64, 42, 65, 46, 71, -1, 7)));
    assertThat(result.locationMap()).extracting(map -> map.get(26)).isEqualTo(new IpynbNotebookParser.IPythonLocation(91, 71, Map.of(-1, 0)));
  }

  @Test
  void testParseInvalidNotebook() {
    var inputFile = createInputFile(baseDir, "invalid_notebook.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    assertThatThrownBy(() -> IpynbNotebookParser.parseNotebook(inputFile))
      .isInstanceOf(IllegalStateException.class)
      .hasMessageContaining("Unexpected token");
  }

}
