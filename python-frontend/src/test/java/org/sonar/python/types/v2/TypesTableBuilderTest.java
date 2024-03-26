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
package org.sonar.python.types.v2;

import java.nio.file.Path;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.TestPythonFile;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.pytype.json.PyTypeTableReader;

public class TypesTableBuilderTest {

  @Test
  void test() {
    var file = Path.of("src/test/resources/v2/code/snippet1.py");
    var pythonFile = new TestPythonFile(Path.of("src/test/resources/v2/code"), file);
    var pyTypeTable = PyTypeTableReader.fromJsonPath(Path.of("src/test/resources/v2/code.json"));
    var typesTable = new TypesTable(pyTypeTable);
    var typesTableBuilder = new TypesTableBuilder(typesTable, pythonFile);
    var fileInput = parseFile(pythonFile);
    typesTableBuilder.annotate(fileInput);
    
    var dName = TreeUtils.firstChild(fileInput, v -> TreeUtils.toOptionalInstanceOf(NameImpl.class, v)
        .map(NameImpl::name)
        .filter("d"::equals)
        .isPresent()
    ).flatMap(TreeUtils.toOptionalInstanceOfMapper(NameImpl.class))
      .orElse(null);
    Assertions.assertNotNull(dName);
    Assertions.assertNotNull(dName.pythonType());
  }

  private static FileInput parseFile(TestPythonFile file) {
    var parser = PythonParser.create();
    var treeMaker = new PythonTreeMaker();

    var astNode = parser.parse(file.content());
    return treeMaker.fileInput(astNode);
  }

}
