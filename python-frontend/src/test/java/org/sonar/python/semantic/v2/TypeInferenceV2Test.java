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
package org.sonar.python.semantic.v2;

import java.io.File;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

class TypeInferenceV2Test {
  private static FileInput fileInput;

  @BeforeAll
  static void init() {
    var context = TestPythonVisitorRunner.createContext(new File("src/test/resources/semantic/v2/script.py"));
    fileInput = context.rootTree();
  }

  @Test
  void test() {
    var pythonFile = PythonTestUtils.pythonFile("script.py");
    var builder = new SymbolTableBuilderV2();
    builder.visitFileInput(fileInput);
    var typeInferenceV2 = new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    fileInput.accept(typeInferenceV2);

    System.out.println("hello");
  }
}
