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

import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;

public class TypesTestUtils {

  public static final ModuleType BUILTINS = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()).getModule();

  public static final PythonType INT_TYPE = BUILTINS.resolveMember("int").get();
  public static final PythonType BOOL_TYPE = BUILTINS.resolveMember("bool").get();
  public static final PythonType STR_TYPE = BUILTINS.resolveMember("str").get();
  public static final PythonType LIST_TYPE = BUILTINS.resolveMember("list").get();

  public static FileInput parseAndInferTypes(String... code) {
    return parseAndInferTypes(PythonTestUtils.pythonFile(""), code);
  }

  public static FileInput parseAndInferTypes(PythonFile pythonFile, String... code) {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols(code);
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()), pythonFile, symbolTable).inferTypes(fileInput);
    return fileInput;
  }
}
