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
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.TestPythonVisitorRunner;

class SymbolTableBuilderV2Test {

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

    Assertions.assertNotNull(fileInput.statements());

    {
      var importStatement = (ImportName) fileInput.statements().statements().get(0);
      var libName = importStatement.modules().get(0).dottedName().names().get(0);
      var symbol = libName.symbolV2();
      Assertions.assertNotNull(symbol);
      Assertions.assertEquals("lib", symbol.name());
      Assertions.assertEquals(3, symbol.usages().size());
    }

    {
      var assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(1);
      var vName = (Name) assignmentStatement.children().get(0).children().get(0);
      var vNameSymbol = vName.symbolV2();
      Assertions.assertNotNull(vNameSymbol);
      Assertions.assertEquals("v", vNameSymbol.name());
      Assertions.assertEquals(1, vNameSymbol.usages().size());

      var libName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      var symbol = libName.symbolV2();
      Assertions.assertNotNull(symbol);
      Assertions.assertEquals("lib", symbol.name());
      Assertions.assertEquals(3, symbol.usages().size());
    }

    {
      var assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(2);
      var aName = (Name) assignmentStatement.children().get(0).children().get(0);
      var aNameSymbol = aName.symbolV2();
      Assertions.assertNotNull(aNameSymbol);
      Assertions.assertEquals("a", aNameSymbol.name());
      Assertions.assertEquals(2, aNameSymbol.usages().size());

      var libName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      var symbol = libName.symbolV2();
      Assertions.assertNotNull(symbol);
      Assertions.assertEquals("lib", symbol.name());
      Assertions.assertEquals(3, symbol.usages().size());
    }

    {
      var assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(3);
      var bName = (Name) assignmentStatement.children().get(0).children().get(0);
      var bNameSymbol = bName.symbolV2();
      Assertions.assertNotNull(bNameSymbol);
      Assertions.assertEquals("b", bNameSymbol.name());
      Assertions.assertEquals(1, bNameSymbol.usages().size());

      var aName = (Name) assignmentStatement.children().get(2).children().get(0).children().get(0);
      var aNameSymbol = aName.symbolV2();
      Assertions.assertNotNull(aNameSymbol);
      Assertions.assertEquals("a", aNameSymbol.name());
      Assertions.assertEquals(2, aNameSymbol.usages().size());
    }

    {
      var assignmentStatement = (AssignmentStatement) fileInput.statements().statements().get(4);
      var xName = (Name) assignmentStatement.children().get(0).children().get(0);
      var xNameSymbol = xName.symbolV2();
      Assertions.assertNotNull(xNameSymbol);
      Assertions.assertEquals("x", xNameSymbol.name());
      Assertions.assertEquals(1, xNameSymbol.usages().size());
    }

    {
      var functionDef = (FunctionDef) fileInput.statements().statements().get(5);
      var scriptFunctionName = (Name) functionDef.name();
      var scriptFunctionNameSymbol = scriptFunctionName.symbolV2();
      Assertions.assertNotNull(scriptFunctionNameSymbol);
      Assertions.assertEquals("script_do_something", scriptFunctionNameSymbol.name());
      Assertions.assertEquals(2, scriptFunctionNameSymbol.usages().size());
    }

    {
      var functionCallName = (Name) fileInput.statements().statements().get(6).children().get(0).children().get(0);
      var functionCallNameSymbol = functionCallName.symbolV2();
      Assertions.assertNotNull(functionCallNameSymbol);
      Assertions.assertEquals("script_do_something", functionCallNameSymbol.name());
      Assertions.assertEquals(2, functionCallNameSymbol.usages().size());
    }

  }
}
