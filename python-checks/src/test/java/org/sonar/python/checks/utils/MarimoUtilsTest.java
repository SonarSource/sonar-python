/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.utils;

import com.sonar.sslr.api.AstNode;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MarimoUtilsTest {

  @Test
  void isTreeInMarimoDecoratedFunction_returnsFalseWhenNotInsideFunction() {
    FileInput fileInput = parse("""
      import marimo
      x = 1
      """);

    SubscriptionContext ctx = mockSubscriptionContext();

    assertThat(MarimoUtils.isTreeInMarimoDecoratedFunction(fileInput, ctx)).isFalse();
  }

  @Test
  void isTreeInMarimoDecoratedFunction_returnsTrueWhenInsideAppCellFunction() {
    String content = """
      import marimo
      app = marimo.App()
      @app.cell
      def some_cell():
          return 1
      """;

    FileInput fileInput = parse(content);
    var typeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", "test.py", content);
    var symbolTableV2 = new SymbolTableBuilderV2(fileInput).build();
    new TypeInferenceV2(typeTable, mockFile, symbolTableV2, "").inferModuleType(fileInput);

    SubscriptionContext ctx = mock(SubscriptionContext.class);
    when(ctx.typeTable()).thenReturn(typeTable);

    var statements = fileInput.statements().statements();
    var functionDef = (FunctionDef) statements.get(statements.size() - 1);
    var returnStatement = functionDef.body().statements().get(0);

    assertThat(MarimoUtils.isTreeInMarimoDecoratedFunction(returnStatement, ctx)).isTrue();
  }

  private static SubscriptionContext mockSubscriptionContext() {
    SubscriptionContext ctx = mock(SubscriptionContext.class);
    when(ctx.typeTable()).thenReturn(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    return ctx;
  }

  private static FileInput parse(String content) {
    PythonParser parser = PythonParser.create();
    AstNode astNode = parser.parse(content);
    return new PythonTreeMaker().fileInput(astNode);
  }
}
