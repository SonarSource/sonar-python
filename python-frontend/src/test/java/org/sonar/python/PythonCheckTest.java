/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python;

import java.io.File;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonInputFileContext;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class PythonCheckTest {

  private static final File FILE = new File("src/test/resources/file.py");
  private static final String MESSAGE = "message";

  private static List<PreciseIssue> scanFileForIssues(PythonCheck check) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(PythonCheckTest.FILE);
    check.scanFile(context);
    return context.getIssues();
  }

  @Test
  void test() {
    PythonVisitorCheck check = new PythonVisitorCheck() {
      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        FunctionSymbol symbol = ((FunctionSymbol) pyFunctionDefTree.name().symbol());
        addIssue(pyFunctionDefTree.name(), pyFunctionDefTree.name().name()).secondary(symbol.definitionLocation(), null);
        super.visitFunctionDef(pyFunctionDefTree);
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(check);

    assertThat(issues).hasSize(2);
    PreciseIssue firstIssue = issues.get(0);

    assertThat(firstIssue.cost()).isNull();
    assertThat(firstIssue.secondaryLocations()).hasSize(1);

    IssueLocation primaryLocation = firstIssue.primaryLocation();
    assertThat(primaryLocation.message()).isEqualTo("hello");

    assertThat(primaryLocation.startLine()).isEqualTo(1);
    assertThat(primaryLocation.endLine()).isEqualTo(1);
    assertThat(primaryLocation.startLineOffset()).isEqualTo(4);
    assertThat(primaryLocation.endLineOffset()).isEqualTo(9);

    assertThat(firstIssue.quickFixes()).isEmpty();
    PythonTextEdit textEdit = Mockito.mock(PythonTextEdit.class);
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("New Quickfix")
      .addTextEdit(textEdit)
      .build();

    firstIssue.addQuickFix(quickFix);
    firstIssue.addQuickFix(quickFix);

    assertThat(firstIssue.quickFixes()).containsExactly(quickFix, quickFix);
  }

  @Test
  void test_cost() {
    PythonVisitorCheck check = new PythonVisitorCheck() {
      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        addIssue(pyFunctionDefTree.name(), MESSAGE).withCost(42);
        super.visitFunctionDef(pyFunctionDefTree);
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(check);
    PreciseIssue firstIssue = issues.get(0);
    assertThat(firstIssue.cost()).isEqualTo(42);
  }

  @Test
  void test_secondary_location() {
    PythonVisitorCheck check = new PythonVisitorCheck() {

      private PreciseIssue preciseIssue;

      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        preciseIssue = addIssue(pyFunctionDefTree.name(), MESSAGE).secondary(pyFunctionDefTree.defKeyword(), "def keyword");
        super.visitFunctionDef(pyFunctionDefTree);
      }
      @Override
      public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
        preciseIssue.secondary(pyReturnStatementTree, "return statement");
        super.visitReturnStatement(pyReturnStatementTree);
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(check);

    List<IssueLocation> secondaryLocations = issues.get(0).secondaryLocations();
    assertThat(secondaryLocations).hasSize(2);

    IssueLocation firstSecondaryLocation = secondaryLocations.get(0);
    IssueLocation secondSecondaryLocation = secondaryLocations.get(1);

    assertThat(firstSecondaryLocation.message()).isEqualTo("def keyword");
    assertThat(firstSecondaryLocation.startLine()).isEqualTo(1);
    assertThat(firstSecondaryLocation.startLineOffset()).isEqualTo(0);
    assertThat(firstSecondaryLocation.endLine()).isEqualTo(1);
    assertThat(firstSecondaryLocation.endLineOffset()).isEqualTo(3);

    assertThat(secondSecondaryLocation.message()).isEqualTo("return statement");
    assertThat(secondSecondaryLocation.startLine()).isEqualTo(3);
    assertThat(secondSecondaryLocation.startLineOffset()).isEqualTo(4);
    assertThat(secondSecondaryLocation.endLine()).isEqualTo(4);
    assertThat(secondSecondaryLocation.endLineOffset()).isEqualTo(5);
  }


  @Test
  void test_scope() {
    PythonVisitorCheck check = new PythonVisitorCheck() {};
    assertThat(check.scope()).isEqualTo(PythonCheck.CheckScope.MAIN);
  }

  @Test
  void test_scanWithoutParsing() {
    PythonVisitorCheck check = new PythonVisitorCheck() {};
    assertThat(check.scanWithoutParsing(mock(PythonInputFileContext.class))).isTrue();
  }
}
