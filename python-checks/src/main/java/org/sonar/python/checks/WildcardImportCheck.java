/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S2208")
public class WildcardImportCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Import only needed names or import the module and then use its members.";

  private static class WildcardImportVisitor extends BaseTreeVisitor {
    private boolean shouldRaiseIssues = false;
    private List<ImportFrom> wildcardImports = new ArrayList<>();

    @Override
    public void visitImportFrom(ImportFrom pyImportFromTree) {
      if (pyImportFromTree.isWildcardImport()) {
        wildcardImports.add(pyImportFromTree);
      }
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      shouldRaiseIssues = true;
      // No point visiting further - wildcard imports are only allowed on the module level
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      shouldRaiseIssues = true;
      // No point visiting further - wildcard imports are only allowed on the module level
    }

    @Override
    public void visitStatementList(StatementList statementList) {
      // We should raise if one of the statements may contain application logic
      this.raiseIssuesIf(() -> statementList.statements().stream().anyMatch(WildcardImportVisitor::mayContainApplicationLogic));
      super.visitStatementList(statementList);
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
      List<ExpressionList> lhsExpressions = pyAssignmentStatementTree.lhsExpressions();
      this.raiseIssuesIf(() -> lhsExpressions.stream().anyMatch(
        expressionList -> expressionList.expressions().stream().anyMatch(WildcardImportVisitor::isDisallowedAssignment)
      ));
    }

    @Override
    public void visitCallExpression(CallExpression pyCallExpressionTree) {
      Symbol symbol = pyCallExpressionTree.calleeSymbol();
      this.raiseIssuesIf(() -> symbol == null || !"warnings.warn".equals(symbol.fullyQualifiedName()));
    }

    private void raiseIssuesIf(BooleanSupplier condition) {
      shouldRaiseIssues = shouldRaiseIssues || condition.getAsBoolean();
    }

    private static boolean isDisallowedAssignment(Expression expression) {
      return !expression.is(Tree.Kind.NAME) || !"__all__".equals(((Name) expression).name());
    }

    private static boolean mayContainApplicationLogic(Tree tree) {
      // Some common control structures and statements that probably describe application logic
      return tree.is(Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT, Tree.Kind.WITH_STMT, Tree.Kind.COMPOUND_ASSIGNMENT);
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      if (ctx.pythonFile().fileName().equals("__init__.py")) {
        // Ignore __init__.py files, as wildcard imports are commonly used to populate those.
        return;
      }

      FileInput fileInput = (FileInput) ctx.syntaxNode();
      StatementList statements = fileInput.statements();
      if (statements == null) {
        return;
      }

      WildcardImportVisitor visitor = new WildcardImportVisitor();
      statements.accept(visitor);

      if (visitor.shouldRaiseIssues) {
        visitor.wildcardImports.forEach(importFrom -> ctx.addIssue(importFrom, MESSAGE));
      }
    });
  }
}
