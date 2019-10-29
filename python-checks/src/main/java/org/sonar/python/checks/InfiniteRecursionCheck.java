/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.api.PythonKeyword;

@Rule(key = "S2190")
public class InfiniteRecursionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a way to break out of this %s's recursion.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      List<Tree> unconditionalCalls = unconditionalRecursiveCalls(functionDef);
      if (!unconditionalCalls.isEmpty()) {
        String message = String.format(MESSAGE, functionDef.isMethodDefinition() ? "method" : "function");
        PreciseIssue issue = ctx.addIssue(functionDef.name(), message);
        unconditionalCalls.forEach(call -> issue.secondary(call, "recursive call"));
      }
    });
  }

  private static List<Tree> unconditionalRecursiveCalls(FunctionDef functionDef) {
    List<String> lookupNames = new ArrayList<>();
    if (functionDef.isMethodDefinition()) {
      lookupNames.add("self." + functionDef.name().name());
      String className = parentClassName(functionDef);
      if (className != null) {
        lookupNames.add(className + "." + functionDef.name().name());
      }
    } else {
      lookupNames.add(functionDef.name().name());
    }
    UnconditionalCallVisitor visitor = new UnconditionalCallVisitor(lookupNames);
    functionDef.body().accept(visitor);
    return visitor.unconditionalCalls;
  }

  @Nullable
  private static String parentClassName(FunctionDef functionDef) {
    if (functionDef.parent().parent().is(Tree.Kind.CLASSDEF)) {
      return ((ClassDef) functionDef.parent().parent()).name().name();
    }
    return null;
  }

  private static class UnconditionalCallVisitor extends BaseTreeVisitor {

    private final List<String> lookupNames;
    private List<Tree> unconditionalCalls = new ArrayList<>();
    private int conditional = 0;
    private int exit = 0;

    private UnconditionalCallVisitor(List<String> lookupNames) {
      this.lookupNames = lookupNames;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (conditional == 0 && exit == 0) {
        Expression callee = callExpression.callee();
        String name = nameOf(callee);
        if (lookupNames.contains(name)) {
          unconditionalCalls.add(callee);
        }
      }
      super.visitCallExpression(callExpression);
    }

    @Nullable
    private static String nameOf(Expression expression) {
      if (expression.is(Tree.Kind.NAME)) {
        return ((Name) expression).name();
      } else if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression qualifiedExpression = (QualifiedExpression) expression;
        String qualifier = nameOf(qualifiedExpression.qualifier());
        if (qualifier != null) {
          return qualifier + "." + qualifiedExpression.name().name();
        }
      }
      return null;
    }

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      super.visitReturnStatement(pyReturnStatementTree);
      exit++;
    }

    @Override
    public void visitRaiseStatement(RaiseStatement pyRaiseStatementTree) {
      super.visitRaiseStatement(pyRaiseStatementTree);
      exit++;
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      if (lookupNames.contains(pyFunctionDefTree.name().name())) {
        exit++;
      }
      // ignore
    }

    @Override
    public void visitImportFrom(ImportFrom pyImportFromTree) {
      for (AliasedName aliasedName : pyImportFromTree.importedNames()) {
        Name alias = aliasedName.alias();
        if (alias != null) {
          if (lookupNames.contains(alias.name())) {
            exit++;
          }
        } else if (aliasedName.dottedName().names().size() == 1) {
          Name name = aliasedName.dottedName().names().get(0);
          if (lookupNames.contains(name.name())) {
            exit++;
          }
        }
      }
      super.visitImportFrom(pyImportFromTree);
    }

    @Override
    public void visitTryStatement(TryStatement pyTryStatementTree) {
      scan(pyTryStatementTree.body());
      conditional++;
      scan(pyTryStatementTree.exceptClauses());
      conditional--;
      scan(pyTryStatementTree.finallyClause());
      conditional++;
      scan(pyTryStatementTree.elseClause());
      conditional--;
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      // ignore
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      // ignore
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignment) {
      assignment.lhsExpressions().stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream)
        .map(UnconditionalCallVisitor::nameOf)
        .filter(lookupNames::contains)
        .findAny().ifPresent(name -> exit++);
      super.visitAssignmentStatement(assignment);
    }

    @Override
    public void visitIfStatement(IfStatement pyIfStatementTree) {
      scan(pyIfStatementTree.condition());
      conditional++;
      scan(pyIfStatementTree.body());
      scan(pyIfStatementTree.elifBranches());
      scan(pyIfStatementTree.elseBranch());
      conditional--;
    }

    @Override
    public void visitComprehensionIf(ComprehensionIf tree) {
      scan(tree.condition());
      conditional++;
      scan(tree.nestedClause());
      conditional--;
    }

    @Override
    public void visitForStatement(ForStatement pyForStatementTree) {
      scan(pyForStatementTree.expressions()); // TODO variable
      scan(pyForStatementTree.testExpressions());
      conditional++;
      scan(pyForStatementTree.body());
      scan(pyForStatementTree.elseClause());
      conditional--;
    }

    @Override
    public void visitComprehensionFor(ComprehensionFor tree) {
      // TODO variable
      super.visitComprehensionFor(tree);
    }

    @Override
    public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
      scan(tree.comprehensionFor());
      conditional++;
      scan(tree.resultExpression());
      conditional--;
    }

    @Override
    public void visitWhileStatement(WhileStatement pyWhileStatementTree) {
      scan(pyWhileStatementTree.condition());
      conditional++;
      scan(pyWhileStatementTree.body());
      scan(pyWhileStatementTree.elseClause());
      conditional--;
    }

    @Override
    public void visitConditionalExpression(ConditionalExpression pyConditionalExpressionTree) {
      scan(pyConditionalExpressionTree.condition());
      conditional++;
      scan(pyConditionalExpressionTree.trueExpression());
      scan(pyConditionalExpressionTree.falseExpression());
      conditional--;
    }

    @Override
    public void visitBinaryExpression(BinaryExpression pyBinaryExpressionTree) {
      String operator = pyBinaryExpressionTree.operator().value();
      if (PythonKeyword.OR.getValue().equals(operator) || PythonKeyword.AND.getValue().equals(operator)) {
        scan(pyBinaryExpressionTree.leftOperand());
        conditional++;
        scan(pyBinaryExpressionTree.rightOperand());
        conditional--;
      } else {
        super.visitBinaryExpression(pyBinaryExpressionTree);
      }
    }
  }

}
