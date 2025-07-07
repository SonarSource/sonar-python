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

import javax.annotation.Nullable;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7503")
public class AsyncFunctionNotAsyncCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use asynchronous features in this function or remove the `async` keyword.";
  private TypeCheckBuilder notImplementedTypeChecker;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkAsyncFunction);
  }

  private void setupTypeChecks(SubscriptionContext ctx) {
    notImplementedTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithName("NotImplemented");
  }

  private void checkAsyncFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    Token asyncKeyword = functionDef.asyncKeyword();
    if (asyncKeyword == null || isException(functionDef)) {
      return;
    }
    AsyncFeatureVisitor visitor = new AsyncFeatureVisitor();
    functionDef.body().accept(visitor);

    if (!visitor.hasAsyncFeature()) {
      ctx.addIssue(functionDef.name(), MESSAGE).secondary(asyncKeyword, "This function is async.");
    }
  }

  private boolean isException(FunctionDef functionDef) {
    return CheckUtils.isAbstract(functionDef) ||
      isTrivialFunction(functionDef.body()) ||
      isDunderMethod(functionDef) ||
      !functionDef.decorators().isEmpty() ||
      mightBeOverridingMethod(functionDef);
  }

  private static boolean isDunderMethod(FunctionDef functionDef) {
    String methodName = functionDef.name().name();
    return methodName.startsWith("__");
  }

  private boolean isTrivialFunction(StatementList body) {
    for (Statement statement : body.statements()) {
      if (!CheckUtils.isEmptyStatement(statement) && !statement.is(Tree.Kind.RAISE_STMT) && !isReturnNotImplemented(statement)) {
        return false;
      }
    }
    return true;
  }

  private boolean isReturnNotImplemented(Statement statement) {
    return statement.is(Tree.Kind.RETURN_STMT) &&
      ((ReturnStatement) statement).expressions().stream().allMatch(e -> notImplementedTypeChecker.check(e.typeV2()) == TriBool.TRUE);
  }

  private static boolean mightBeOverridingMethod(FunctionDef functionDef) {
    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    return functionType.owner() instanceof ClassType classType && (classType.hasUnresolvedHierarchy() || classType.inheritedMember(functionType.name()).isPresent());
  }

  private static class AsyncFeatureVisitor extends BaseTreeVisitor {

    private boolean asyncFeatureFound = false;

    public boolean hasAsyncFeature() {
      return asyncFeatureFound;
    }

    @Override
    public void visitAwaitExpression(AwaitExpression awaitExpression) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitForStatement(ForStatement forStatement) {
      if (forStatement.isAsync()) {
        asyncFeatureFound = true;
        return;
      }
      if (!asyncFeatureFound) {
        super.visitForStatement(forStatement);
      }
    }

    @Override
    public void visitWithStatement(WithStatement withStatement) {
      if (withStatement.isAsync()) {
        asyncFeatureFound = true;
      }
      if (!asyncFeatureFound) {
        super.visitWithStatement(withStatement);
      }
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStatement) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpression) {
      asyncFeatureFound = true;
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Skip nested functions
    }

    @Override
    public void visitComprehensionFor(ComprehensionFor tree) {
      asyncFeatureFound |= tree.asyncToken() != null;
      super.visitComprehensionFor(tree);
    }

    @Override
    protected void scan(@Nullable Tree tree) {
      // Stop scanning if we've already found an async feature
      if (!asyncFeatureFound && tree != null) {
        tree.accept(this);
      }
    }
  }
}
