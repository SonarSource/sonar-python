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
package org.sonar.python.checks;

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.CheckUtils.classHasInheritance;
import static org.sonar.python.checks.utils.CheckUtils.getParentClassDef;

@Rule(key = "S2325")
public class MethodShouldBeStaticCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make this method static.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef funcDef = (FunctionDef) ctx.syntaxNode();
      if (funcDef.isMethodDefinition()
        && !classHasInheritance(getParentClassDef(funcDef))
        && !isBuiltInMethod(funcDef)
        && !isStatic(funcDef)
        && hasValuableCode(funcDef)
        && !mayRaiseNotImplementedError(funcDef)
        && !isUsingSelfArg(funcDef)
        && funcDef.decorators().isEmpty()
      ) {
        ctx.addIssue(funcDef.name(), MESSAGE);
      }
    });
  }

  private static boolean mayRaiseNotImplementedError(FunctionDef funcDef) {
    RaiseStatementVisitor visitor = new RaiseStatementVisitor();
    funcDef.accept(visitor);
    return visitor.hasNotImplementedError;

  }

  private static boolean hasValuableCode(FunctionDef funcDef) {
    List<Statement> statements = funcDef.body().statements();
    return !statements.stream().allMatch(st -> isStringLiteral(st) || st.is(Tree.Kind.PASS_STMT) || isEllipsis(st));
  }

  private static boolean isStringLiteral(Statement st) {
    return st.is(Tree.Kind.EXPRESSION_STMT) && ((ExpressionStatement) st).expressions().stream().allMatch(e -> e.is(Tree.Kind.STRING_LITERAL));
  }

  private static boolean isEllipsis(Statement st) {
    return st.is(Tree.Kind.EXPRESSION_STMT) && ((ExpressionStatement) st).expressions().stream().allMatch(expr -> expr.is(Tree.Kind.ELLIPSIS));
  }

  private static boolean isUsingSelfArg(FunctionDef funcDef) {
    ParameterList parameters = funcDef.parameters();
    if (parameters == null) {
      // if a method has no parameters then it can't be a instance method.
      return true;
    }
    List<AnyParameter> params = parameters.all();
    if (params.isEmpty()) {
      return false;
    }
    if (params.get(0).is(Tree.Kind.TUPLE_PARAMETER)) {
      return false;
    }
    Parameter first = (Parameter) params.get(0);
    Name paramName = first.name();
    if (paramName == null) {
      // star argument should not raise issue
      return true;
    }
    SelfVisitor visitor = new SelfVisitor(paramName.name());
    funcDef.body().accept(visitor);
    return visitor.isUsingSelfArg;
  }

  private static boolean isStatic(FunctionDef funcDef) {
    return funcDef.decorators().stream()
      .map(d -> TreeUtils.decoratorNameFromExpression(d.expression()))
      .anyMatch(n -> "staticmethod".equals(n) || "classmethod".equals(n));
  }

  private static boolean isBuiltInMethod(FunctionDef funcDef) {
    String name = funcDef.name().name();
    String doubleUnderscore = "__";
    return name.startsWith(doubleUnderscore) && name.endsWith(doubleUnderscore);
  }

  private static class RaiseStatementVisitor extends BaseTreeVisitor {
    private int withinRaise = 0;
    boolean hasNotImplementedError = false;

    @Override
    public void visitRaiseStatement(RaiseStatement pyRaiseStatementTree) {
      withinRaise++;
      scan(pyRaiseStatementTree.expressions());
      withinRaise--;
    }

    @Override
    public void visitName(Name pyNameTree) {
      if (withinRaise > 0) {
        hasNotImplementedError |= pyNameTree.name().equals("NotImplementedError");
      }
    }
  }

  private static class SelfVisitor extends BaseTreeVisitor {
    private final String selfName;
    boolean isUsingSelfArg = false;

    SelfVisitor(String selfName) {
      this.selfName = selfName;
    }

    @Override
    public void visitName(Name pyNameTree) {
      isUsingSelfArg |= selfName.equals(pyNameTree.name());
    }
  }
}
