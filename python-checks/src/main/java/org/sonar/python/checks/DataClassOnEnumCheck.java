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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

import static org.sonar.python.checks.utils.CheckUtils.IS_ENUM_MATCHER;

@Rule(key = "S8490")
public class DataClassOnEnumCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this \"@dataclass\" decorator; it is incompatible with Enum classes.";

  private static final TypeMatcher IS_DATACLASS_MATCHER = TypeMatchers.isType("dataclasses.dataclass");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, DataClassOnEnumCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (classDef.decorators().isEmpty()) {
      return;
    }

    if (!IS_ENUM_MATCHER.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    for (Decorator decorator : classDef.decorators()) {
      Expression decoratorExpr = getDecoratorFunctionExpression(decorator);
      if (IS_DATACLASS_MATCHER.isTrueFor(decoratorExpr, ctx)) {
        ctx.addIssue(decorator, MESSAGE);
      }
    }
  }

  private static Expression getDecoratorFunctionExpression(Decorator decorator) {
    Expression expr = decorator.expression();
    if (expr instanceof CallExpression callExpr) {
      return callExpr.callee();
    }
    return expr;
  }
}
