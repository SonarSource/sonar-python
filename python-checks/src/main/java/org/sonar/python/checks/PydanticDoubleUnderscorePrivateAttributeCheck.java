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
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8973")
public class PydanticDoubleUnderscorePrivateAttributeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this double underscore prefix with a single underscore to avoid Python name mangling.";

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");
  private static final TypeMatcher IS_PYDANTIC_PRIVATE_ATTR = TypeMatchers.isType("pydantic.PrivateAttr");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, PydanticDoubleUnderscorePrivateAttributeCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!IS_PYDANTIC_MODEL.isTrueFor(classDef.name(), ctx)) {
      return;
    }

    classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AnnotatedAssignment.class))
      .forEach(annotatedAssignment -> checkAttribute(ctx, annotatedAssignment.variable()));

    classDef.body().statements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(AssignmentStatement.class))
      .filter(stmt -> isPrivateAttrCall(ctx, stmt.assignedValue()))
      .flatMap(stmt -> stmt.lhsExpressions().stream())
      .flatMap(exprList -> exprList.expressions().stream())
      .forEach(variable -> checkAttribute(ctx, variable));
  }

  private static boolean isPrivateAttrCall(SubscriptionContext ctx, Expression assignedValue) {
    if (!(assignedValue instanceof CallExpression callExpression)) {
      return false;
    }
    return IS_PYDANTIC_PRIVATE_ATTR.isTrueFor(callExpression.callee(), ctx);
  }

  private static void checkAttribute(SubscriptionContext ctx, Expression variable) {
    if (!(variable instanceof Name name)) {
      return;
    }

    String attributeName = name.name();
    if (hasDoubleUnderscorePrefix(attributeName)) {
      ctx.addIssue(name, MESSAGE);
    }
  }

  /**
   * Returns true if the name has a double leading underscore prefix but is NOT a dunder method/attribute
   * (i.e., does not also end with double underscore like {@code __init__}).
   */
  private static boolean hasDoubleUnderscorePrefix(String name) {
    return name.startsWith("__") && !name.endsWith("__");
  }

}
