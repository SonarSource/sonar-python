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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.CheckUtils.IS_ENUM_MATCHER;

@Rule(key = "S9156")
public class CachedInstanceMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE =
    "Remove \"@lru_cache\"/\"@cache\" from this instance method; the cache retains \"self\" and can leak memory.";

  private static final TypeMatcher CACHE_DECORATOR_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("functools.lru_cache"),
    TypeMatchers.isType("functools.cache"));

  // Methods implicitly converted into class methods by Python, hence they never retain "self"
  private static final Set<String> IMPLICIT_CLASS_METHODS = Set.of("__init_subclass__", "__class_getitem__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, CachedInstanceMethodCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (functionDef.decorators().isEmpty()) {
      return;
    }
    if (!(functionDef.name().typeV2() instanceof FunctionType functionType) || !functionType.isInstanceMethod()) {
      return;
    }
    if (IMPLICIT_CLASS_METHODS.contains(functionDef.name().name())) {
      return;
    }
    if (isEnumMethod(functionDef, ctx)) {
      return;
    }

    for (Decorator decorator : functionDef.decorators()) {
      if (CACHE_DECORATOR_MATCHER.isTrueFor(decoratorFunctionExpression(decorator), ctx)) {
        ctx.addIssue(decorator, MESSAGE);
      }
    }
  }

  private static boolean isEnumMethod(FunctionDef functionDef, SubscriptionContext ctx) {
    return TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF) instanceof ClassDef classDef
            && IS_ENUM_MATCHER.isTrueFor(classDef.name(), ctx);
  }

  private static Expression decoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }
}
