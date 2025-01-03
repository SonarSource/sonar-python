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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;

import static org.sonar.plugins.python.api.tree.Tree.Kind.KEY_VALUE_PAIR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.SET_LITERAL;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

@Rule(key = "S6662")
public class HashMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure this expression is hashable.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(SET_LITERAL, ctx -> ((SetLiteral) ctx.syntaxNode()).elements().forEach(elem -> checkHashMethod(elem, ctx)));
    context.registerSyntaxNodeConsumer(KEY_VALUE_PAIR, ctx -> checkHashMethod(((KeyValuePair) ctx.syntaxNode()).key(), ctx));
  }

  /**
   Note that, by default "object" itself defines a __hash__ method, and it is removed from classes whenever one of those two things happen (see Python
   <a href="https://docs.python.org/3/reference/datamodel.html#object.__hash__">documentation</a>):
   <ol>
   <li>A class implements __eq__ without implementing __hash__</li>
   <li>__hash__ is manually set to None (common in mutable sequences like list)</li>
   </ol>

   The implementation will only focus on 2., because 1. is causing too many FPs for typeshed symbols (e.g. int)
   */
  private static void checkHashMethod(Expression expression, SubscriptionContext ctx) {
    InferredType type = expression.type();
    boolean hashIsManuallySetToNone = type.resolveMember("__hash__")
      .filter(symbol -> ((SymbolImpl) symbol).inferredType().canOnlyBe(NONE_TYPE))
      .isPresent();
    if (hashIsManuallySetToNone) {
      ctx.addIssue(expression, MESSAGE);
    }
  }
}
