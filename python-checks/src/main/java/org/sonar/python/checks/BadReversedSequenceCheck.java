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
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9405")
public class BadReversedSequenceCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE =
    "Change the argument of this \"reversed()\" call to a reversible object " +
    "(one implementing \"__reversed__\", or both \"__len__\" and \"__getitem__\").";

  private static final TypeMatcher IS_REVERSED = TypeMatchers.isType("builtins.reversed");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BadReversedSequenceCheck::check);
  }

  private static void check(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!IS_REVERSED.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }
    TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
      .map(RegularArgument::expression)
      .filter(argument -> argument.is(Tree.Kind.GENERATOR_EXPR) || isNotReversible(argument.typeV2()))
      .ifPresent(argument -> ctx.addIssue(argument, MESSAGE));
  }

  private static boolean isNotReversible(PythonType type) {
    // For a union type, suppress the issue when any candidate is reversible: flow-insensitive
    // inference often widens a variable to also include a non-reversible member (e.g. "list | None")
    // even though the value actually reaching reversed() is reversible.
    Set<PythonType> candidates = type instanceof UnionType unionType ? unionType.candidates() : Set.of(type);
    return candidates.stream().noneMatch(BadReversedSequenceCheck::isReversible);
  }

  // A reversible object either implements __reversed__, or implements both __getitem__ and __len__ (the sequence protocol).
  // A member check that is not definitely FALSE (i.e. TRUE or UNKNOWN) is treated as reversible to stay conservative.
  private static boolean isReversible(PythonType type) {
    if (type.hasMember("__reversed__") != TriBool.FALSE) {
      return true;
    }
    return type.hasMember("__getitem__") != TriBool.FALSE && type.hasMember("__len__") != TriBool.FALSE;
  }
}
