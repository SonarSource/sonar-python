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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5828")
public class InvalidOpenModeCheck extends PythonSubscriptionCheck {

  private static final String VALID_MODES = "rwatb+Ux";
  private static final Pattern INVALID_CHARACTERS = Pattern.compile("[^" + VALID_MODES + "]");
  private static final String MESSAGE = "Fix this invalid mode string.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol == null || !"open".equals(calleeSymbol.fullyQualifiedName())) {
        return;
      }
      List<Argument> arguments = callExpression.arguments();
      RegularArgument modeArgument = TreeUtils.nthArgumentOrKeyword(1, "mode", arguments);
      if (modeArgument == null) {
        return;
      }
      Expression modeExpression = modeArgument.expression();
      if (modeExpression.is(Tree.Kind.STRING_LITERAL)) {
        checkOpenMode(ctx, modeExpression, (StringLiteral) modeExpression);
      } else if (modeExpression.is(Tree.Kind.NAME)) {
        Tree assignedValue = Expressions.singleAssignedValue((Name) modeExpression);
        if (assignedValue != null && assignedValue.is(Tree.Kind.STRING_LITERAL)) {
          checkOpenMode(ctx, modeExpression, (StringLiteral) assignedValue);
        }
      }
    });
  }

  private static void checkOpenMode(SubscriptionContext ctx, Expression openExpression, StringLiteral stringLiteral) {
    if (stringLiteral.stringElements().stream().anyMatch(StringElement::isInterpolated)) {
      return;
    }
    String mode = stringLiteral.trimmedQuotesValue();
    if (mode.length() > VALID_MODES.length() || INVALID_CHARACTERS.matcher(mode).matches()) {
      raiseIssue(openExpression, stringLiteral, ctx);
      return;
    }
    Set<Character> modeSet = new HashSet<>();
    for (int i = 0; i < mode.length(); i++) {
      char character = mode.charAt(i);
      if (!modeSet.add(character)) {
        // there are duplicates and an issue should be raised
        raiseIssue(openExpression, stringLiteral, ctx);
        return;
      }
    }
    if (isInvalidMode(modeSet)) {
      raiseIssue(openExpression, stringLiteral, ctx);
    }
  }

  private static boolean isInvalidMode(Set<Character> modeSet) {
    boolean creating = modeSet.contains('x');
    boolean universalNewlines = modeSet.contains('U');
    boolean reading = modeSet.contains('r') || universalNewlines;
    boolean writing = modeSet.contains('w');
    boolean appending = modeSet.contains('a');
    boolean text = modeSet.contains('t');
    boolean binary = modeSet.contains('b');
    boolean updating = modeSet.contains('+');
    if (universalNewlines && (writing || appending || creating || updating)) {
      return true;
    }
    if (text && binary) {
      return true;
    }
    // 1 and only one of these mode should be enabled
    return (reading ? 1 : 0) + (writing ? 1 : 0) + (appending ? 1 : 0) + (creating ? 1 : 0) != 1;
  }

  private static void raiseIssue(Expression expression, StringLiteral stringLiteral, SubscriptionContext ctx) {
    PreciseIssue issue = ctx.addIssue(expression, MESSAGE);
    if (expression != stringLiteral) {
      issue.secondary(stringLiteral, null);
    }
  }
}
