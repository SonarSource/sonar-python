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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2053")
public class PredictableSaltCheck extends PythonSubscriptionCheck {

  private static final String MISSING_SALT_MESSAGE = "Add an unpredictable salt value to this hash.";
  private static final String PREDICTABLE_SALT_MESSAGE = "Make this salt unpredictable.";
  private Map<String, Integer> sensitiveArgumentByFQN;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> handleCallExpression((CallExpression) ctx.syntaxNode(), ctx));
  }

  private void handleCallExpression(CallExpression callExpression, SubscriptionContext ctx) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    if (sensitiveArgumentByFQN().containsKey(calleeSymbol.fullyQualifiedName())) {
      int argNb = sensitiveArgumentByFQN().get(calleeSymbol.fullyQualifiedName());
      checkArguments(callExpression, argNb, ctx);
    }
  }

  private static void checkArguments(CallExpression callExpression, int argNb, SubscriptionContext ctx) {
    if (callExpression.arguments().size() <= argNb) {
      ctx.addIssue(callExpression.callee(), MISSING_SALT_MESSAGE);
    }
    for (int i = 0; i < callExpression.arguments().size(); i++) {
      Argument argument = callExpression.arguments().get(i);
      if (argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
        RegularArgument regularArgument = (RegularArgument) argument;
        Name keywordArgument = regularArgument.keywordArgument();
        if (keywordArgument != null) {
          if (keywordArgument.name().equals("salt")) {
            checkSensitiveArgument(regularArgument, ctx);
          }
        } else if (i == argNb) {
          checkSensitiveArgument(regularArgument, ctx);
        }
      }
    }
  }

  private static void checkSensitiveArgument(RegularArgument regularArgument, SubscriptionContext ctx) {
    if (regularArgument.expression().is(Tree.Kind.NAME)) {
      Expression expression = Expressions.singleAssignedValue((Name) regularArgument.expression());
      if (expression == null) {
        return;
      }
      if (expression.is(Tree.Kind.STRING_LITERAL)) {
        AssignmentStatement assignmentStatement = (AssignmentStatement) TreeUtils.firstAncestorOfKind(expression, Tree.Kind.ASSIGNMENT_STMT);
        ctx.addIssue(regularArgument, PREDICTABLE_SALT_MESSAGE).secondary(assignmentStatement, null);
      }
    }
    if (regularArgument.expression().is(Tree.Kind.STRING_LITERAL)) {
      ctx.addIssue(regularArgument, PREDICTABLE_SALT_MESSAGE);
    }
  }

  private Map<String, Integer> sensitiveArgumentByFQN() {
    if (sensitiveArgumentByFQN == null) {
      sensitiveArgumentByFQN = new HashMap<>();
      sensitiveArgumentByFQN.put("hashlib.pbkdf2_hmac", 2);
      sensitiveArgumentByFQN.put("hashlib.scrypt", 4);
      sensitiveArgumentByFQN.put("crypt.crypt", 1);
      sensitiveArgumentByFQN.put("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", 2);
      sensitiveArgumentByFQN.put("Cryptodome.Protocol.KDF.PBKDF2", 1);
      sensitiveArgumentByFQN.put("Cryptodome.Protocol.KDF.scrypt", 1);
      sensitiveArgumentByFQN.put("Cryptodome.Protocol.KDF.bcrypt", 2);
      sensitiveArgumentByFQN = Collections.unmodifiableMap(sensitiveArgumentByFQN);
    }
    return sensitiveArgumentByFQN;
  }
}
