/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5659")
public class JwtVerificationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Don't use a JWT token without verifying its signature.";

  // https://github.com/davedoesdev/python-jwt
  // "From version 2.0.1 the namespace has changed from jwt to python_jwt, in order to avoid conflict with PyJWT"
  private static final Set<String> PROCESS_JWT_NAMES = new HashSet<>(Arrays.asList(
    "python_jwt.process_jwt",
    "jwt.process_jwt"));
  private static final Set<String> VERIFY_JWT_NAMES = new HashSet<>(Arrays.asList(
    "python_jwt.verify_jwt",
    "jwt.verify_jwt"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression call = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = call.calleeSymbol();
      if (calleeSymbol == null) {
        return;
      }
      if ("jwt.decode".equals(calleeSymbol.fullyQualifiedName())) {
        RegularArgument verifyArg = TreeUtils.argumentByKeyword("verify", call.arguments());
        if (verifyArg != null && Expressions.isFalsy(verifyArg.expression())) {
          ctx.addIssue(verifyArg, MESSAGE);
        }
      } else if (PROCESS_JWT_NAMES.contains(calleeSymbol.fullyQualifiedName())) {
        Tree scriptOrFunction = TreeUtils.firstAncestorOfKind(call, Kind.FILE_INPUT, Kind.FUNCDEF);
        if (!TreeUtils.hasDescendant(scriptOrFunction, JwtVerificationCheck::isCallToVerifyJwt)) {
          ctx.addIssue(call, MESSAGE);
        }
      }
    });
  }

  private static boolean isCallToVerifyJwt(Tree t) {
    if (t.is(Kind.CALL_EXPR)) {
      Symbol symbol = ((CallExpression) t).calleeSymbol();
      return symbol != null && VERIFY_JWT_NAMES.contains(symbol.fullyQualifiedName());
    }
    return false;
  }

}
