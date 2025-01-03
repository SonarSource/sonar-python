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

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = "S5445")
public class TempFileCreationCheck extends PythonSubscriptionCheck {

  private static final List<String> SUSPICIOUS_CALLS = Arrays.asList("os.tempnam", "os.tmpnam", "tempfile.mktemp");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpr = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpr.calleeSymbol();
      isInsecureTempFile(symbol).ifPresent(s -> ctx.addIssue(callExpr, String.format("'%s' is insecure. Use 'tempfile.TemporaryFile' instead", s)));
    });
  }

  private static Optional<String> isInsecureTempFile(@Nullable Symbol symbol) {
    if (symbol == null) {
      return Optional.empty();
    }
    return SUSPICIOUS_CALLS.stream().filter(call -> call.equals(symbol.fullyQualifiedName())).findFirst();
  }
}
