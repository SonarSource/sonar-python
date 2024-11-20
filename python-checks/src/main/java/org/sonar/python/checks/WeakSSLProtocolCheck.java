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

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.checks.cdk.WeakSSLProtocolCheckPart;

@Rule(key = "S4423")
public class WeakSSLProtocolCheck extends PythonSubscriptionCheck {
  private static final List<String> WEAK_PROTOCOL_CONSTANTS = Arrays.asList(
    "ssl.PROTOCOL_SSLv2",
    "ssl.PROTOCOL_SSLv3",
    "ssl.PROTOCOL_SSLv23",
    "ssl.PROTOCOL_TLS",
    "ssl.PROTOCOL_TLSv1",
    "ssl.PROTOCOL_TLSv1_1",
    "OpenSSL.SSL.SSLv2_METHOD",
    "OpenSSL.SSL.SSLv3_METHOD",
    "OpenSSL.SSL.SSLv23_METHOD",
    "OpenSSL.SSL.TLSv1_METHOD",
    "OpenSSL.SSL.TLSv1_1_METHOD"
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ctx -> {
      Name pyNameTree = (Name) ctx.syntaxNode();
      if (isWeakProtocol(pyNameTree.symbol())) {
        ctx.addIssue(pyNameTree, "Change this code to use a stronger protocol.");
      }
    });

    new WeakSSLProtocolCheckPart().initialize(context);
  }

  private static boolean isWeakProtocol(@Nullable Symbol symbol) {
    return symbol != null && WEAK_PROTOCOL_CONSTANTS.contains(symbol.fullyQualifiedName());
  }
}
