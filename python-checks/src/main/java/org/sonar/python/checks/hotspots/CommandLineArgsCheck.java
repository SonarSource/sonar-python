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
package org.sonar.python.checks.hotspots;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = CommandLineArgsCheck.CHECK_KEY)
public class CommandLineArgsCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4823";
  private static final String MESSAGE = "Make sure that command line arguments are used safely here.";
  private static final Set<String> questionableFunctions = immutableSet("argparse.ArgumentParser", "optparse.OptionParser");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, CommandLineArgsCheck::checkSysArgNode);
    super.initialize(context);
  }

  private static void checkSysArgNode(SubscriptionContext ctx) {
    Name nameTree = (Name) ctx.syntaxNode();
    Tree parent = nameTree.parent();
    Symbol symbol = nameTree.symbol();
    if (symbol != null && "sys.argv".equals(symbol.fullyQualifiedName())) {
      if (isWithinImport(parent)) {
        return;
      }
      ctx.addIssue(nameTree, MESSAGE);
    }
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
