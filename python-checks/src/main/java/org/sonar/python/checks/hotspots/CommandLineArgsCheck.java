/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks.hotspots;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.checks.AbstractCallExpressionCheck;

@Rule(key = CommandLineArgsCheck.CHECK_KEY)
public class CommandLineArgsCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4823";
  private static final String MESSAGE = "Make sure that command line arguments are used safely here.";
  private static final Set<String> questionableFunctions = immutableSet("argparse.ArgumentParser", "optparse.OptionParser");

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.CALL_EXPR, PythonGrammar.NAME);
  }

  @Override
  public void visitNode(AstNode node) {
    if (node.is(PythonGrammar.NAME)) {
      if (isSysArgvNode(node)) {
        // highlight `sys.argv`
        addIssue(node.getParent().getParent(), MESSAGE);
      }
    } else {
      super.visitNode(node);
    }
  }

  private boolean isSysArgvNode(AstNode node) {
    if (!node.getTokenValue().equals("argv")) {
      return false;
    }
    // `NAME` has always a parent
    AstNode parent = node.getParent();
    return parent.is(PythonGrammar.TRAILER) && parent.getParent().getTokenValue().equals("sys");
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
