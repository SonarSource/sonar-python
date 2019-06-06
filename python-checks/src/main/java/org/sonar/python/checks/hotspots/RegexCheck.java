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
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.semantic.Symbol;

@Rule(key = RegexCheck.CHECK_KEY)
public class RegexCheck extends PythonCheck {
  public static final String CHECK_KEY = "S4784";
  private static final String MESSAGE = "Make sure that using a regular expression is safe here.";
  private static final int REGEX_ARGUMENT = 0;
  private static final Set<String> questionableFunctions = immutableSet(
    "django.core.validators.RegexValidator", "django.urls.re_path",
    "re.compile", "re.match", "re.search", "re.fullmatch", "re.split", "re.findall", "re.finditer", "re.sub", "re.subn",
    "regex.compile", "regex.match", "regex.search", "regex.fullmatch", "regex.split", "regex.findall", "regex.finditer", "regex.sub", "regex.subn",
    "regex.subf", "regex.subfn", "regex.splititer");

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.ATTRIBUTE_REF, PythonGrammar.ATOM);
  }

  @Override
  public void visitNode(AstNode node) {
    Symbol symbol = getContext().symbolTable().getSymbol(node);
    if (symbol != null && questionableFunctions.contains(symbol.qualifiedName())) {
      AstNode parent = node.getParent();
      if (parent != null && parent.is(PythonGrammar.CALL_EXPR)) {
        AstNode argListNode = parent.getFirstChild(PythonGrammar.ARGLIST);
        if (argListNode != null) {
          // ARGLIST is not null => we have at least one child
          checkRegexArgument(argListNode.getChildren().get(REGEX_ARGUMENT));
        }
      }
    }
  }

  private void checkRegexArgument(AstNode arg) {
    AstNode atom = arg.getFirstDescendant(PythonGrammar.ATOM);
    if (atom == null) {
      return;
    }
    String literal = atom.getTokenValue();
    Symbol argSymbol = getContext().symbolTable().getSymbol(atom);
    IssueLocation secondaryLocation = null;
    if (argSymbol != null && argSymbol.writeUsages().size() == 1) {
      AstNode expressionStatement = argSymbol.writeUsages().iterator().next().getFirstAncestor(PythonGrammar.EXPRESSION_STMT);
      if (isAssignment(expressionStatement)) {
        AstNode expression = expressionStatement.getChildren().get(2);
        literal = expression.getTokenValue();
        secondaryLocation = IssueLocation.preciseLocation(expression, "");
      }
    }
    if (isSuspiciousRegex(literal)) {
      PreciseIssue preciseIssue = addIssue(atom, MESSAGE);
      if (secondaryLocation != null) {
        preciseIssue.secondary(secondaryLocation);
      }
    }
  }

  private static boolean isAssignment(@CheckForNull AstNode expressionStatement) {
    return expressionStatement != null &&
      expressionStatement.getChildren().size() == 3 &&
      expressionStatement.getChildren().get(1).is(PythonPunctuator.ASSIGN);
  }

  /**
   * This rule flags any execution of a hardcoded regular expression which has at least 3 characters and at least
   * two instances of any of the following characters: "*+{" (Example: (a+)*)
   */
  private static boolean isSuspiciousRegex(String regexp) {
    if (regexp.length() > 2) {
      int nOfSuspiciousChars = regexp.length() - regexp.replaceAll("[*+{]", "").length();
      return nOfSuspiciousChars > 1;
    }
    return false;
  }
}
