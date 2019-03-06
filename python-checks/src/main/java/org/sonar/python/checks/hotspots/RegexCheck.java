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
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.semantic.Symbol;

@Rule(key = RegexCheck.CHECK_KEY)
public class RegexCheck extends PythonCheck {
  public static final String CHECK_KEY = "S4784";
  private static final String MESSAGE = "Make sure that using a regular expression is safe here.";
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
        addIssue(parent, MESSAGE);
      } else {
        addIssue(node, MESSAGE);
      }
    }
  }
}
