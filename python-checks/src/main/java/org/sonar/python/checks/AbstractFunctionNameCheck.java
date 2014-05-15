/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.regex.Pattern;

public abstract class AbstractFunctionNameCheck extends SquidCheck<Grammar> {

  private static final String DEFAULT = "^[a-z_][a-z0-9_]{2,30}$";

  @RuleProperty(
    key = "format",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;
  private Pattern pattern = null;

  @Override
  public void init() {
    pattern = Pattern.compile(format);
    subscribeTo(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (!shouldCheckFunctionDeclaration(astNode)) {
      return;
    }
    AstNode nameNode = astNode.getFirstChild(PythonGrammar.FUNCNAME);
    String name = nameNode.getTokenValue();
    if (!pattern.matcher(name).matches()) {
      getContext().createLineViolation(this,
        "Rename {0} \"{1}\" to match the regular expression {2}.", nameNode, typeName(), name, format);
    }
  }

  public abstract String typeName();

  public abstract boolean shouldCheckFunctionDeclaration(AstNode astNode);

}
