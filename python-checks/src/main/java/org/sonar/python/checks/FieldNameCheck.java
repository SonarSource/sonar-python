/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = FieldNameCheck.CHECK_KEY)
public class FieldNameCheck extends PythonCheck {

  public static final String CHECK_KEY = "S116";

  private static final String MESSAGE = "Rename this field \"%s\" to match the regular expression %s.";

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][_a-z0-9]*$";
  @RuleProperty(key = "format", defaultValue = DEFAULT)
  public String format = DEFAULT;

  private Pattern pattern = null;
  private Pattern constantPattern = null;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (!CheckUtils.classHasInheritance(astNode)) {
      List<Token> allFields = new NewSymbolsAnalyzer().getClassFields(astNode);
      checkNames(allFields);
    }
  }

  private void checkNames(List<Token> varNames) {
    if (constantPattern == null) {
      constantPattern = Pattern.compile(CONSTANT_PATTERN);
    }
    for (Token name : varNames) {
      if (!constantPattern.matcher(name.getValue()).matches()) {
        checkName(name);
      }
    }
  }

  private void checkName(Token token) {
    String name = token.getValue();
    if (pattern == null) {
      pattern = Pattern.compile(format);
    }
    if (!pattern.matcher(name).matches()) {
      addIssue(token, String.format(MESSAGE, name, format));
    }
  }

}
