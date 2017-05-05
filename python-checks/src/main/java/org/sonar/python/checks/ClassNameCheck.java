/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;

@Rule(key = ClassNameCheck.CHECK_KEY)
@ActivatedByDefault
public class ClassNameCheck extends PythonCheck {

  public static final String CHECK_KEY = "S101";
  private static final String DEFAULT = "^[A-Z_][a-zA-Z0-9]+$";
  private static final String MESSAGE = "Rename class \"%s\" to match the regular expression %s.";

  @RuleProperty(
    key = "format",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;
  private Pattern pattern = null;

  private Pattern pattern() {
    if (pattern == null) {
      pattern = Pattern.compile(format);
    }
    return pattern;
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return ImmutableSet.of(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    AstNode classNameNode = astNode.getFirstChild(PythonGrammar.CLASSNAME);
    String className = classNameNode.getTokenValue();
    if (!pattern().matcher(className).matches()) {
      String message = String.format(MESSAGE, className, format);
      addIssue(classNameNode, message);
    }
  }

}
