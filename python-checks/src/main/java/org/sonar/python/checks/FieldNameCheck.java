/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.List;
import java.util.regex.Pattern;

@Rule(
    key = FieldNameCheck.CHECK_KEY,
    priority = Priority.MINOR,
    name = "Field names should comply with a naming convention",
    tags = Tags.CONVENTION
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.READABILITY)
@SqaleConstantRemediation("2min")
@ActivatedByDefault
public class FieldNameCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S116";

  private static final String MESSAGE = "Rename this field \"%s\" to match the regular expression %s.";

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][_a-z0-9]*$";
  @RuleProperty(key = "format", defaultValue = DEFAULT)
  public String format = DEFAULT;

  private Pattern pattern = null;
  private Pattern constantPattern = null;

  @Override
  public void init() {
    pattern = Pattern.compile(format);
    constantPattern = Pattern.compile(CONSTANT_PATTERN);
    subscribeTo(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (CheckUtils.classHasNoInheritance(astNode)) {
      List<Token> allFields = new NewSymbolsAnalyzer().getClassFields(astNode);
      checkNames(allFields);
    }
  }

  private void checkNames(List<Token> varNames) {
    for (Token name : varNames) {
      if (!constantPattern.matcher(name.getValue()).matches()) {
        checkName(name);
      }
    }
  }

  private void checkName(Token token) {
    String name = token.getValue();
    if (!pattern.matcher(name).matches()) {
      getContext().createLineViolation(this, String.format(MESSAGE, name, format), token.getLine());
    }
  }

}
