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

import com.google.common.collect.ImmutableList;
import com.sonar.sslr.api.AstNode;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;

import java.util.List;

@Rule(
  key = "S1764",
  name = "Identical expressions should not be used on both sides of a binary operator",
  priority = Priority.MAJOR,
  tags = {Tags.BUG, Tags.CERT})
@SqaleConstantRemediation("2min")
@ActivatedByDefault
public class IdenticalExpressionOnBinaryOperatorCheck extends PythonCheck {

  private static final List<String> EXCLUDED_OPERATOR_TYPES = ImmutableList.of(
    "*",
    "+"
  );

  @Override
  public void init() {
    subscribeTo(
      PythonGrammar.M_EXPR,
      PythonGrammar.A_EXPR,
      PythonGrammar.SHIFT_EXPR,
      PythonGrammar.AND_EXPR,
      PythonGrammar.XOR_EXPR,
      PythonGrammar.OR_EXPR,
      PythonGrammar.COMPARISON,
      PythonGrammar.OR_TEST,
      PythonGrammar.AND_TEST
    );
  }

  @Override
  public void visitNode(AstNode expression) {
    List<AstNode> children = expression.getChildren();
    AstNode leftOperand = children.get(0);
    String operator = children.get(1).getTokenValue();
    AstNode rightOperand = children.get(2);
    if (!EXCLUDED_OPERATOR_TYPES.contains(operator) && CheckUtils.equalNodes(leftOperand, rightOperand) && !isLeftShiftBy1(leftOperand, operator)) {
      addIssue(rightOperand, "Correct one of the identical sub-expressions on both sides of operator \"" + operator + "\".")
        .secondary(leftOperand, "");
    }
  }

  private static boolean isLeftShiftBy1(AstNode leftOperand, String operator) {
    return "<<".equals(operator) && "1".equals(leftOperand.getTokenValue());
  }
}
