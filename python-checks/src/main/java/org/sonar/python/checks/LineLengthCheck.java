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

import com.sonar.sslr.api.AstAndTokenVisitor;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;


@Rule(
    key = LineLengthCheck.CHECK_KEY,
    priority = Priority.MINOR,
    name = "Lines should not be too long",
    tags = Tags.CONVENTION
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.READABILITY)
@SqaleConstantRemediation("1min")
public class LineLengthCheck extends SquidCheck<Grammar> implements AstAndTokenVisitor {

  public static final String CHECK_KEY = "LineLength";
  private static final int DEFAULT_MAXIMUM_LINE_LENGTH = 120;

  @RuleProperty(
    key = "maximumLineLength",
    defaultValue = "" + DEFAULT_MAXIMUM_LINE_LENGTH)
  public int maximumLineLength = DEFAULT_MAXIMUM_LINE_LENGTH;

  public int getMaximumLineLength() {
    return maximumLineLength;
  }

  private Token previousToken;

  @Override
  public void visitFile(AstNode astNode) {
    previousToken = null;
  }

  @Override
  public void leaveFile(AstNode astNode) {
    previousToken = null;
  }

  @Override
  public void visitToken(Token token) {
    if (!token.isGeneratedCode()) {
      if (previousToken != null && previousToken.getLine() != token.getLine()) {
        // Note that AbstractLineLengthCheck doesn't support tokens which span multiple lines - see SONARPLUGINS-2025
        String[] lines = previousToken.getValue().split("\r?\n|\r", -1);
        int length = previousToken.getColumn();
        for (int line = 0; line < lines.length; line++) {
          length += lines[line].length();
          if (length > getMaximumLineLength()) {
            // Note that method from AbstractLineLengthCheck generates other message - see SONARPLUGINS-1809
            getContext().createLineViolation(this,
              "The line contains {0,number,integer} characters which is greater than {1,number,integer} authorized.",
              previousToken.getLine(),
              length,
              getMaximumLineLength());
          }
          length = 0;
        }
      }
      previousToken = token;
    }
  }

}
