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
import com.sonar.sslr.api.Trivia;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.regex.Pattern;

@Rule(
    key = FixmeCommentCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "\"FIXME\" tags should be handled"
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.INSTRUCTION_RELIABILITY)
@SqaleConstantRemediation("20min")
@ActivatedByDefault
public class FixmeCommentCheck extends SquidCheck<Grammar> implements AstAndTokenVisitor {
  public static final String CHECK_KEY = "S1134";

  private static final String FIXME_COMMENT_PATTERN = "^#[ ]*fixme.*";

  private Pattern pattern;

  @Override
  public void visitFile(AstNode astNode) {
    pattern = Pattern.compile(FIXME_COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
  }

  @Override
  public void visitToken(Token token) {
    for (Trivia trivia : token.getTrivia()) {
      if (trivia.isComment()) {
        String comment = trivia.getToken().getValue();
        if (pattern.matcher(comment).matches()) {
          getContext().createLineViolation(this, "Take the required action to fix the issue indicated by this \"FIXME\" comment.", trivia.getToken().getLine());
        }
      }
    }
  }
}

