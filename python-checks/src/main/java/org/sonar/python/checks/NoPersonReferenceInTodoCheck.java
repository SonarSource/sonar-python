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
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;

@Rule(key = NoPersonReferenceInTodoCheck.CHECK_KEY)
public class NoPersonReferenceInTodoCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1707";
  public static final String MESSAGE = "Add a citation of the person who can best explain this comment.";

  private static final String DEFAULT_PERSON_REFERENCE_PATTERN = "[ ]*\\([ _a-zA-Z0-9@.]+\\)";
  private static final String COMMENT_PATTERN = "^#[ ]*(todo|fixme)";
  private Pattern patternTodoFixme;
  private Pattern patternPersonReference;

  @RuleProperty(
      key = "pattern",
      defaultValue = DEFAULT_PERSON_REFERENCE_PATTERN)
  public String personReferencePatternString = DEFAULT_PERSON_REFERENCE_PATTERN;

  @Override
  public void visitFile(AstNode astNode) {
    patternTodoFixme = Pattern.compile(COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
    patternPersonReference = Pattern.compile(personReferencePatternString);
  }

  @Override
  public void visitToken(Token token) {
    for (Trivia trivia : token.getTrivia()) {
      if (trivia.isComment()) {
        visitComment(trivia);
      }
    }
  }

  private void visitComment(Trivia trivia) {
    String comment = trivia.getToken().getValue();
    Matcher matcher = patternTodoFixme.matcher(comment);
    if (matcher.find()) {
      String tail = comment.substring(matcher.end());
      if (!patternPersonReference.matcher(tail).find()) {
        addIssue(trivia.getToken(), MESSAGE);
      }
    }
  }
}

