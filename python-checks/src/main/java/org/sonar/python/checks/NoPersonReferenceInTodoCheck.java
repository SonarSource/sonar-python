/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;

@Rule(key = "S1707")
public class NoPersonReferenceInTodoCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Add a citation of the person who can best explain this comment.";

  private static final String DEFAULT_PERSON_REFERENCE_PATTERN = "[ ]*\\([ _a-zA-Z0-9@.]+\\)";
  private static final String COMMENT_PATTERN = "^#[ ]*(todo|fixme)";
  private Pattern patternTodoFixme;
  private Pattern patternPersonReference;

  @RuleProperty(
    key = "pattern",
    description = "A regular expression defining the pattern that should be present after \"TODO\" or \"FIXME\"",
    defaultValue = DEFAULT_PERSON_REFERENCE_PATTERN)
  public String personReferencePatternString = DEFAULT_PERSON_REFERENCE_PATTERN;

  @Override
  public void initialize(Context context) {
    patternTodoFixme = Pattern.compile(COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
    patternPersonReference = Pattern.compile(personReferencePatternString);
    context.registerSyntaxNodeConsumer(Tree.Kind.TOKEN, ctx -> {
      Token token = (Token) ctx.syntaxNode();
      for (Trivia trivia : token.trivia()) {
        checkComment(trivia, ctx);
      }
    });
  }

  private void checkComment(Trivia trivia, SubscriptionContext ctx) {
    String comment = trivia.value();
    Matcher matcher = patternTodoFixme.matcher(comment);
    if (matcher.find()) {
      String tail = comment.substring(matcher.end());
      if (!patternPersonReference.matcher(tail).find()) {
        ctx.addIssue(trivia.token(), MESSAGE);
      }
    }
  }
}

