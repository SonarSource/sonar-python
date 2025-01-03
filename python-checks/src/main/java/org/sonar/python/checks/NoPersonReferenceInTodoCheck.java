/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
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

