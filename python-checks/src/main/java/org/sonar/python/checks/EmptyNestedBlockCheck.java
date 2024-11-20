/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.api.PythonTokenType;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = EmptyNestedBlockCheck.CHECK_KEY)
public class EmptyNestedBlockCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S108";
  public static final String QUICK_FIX_MESSAGE = "Add a TODO comment";
  private static final String MESSAGE = "Either remove or fill this block of code.";
  private static final String TODO_COMMENT_TEXT = "# TODO: Add implementation";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.STATEMENT_LIST, ctx -> {
      StatementList statementListTree = (StatementList) ctx.syntaxNode();
      if (statementListTree.statements().stream().anyMatch(stmt -> !stmt.is(Kind.PASS_STMT))) {
        return;
      }
      Tree parent = statementListTree.parent();
      if (parent.is(Kind.FUNCDEF) || parent.is(Kind.CLASSDEF) || parent.is(Kind.EXCEPT_CLAUSE)) {
        return;
      }
      List<Token> parentTokens = TreeUtils.tokens(statementListTree.parent());
      int from = parentTokens.stream().filter(t -> t.type() == PythonTokenType.NEWLINE).findFirst()
        .map(parentTokens::indexOf)
        .orElseThrow(() -> new IllegalStateException(String.format("No newline token in parent of statement list at line %s", statementListTree.firstToken().line())));
      // sublist call is excluding last index and token following last token of statement list (dedent) should be included in the comment verification.
      int to = parentTokens.indexOf(statementListTree.lastToken()) + 2;
      if (!containsComment(parentTokens.subList(from, to))) {
        var passTreeElement = Optional.of(statementListTree)
          .map(StatementList::statements)
          .map(Collection::stream)
          .flatMap(Stream::findFirst)
          .map(Tree.class::cast)
          .orElseGet(statementListTree::firstToken);

        var issue = ctx.addIssue(passTreeElement, MESSAGE);

        if (passTreeElement.firstToken().line() != parent.firstToken().line()) {
          var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE,
            TextEditUtils.insertLineBefore(passTreeElement, TODO_COMMENT_TEXT));
          issue.addQuickFix(quickFix);
        } else {
          var indent = TreeUtils.findIndentationSize(passTreeElement);
          if (indent > 0) {
            var offset = parent.firstToken().column() + indent;
            var textToInsert = "\n" + " ".repeat(offset) + TODO_COMMENT_TEXT + "\n" + " ".repeat(offset);
            var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE,
              TextEditUtils.insertBefore(passTreeElement, textToInsert));
            issue.addQuickFix(quickFix);
          }
        }

      }
    });
  }

  private static boolean containsComment(List<Token> tokens) {
    return tokens.stream().anyMatch(t -> !t.trivia().isEmpty());
  }
}
