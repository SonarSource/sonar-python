/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;

@Rule(key = EmptyNestedBlockCheck.CHECK_KEY)
public class EmptyNestedBlockCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S108";
  public static final String QUICK_FIX_MESSAGE = "Add an empty TODO comment.";
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

        var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(PythonTextEdit.insertLineBefore(passTreeElement, TODO_COMMENT_TEXT))
          .build();

        var issue = (IssueWithQuickFix) ctx.addIssue(passTreeElement, MESSAGE);
        issue.addQuickFix(quickFix);
      }
    });
  }

  private static boolean containsComment(List<Token> tokens) {
    return tokens.stream().anyMatch(t -> !t.trivia().isEmpty());
  }
}
