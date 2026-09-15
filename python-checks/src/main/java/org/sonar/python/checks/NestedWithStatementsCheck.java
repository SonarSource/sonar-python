/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9154")
public class NestedWithStatementsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Combine these nested \"with\" statements into a single \"with\" with multiple contexts.";
  private static final String SECONDARY_MESSAGE = "Nested \"with\" statement";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, NestedWithStatementsCheck::checkWithStatement);
  }

  private static void checkWithStatement(SubscriptionContext ctx) {
    WithStatement withStatement = (WithStatement) ctx.syntaxNode();
    if (hasCombinableSameAsyncParent(withStatement)) {
      return;
    }
    List<WithStatement> nestedChain = collectSoleNestedSameAsyncChain(withStatement);
    if (nestedChain.isEmpty()) {
      return;
    }

    PreciseIssue issue = ctx.addIssue(withStatement.withKeyword(), MESSAGE);
    nestedChain.forEach(nested -> issue.secondary(nested.withKeyword(), SECONDARY_MESSAGE));
  }

  private static boolean hasCombinableSameAsyncParent(WithStatement inner) {
    return inner.parent() instanceof StatementList statementList
      && statementList.statements().size() == 1
      && statementList.parent() instanceof WithStatement outer
      && isCombinableNesting(outer, inner);
  }

  private static List<WithStatement> collectSoleNestedSameAsyncChain(WithStatement outermost) {
    List<WithStatement> chain = new ArrayList<>();
    Optional<WithStatement> current = soleNestedSameAsyncWith(outermost);
    while (current.isPresent()) {
      chain.add(current.get());
      current = soleNestedSameAsyncWith(current.get());
    }
    return chain;
  }

  private static Optional<WithStatement> soleNestedSameAsyncWith(WithStatement outer) {
    List<Statement> statements = outer.statements().statements();
    if (statements.size() != 1) {
      return Optional.empty();
    }
    Statement only = statements.get(0);
    if (only instanceof WithStatement inner && isCombinableNesting(outer, inner)) {
      return Optional.of(inner);
    }
    return Optional.empty();
  }

  private static boolean isCombinableNesting(WithStatement outer, WithStatement inner) {
    return outer.isAsync() == inner.isAsync() && !hasCommentsBetween(outer, inner);
  }

  private static boolean hasCommentsBetween(WithStatement outer, WithStatement inner) {
    return TreeUtils.tokens(outer).stream()
            .flatMap(token -> token.trivia().stream())
            .map(trivia -> trivia.token().line())
            .anyMatch(line -> line > outer.colon().line() && line < inner.firstToken().line());
  }
}
