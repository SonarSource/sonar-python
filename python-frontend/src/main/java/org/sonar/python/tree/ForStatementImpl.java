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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ForStatementImpl extends PyTree implements ForStatement {

  private final Token forKeyword;
  private final List<Expression> expressions;
  private final List<Token> expressionsCommas;
  private final Token inKeyword;
  private final List<Expression> testExpressions;
  private final List<Token> testExpressionsCommas;
  private final Token colon;
  private final Token firstNewLine;
  private final Token firstIndent;
  private final StatementList body;
  private final Token firstDedent;
  private final ElseClause elseClause;
  private final Token asyncKeyword;
  private final boolean isAsync;

  public ForStatementImpl(Token forKeyword, List<Expression> expressions, List<Token> expressionsCommas, Token inKeyword, List<Expression> testExpressions,
                          List<Token> testExpressionsCommas, Token colon, @Nullable Token firstNewLine, @Nullable Token firstIndent, StatementList body,
                          @Nullable Token firstDedent, @Nullable ElseClause elseClause, @Nullable Token asyncKeyword) {
    this.forKeyword = forKeyword;
    this.expressions = expressions;
    this.expressionsCommas = expressionsCommas;
    this.inKeyword = inKeyword;
    this.testExpressions = testExpressions;
    this.testExpressionsCommas = testExpressionsCommas;
    this.colon = colon;
    this.firstNewLine = firstNewLine;
    this.firstIndent = firstIndent;
    this.body = body;
    this.firstDedent = firstDedent;
    this.elseClause = elseClause;
    this.asyncKeyword = asyncKeyword;
    this.isAsync = asyncKeyword != null;
  }

  @Override
  public Kind getKind() {
    return Kind.FOR_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitForStatement(this);
  }

  @Override
  public Token forKeyword() {
    return forKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public Token inKeyword() {
    return inKeyword;
  }

  @Override
  public List<Expression> testExpressions() {
    return testExpressions;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public StatementList body() {
    return body;
  }

  @CheckForNull
  @Override
  public ElseClause elseClause() {
    return elseClause;
  }

  @Override
  public boolean isAsync() {
    return isAsync;
  }

  @CheckForNull
  @Override
  public Token asyncKeyword() {
    return asyncKeyword;
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> expressionsWithCommas = addCommas(expressions, expressionsCommas);
    List<Tree> testExpressionsWithCommas = addCommas(testExpressions, testExpressionsCommas);
    return Stream.of(Arrays.asList(asyncKeyword, forKeyword), expressionsWithCommas, Collections.singletonList(inKeyword), testExpressionsWithCommas,
      Arrays.asList(colon, firstNewLine, firstIndent, body, firstDedent, elseClause))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  private static List<Tree> addCommas(List<Expression> expressions, List<Token> commas) {
    List<Tree> expressionsWithSeparator = new ArrayList<>();
    int index = 0;
    for (Expression expression : expressions) {
      expressionsWithSeparator.add(expression);
      if (index < commas.size()) {
        expressionsWithSeparator.add(commas.get(index));
      }
      index++;
    }
    return expressionsWithSeparator;
  }
}
