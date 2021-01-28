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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

/**
 * Even if decorator is not a statement, it extends {@code SimpleStatement} in order to properly resolve
 * its last token as it's a newline token
 */
public class DecoratorImpl extends SimpleStatement implements Decorator {
  private final Token atToken;
  private final Token newLineToken;
  private final Expression expression;
  private final DottedName name;

  public DecoratorImpl(Token atToken, Expression expression, @Nullable Token newLineToken) {
    this.atToken = atToken;
    this.expression = expression;
    this.name = new DottedNameImpl(nameTreesFromExpression(expression));
    this.newLineToken = newLineToken != null ? newLineToken : null;
  }

  @Override
  public Token atToken() {
    return atToken;
  }

  @Override
  @Deprecated
  public DottedName name() {
    return name;
  }

  @CheckForNull
  @Override
  @Deprecated
  public Token leftPar() {
    if (expression.is(Kind.CALL_EXPR)) {
      return ((CallExpression) expression).leftPar();
    }
    return null;
  }

  @CheckForNull
  @Override
  public ArgList arguments() {
    if (expression.is(Kind.CALL_EXPR)) {
      return ((CallExpression) expression).argumentList();
    }
    return null;
  }

  @CheckForNull
  @Override
  @Deprecated
  public Token rightPar() {
    if (expression.is(Kind.CALL_EXPR)) {
      return ((CallExpression) expression).rightPar();
    }
    return null;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDecorator(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(atToken, expression, newLineToken).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.DECORATOR;
  }

  private static List<Name> nameTreesFromExpression(Expression expression) {
    if (expression.is(Kind.NAME)) {
      List<Name> result = new ArrayList<>();
      result.add((Name) expression);
      return result;
    } else if (expression.is(Kind.QUALIFIED_EXPR)) {
      return nameTreesFromQualifiedExpression((QualifiedExpression) expression);
    } else if (expression.is(Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Expression callee = callExpression.callee();
      return nameTreesFromExpression(callee);
    }
    return Collections.emptyList();
  }

  private static List<Name> nameTreesFromQualifiedExpression(QualifiedExpression qualifiedExpression) {
    Name exprName = qualifiedExpression.name();
    Expression qualifier = qualifiedExpression.qualifier();
    List<Name> names = nameTreesFromExpression(qualifier);
    if (!names.isEmpty()) {
      names.add(exprName);
    }
    return names;
  }
}
