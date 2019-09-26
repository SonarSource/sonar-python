/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.WithItem;
import org.sonar.python.api.tree.WithStatement;
import org.sonar.python.api.tree.Tree;

public class WithStatementImpl extends PyTree implements WithStatement {

  private final List<WithItem> withItems;
  private final StatementList statements;
  private final Token asyncKeyword;
  private final boolean isAsync;
  private final Token colon;

  public WithStatementImpl(AstNode node, List<WithItem> withItems, Token colon, StatementList statements, @Nullable Token asyncKeyword) {
    super(node);
    this.withItems = withItems;
    this.colon = colon;
    this.statements = statements;
    this.asyncKeyword = asyncKeyword;
    this.isAsync = asyncKeyword != null;
  }

  @Override
  public List<WithItem> withItems() {
    return withItems;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public StatementList statements() {
    return statements;
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
  public Kind getKind() {
    return Kind.WITH_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitWithStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(asyncKeyword), withItems, Arrays.asList(colon, statements))
      .flatMap(List::stream).collect(Collectors.toList());
  }

  public static class WithItemImpl extends PyTree implements WithItem {

    private final Expression test;
    private final Token as;
    private final Expression expr;

    public WithItemImpl(AstNode node, Expression test, @Nullable Token as, @Nullable Expression expr) {
      super(node);
      this.test = test;
      this.as = as;
      this.expr = expr;
    }

    @Override
    public Expression test() {
      return test;
    }

    @CheckForNull
    @Override
    public Token as() {
      return as;
    }

    @CheckForNull
    @Override
    public Expression expression() {
      return expr;
    }

    @Override
    public Kind getKind() {
      return Kind.WITH_ITEM;
    }

    @Override
    public void accept(TreeVisitor visitor) {
      visitor.visitWithItem(this);
    }

    @Override
    public List<Tree> children() {
      return Stream.of(test, as, expr).filter(Objects::nonNull).collect(Collectors.toList());
    }
  }
}
