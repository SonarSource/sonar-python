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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.ArgList;
import org.sonar.python.api.tree.Decorator;
import org.sonar.python.api.tree.DottedName;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class DecoratorImpl extends PyTree implements Decorator {
  private final Token atToken;
  private final DottedName dottedName;
  private final Token lPar;
  private final ArgList argListTree;
  private final Token rPar;
  private final Token newLineToken;

  public DecoratorImpl(Token atToken, DottedName dottedName,
                       @Nullable Token lPar, @Nullable ArgList argListTree, @Nullable Token rPar, @Nullable Token newLineToken) {
    super(atToken, rPar == null ? dottedName.lastToken() : rPar);
    this.atToken = atToken;
    this.dottedName = dottedName;
    this.lPar = lPar != null ? lPar : null;
    this.argListTree = argListTree;
    this.rPar = rPar != null ? rPar : null;
    this.newLineToken = newLineToken != null ? newLineToken : null;
  }

  @Override
  public Token atToken() {
    return atToken;
  }

  @Override
  public DottedName name() {
    return dottedName;
  }

  @CheckForNull
  @Override
  public Token leftPar() {
    return lPar;
  }

  @CheckForNull
  @Override
  public ArgList arguments() {
    return argListTree;
  }

  @CheckForNull
  @Override
  public Token rightPar() {
    return rPar;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDecorator(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(atToken, dottedName, lPar, argListTree, rPar, newLineToken).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.DECORATOR;
  }
}
