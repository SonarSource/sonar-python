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
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyDecoratorTreeImpl extends PyTree implements PyDecoratorTree {
  private final PyToken atToken;
  private final PyDottedNameTree dottedName;
  private final PyToken lPar;
  private final PyArgListTree argListTree;
  private final PyToken rPar;

  public PyDecoratorTreeImpl(AstNode astNode, PyToken atToken, PyDottedNameTree dottedName, @Nullable PyToken lPar, @Nullable PyArgListTree argListTree, @Nullable PyToken rPar) {
    super(astNode);
    this.atToken = atToken;
    this.dottedName = dottedName;
    this.lPar = lPar != null ? lPar : null;
    this.argListTree = argListTree;
    this.rPar = rPar != null ? rPar : null;
  }

  @Override
  public PyToken atToken() {
    return atToken;
  }

  @Override
  public PyDottedNameTree name() {
    return dottedName;
  }

  @CheckForNull
  @Override
  public PyToken leftPar() {
    return lPar;
  }

  @CheckForNull
  @Override
  public PyArgListTree arguments() {
    return argListTree;
  }

  @CheckForNull
  @Override
  public PyToken rightPar() {
    return rPar;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitDecorator(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(atToken, dottedName, lPar, argListTree, rPar).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.DECORATOR;
  }
}
