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
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTypeAnnotationTree;
import org.sonar.python.api.tree.Tree;

public class PyFunctionDefTreeImpl extends PyTree implements PyFunctionDefTree {

  private final List<PyDecoratorTree> decorators;
  private final PyToken asyncKeyword;
  private final PyToken defKeyword;
  private final PyNameTree name;
  private final PyToken leftPar;
  private final PyParameterListTree parameters;
  private final PyToken rightPar;
  private final PyTypeAnnotationTree returnType;
  private final PyToken colon;
  private final PyStatementListTree body;
  private final boolean isMethodDefinition;
  private final PyToken docstring;

  public PyFunctionDefTreeImpl(AstNode astNode, List<PyDecoratorTree> decorators, @Nullable PyToken asyncKeyword, PyToken defKeyword, PyNameTree name,
                               PyToken leftPar, @Nullable PyParameterListTree parameters, PyToken rightPar, @Nullable PyTypeAnnotationTree returnType,
                               PyToken colon, PyStatementListTree body, boolean isMethodDefinition, @Nullable PyToken docstring) {
    super(astNode);
    this.decorators = decorators;
    this.asyncKeyword = asyncKeyword;
    this.defKeyword = defKeyword;
    this.name = name;
    this.leftPar = leftPar;
    this.parameters = parameters;
    this.rightPar = rightPar;
    this.returnType = returnType;
    this.colon = colon;
    this.body = body;
    this.isMethodDefinition = isMethodDefinition;
    this.docstring = docstring;
  }

  @Override
  public List<PyDecoratorTree> decorators() {
    return decorators;
  }

  @Override
  public PyToken defKeyword() {
    return defKeyword;
  }

  @CheckForNull
  @Override
  public PyToken asyncKeyword() {
    return asyncKeyword;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @Override
  public PyToken leftPar() {
    return leftPar;
  }

  @CheckForNull
  @Override
  public PyParameterListTree parameters() {
    return parameters;
  }

  @Override
  public PyToken rightPar() {
    return rightPar;
  }

  @CheckForNull
  @Override
  public PyTypeAnnotationTree returnTypeAnnotation() {
    return returnType;
  }

  @Override
  public PyToken colon() {
    return colon;
  }

  @Override
  public PyStatementListTree body() {
    return body;
  }

  @Override
  public boolean isMethodDefinition() {
    return isMethodDefinition;
  }

  @CheckForNull
  @Override
  public PyToken docstring() {
    return docstring;
  }

  @Override
  public Kind getKind() {
    return Kind.FUNCDEF;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitFunctionDef(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(decorators, Arrays.asList(asyncKeyword, defKeyword, name, leftPar, parameters, rightPar, returnType, colon, docstring, body))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
