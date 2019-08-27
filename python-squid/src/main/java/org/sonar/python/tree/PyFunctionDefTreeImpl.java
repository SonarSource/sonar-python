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
import com.sonar.sslr.api.Token;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTypedArgListTree;

public class PyFunctionDefTreeImpl extends PyTree implements PyFunctionDefTree {

  private final PyNameTree name;
  private final PyTypedArgListTree typedArgs;
  private final PyStatementListTree body;
  private final boolean isMethodDefinition;

  public PyFunctionDefTreeImpl(AstNode astNode, PyNameTree name, PyTypedArgListTree typedArgs, PyStatementListTree body, boolean isMethodDefinition) {
    super(astNode);
    this.name = name;
    this.typedArgs = typedArgs;
    this.body = body;
    this.isMethodDefinition = isMethodDefinition;
  }

  @Override
  public List<PyDecoratorTree> decorators() {
    return null;
  }

  @Override
  public Token defKeyword() {
    return null;
  }

  @CheckForNull
  @Override
  public Token asyncKeyword() {
    return null;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @Override
  public Token leftPar() {
    return null;
  }

  @Override
  public PyTypedArgListTree typedArgs() {
    return typedArgs;
  }

  @Override
  public Token rightPar() {
    return null;
  }

  @CheckForNull
  @Override
  public Token dash() {
    return null;
  }

  @CheckForNull
  @Override
  public Token gt() {
    return null;
  }

  @CheckForNull
  @Override
  public PyExpressionTree annotationReturn() {
    return null;
  }

  @Override
  public Token colon() {
    return null;
  }

  @Override
  public PyStatementListTree body() {
    return body;
  }

  @Override
  public boolean isMethodDefinition() {
    return isMethodDefinition;
  }

  @Override
  public Kind getKind() {
    return Kind.FUNCDEF;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitFunctionDef(this);
  }
}
