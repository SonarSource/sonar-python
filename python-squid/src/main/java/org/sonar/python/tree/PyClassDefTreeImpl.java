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
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyClassDefTreeImpl extends PyTree implements PyClassDefTree {

  private final PyNameTree name;
  private final PyArgListTree args;
  private final List<PyStatementTree> body;

  public PyClassDefTreeImpl(AstNode astNode, PyNameTree name, PyArgListTree args, List<PyStatementTree> body) {
    super(astNode);
    this.name = name;
    this.args = args;
    this.body = body;
  }

  @Override
  public Kind getKind() {
    return Kind.CLASSDEF;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitClassDef(this);
  }

  @Override
  public List<PyDecoratorTree> decorators() {
    return null;
  }

  @Override
  public Token classKeyword() {
    return null;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @CheckForNull
  @Override
  public Token leftPar() {
    return null;
  }

  @CheckForNull
  @Override
  public PyArgListTree args() {
    return args;
  }

  @CheckForNull
  @Override
  public Token rightPar() {
    return null;
  }

  @Override
  public Token colon() {
    return null;
  }

  @Override
  public List<PyStatementTree> body() {
    return body;
  }
}
