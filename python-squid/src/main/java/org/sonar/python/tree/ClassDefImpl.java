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
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.ArgList;
import org.sonar.python.api.tree.ClassDef;
import org.sonar.python.api.tree.Decorator;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;

public class ClassDefImpl extends PyTree implements ClassDef {
  private final List<Decorator> decorators;
  private final Token classKeyword;
  private final Name name;
  private final Token leftPar;
  private final ArgList args;
  private final Token rightPar;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final Token dedent;
  private final StatementList body;
  private final Token docstring;
  private final Set<Symbol> classFields = new HashSet<>();
  private final Set<Symbol> instanceFields = new HashSet<>();

  public ClassDefImpl(AstNode astNode, List<Decorator> decorators, Token classKeyword, Name name,
                            @Nullable Token leftPar, @Nullable ArgList args, @Nullable Token rightPar,
                            Token colon, Token newLine, Token indent, StatementList body, Token dedent, Token docstring) {
    super(astNode);
    this.decorators = decorators;
    this.classKeyword = classKeyword;
    this.name = name;
    this.leftPar = leftPar;
    this.args = args;
    this.rightPar = rightPar;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.docstring = docstring;
    this.body = body;
    this.dedent = dedent;
  }

  @Override
  public Kind getKind() {
    return Kind.CLASSDEF;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitClassDef(this);
  }

  @Override
  public List<Decorator> decorators() {
    return decorators;
  }

  @Override
  public Token classKeyword() {
    return classKeyword;
  }

  @Override
  public Name name() {
    return name;
  }

  @CheckForNull
  @Override
  public Token leftPar() {
    return leftPar;
  }

  @CheckForNull
  @Override
  public ArgList args() {
    return args;
  }

  @CheckForNull
  @Override
  public Token rightPar() {
    return rightPar;
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
  public Token docstring() {
    return docstring;
  }

  @Override
  public Set<Symbol> classFields() {
    return classFields;
  }

  @Override
  public Set<Symbol> instanceFields() {
    return instanceFields;
  }

  public void addClassField(Symbol field) {
    classFields.add(field);
  }

  public void addInstanceField(Symbol field) {
    instanceFields.add(field);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(decorators, Arrays.asList(classKeyword, name, leftPar, args, rightPar, colon, newLine, indent, docstring, body, dedent))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
