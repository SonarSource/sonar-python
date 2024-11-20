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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.TypeParams;

public class ClassDefImpl extends PyTree implements ClassDef {
  private final List<Decorator> decorators;
  private final Token classKeyword;
  private final Name name;
  private final TypeParams typeParams;
  private final Token leftPar;
  private final ArgList args;
  private final Token rightPar;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final Token dedent;
  private final StatementList body;
  private final StringLiteral docstring;
  private final Set<Symbol> classFields = new HashSet<>();
  private final Set<Symbol> instanceFields = new HashSet<>();

  public ClassDefImpl(List<Decorator> decorators, Token classKeyword, Name name,
                            @Nullable TypeParams typeParams, @Nullable Token leftPar, @Nullable ArgList args, @Nullable Token rightPar,
                            Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList body, @Nullable Token dedent, StringLiteral docstring) {
    this.decorators = decorators;
    this.classKeyword = classKeyword;
    this.name = name;
    this.typeParams = typeParams;
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
  public TypeParams typeParams() {
    return typeParams;
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
  public StringLiteral docstring() {
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
  public List<Tree> computeChildren() {
    return Stream.of(decorators, Arrays.asList(classKeyword, name, typeParams, leftPar, args, rightPar, colon, newLine, indent, body, dedent))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
