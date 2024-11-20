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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class FileInputImpl extends PyTree implements FileInput {

  private final StatementList statements;
  private final Token endOfFile;
  private final StringLiteral docstring;
  private final Set<Symbol> globalVariables = new HashSet<>();

  public FileInputImpl(@Nullable StatementList statements, Token endOfFile, @Nullable StringLiteral docstring) {
    this.statements = statements;
    this.endOfFile = endOfFile;
    this.docstring = docstring;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.FILE_INPUT;
  }

  @Override
  @CheckForNull
  public StatementList statements() {
    return statements;
  }

  @CheckForNull
  @Override
  public StringLiteral docstring() {
    return docstring;
  }

  @Override
  public Set<Symbol> globalVariables() {
    return globalVariables;
  }

  public void addGlobalVariables(Symbol globalVariable) {
    globalVariables.add(globalVariable);
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitFileInput(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(statements, endOfFile).filter(Objects::nonNull).toList();
  }
}
