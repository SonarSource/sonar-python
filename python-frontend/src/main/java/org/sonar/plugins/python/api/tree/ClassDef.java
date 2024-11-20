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
package org.sonar.plugins.python.api.tree;

import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;

/**
 * <pre>
 *   {@link #decorators()}
 *   class {@link #name()}( {@link #args()} ):
 *     {@link #docstring()}
 *     {@link #body()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/compound_stmts.html#class-definitions
 */
public interface ClassDef extends Statement {

  List<Decorator> decorators();

  Token classKeyword();

  Name name();

  @CheckForNull
  TypeParams typeParams();

  @CheckForNull
  Token leftPar();

  /**
   * null if class is defined without args {@code class Foo:...} or {@code class Foo():...}
   */
  @CheckForNull
  ArgList args();

  @CheckForNull
  Token rightPar();

  Token colon();

  StatementList body();

  @CheckForNull
  StringLiteral docstring();

  /**
   * Contains fields and methods symbols
   */
  Set<Symbol> classFields();

  Set<Symbol> instanceFields();

}
