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
package org.sonar.plugins.python.api.symbols;

import java.util.List;
import javax.annotation.CheckForNull;
import com.google.common.annotations.Beta;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.types.InferredType;

public interface FunctionSymbol extends Symbol {
  List<Parameter> parameters();

  /**
   * When true, it denotes a function symbol for a function stub.
   * <pre>
   *   def fn(p1, p2): ...
   * </pre>
   */
  boolean isStub();

  boolean isAsynchronous();

  boolean hasVariadicParameter();

  boolean isInstanceMethod();

  /**
   * Returns the known names of the decorators.
   * It might not be a reliable way to infer the number of decorators
   * as they can take more complex expressions since Python 3.9
   */
  List<String> decorators();

  boolean hasDecorators();

  @CheckForNull
  LocationInFile definitionLocation();

  /**
   * Returns fully qualified name of the return type if any
   */
  @Beta
  @CheckForNull
  String annotatedReturnTypeName();

  interface Parameter {
    @CheckForNull
    String name();
    InferredType declaredType();
    boolean hasDefaultValue();
    boolean isVariadic();
    boolean isKeywordOnly();
    boolean isPositionalOnly();

    /**
     * Returns true for **kwargs
     */
    boolean isKeywordVariadic();

    /**
     * Returns true for *args
     */
    boolean isPositionalVariadic();
    @CheckForNull
    LocationInFile location();
  }
}
