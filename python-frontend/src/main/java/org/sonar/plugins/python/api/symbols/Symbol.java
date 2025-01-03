/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import com.google.common.annotations.Beta;
import java.util.List;
import javax.annotation.CheckForNull;

public interface Symbol {

  String name();

  List<Usage> usages();

  @CheckForNull
  String fullyQualifiedName();

  boolean is(Kind... kinds);

  Kind kind();

  /**
   * Returns fully qualified name of the type if any
   */
  @Beta
  @CheckForNull
  String annotatedTypeName();

  enum Kind {
    FUNCTION,
    CLASS,
    AMBIGUOUS,
    OTHER
  }
}
