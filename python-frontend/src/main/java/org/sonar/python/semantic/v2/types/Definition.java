/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2.types;

import javax.annotation.Nonnull;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;

/**
 * This represents a class or function definition
 * It can be modelled as an assignment of a class / function type to their symbol
 */
public class Definition extends Propagation {

  public Definition(SymbolV2 symbol, Name name) {
    super(symbol, name);
  }


  @Override
  public PythonType rhsType() {
    return lhsName().typeV2();
  }

  @Override
  @Nonnull
  Tree scopeTree(Name name) {
    return scopeTree(name, lhsName().parent());
  }
}
