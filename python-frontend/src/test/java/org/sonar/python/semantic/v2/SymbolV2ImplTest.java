/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Name;

class SymbolV2ImplTest {
  @Test
  void getSingleBindingUsage_should_return_empty_for_no_usages() {
    SymbolV2 symbol = new SymbolV2Impl("testSymbol");
    assertThat(symbol.getSingleBindingUsage()).isEmpty();
  }

  @Test
  void getSingleBindingUsage_should_return_empty_for_multiple_binding_usages() {
    SymbolV2Impl symbol = new SymbolV2Impl("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.ASSIGNMENT_LHS);

    assertThat(symbol.getSingleBindingUsage()).isEmpty();
  }

  @Test
  void getSingleBindingUsage_should_return_binding_usage_when_exactly_one_exists() {
    SymbolV2Impl symbol = new SymbolV2Impl("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.OTHER);

    Optional<UsageV2> singleBindingUsage = symbol.getSingleBindingUsage();
    assertThat(singleBindingUsage).isPresent();
    assertThat(singleBindingUsage.get().kind()).isEqualTo(UsageV2.Kind.IMPORT);
    assertThat(singleBindingUsage.get().tree()).isEqualTo(name1);
  }
}
