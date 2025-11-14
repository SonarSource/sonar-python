/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.tree.Name;

class SymbolV2Test {
  @Test
  void getSingleBindingUsage_should_return_empty_for_no_usages() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    assertThat(symbol.getSingleBindingUsage()).isEmpty();
  }

  @Test
  void getSingleBindingUsage_should_return_empty_for_multiple_binding_usages() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.ASSIGNMENT_LHS);

    assertThat(symbol.getSingleBindingUsage()).isEmpty();
  }

  @Test
  void getSingleBindingUsage_should_return_binding_usage_when_exactly_one_exists() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.OTHER);

    Optional<UsageV2> singleBindingUsage = symbol.getSingleBindingUsage();
    assertThat(singleBindingUsage).isPresent();
    assertThat(singleBindingUsage.get().kind()).isEqualTo(UsageV2.Kind.IMPORT);
    assertThat(singleBindingUsage.get().tree()).isEqualTo(name1);
  }

  @Test
  void hasSingleBindingUsage_should_return_false_for_no_usages() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    assertThat(symbol.hasSingleBindingUsage()).isFalse();
  }

  @Test
  void hasSingleBindingUsage_should_return_false_for_multiple_binding_usages() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.ASSIGNMENT_LHS);

    assertThat(symbol.hasSingleBindingUsage()).isFalse();
  }

  @Test
  void hasSingleBindingUsage_should_return_true_for_single_binding_usage() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    Name name1 = mock(Name.class);
    Name name2 = mock(Name.class);

    symbol.addUsage(name1, UsageV2.Kind.IMPORT);
    symbol.addUsage(name2, UsageV2.Kind.OTHER);

    assertThat(symbol.hasSingleBindingUsage()).isTrue();
  }

  @Test
  void hasSingleBindingUsage_should_return_false_for_only_non_binding_usages() {
    SymbolV2 symbol = new SymbolV2("testSymbol");
    Name name = mock(Name.class);

    symbol.addUsage(name, UsageV2.Kind.OTHER);

    assertThat(symbol.hasSingleBindingUsage()).isFalse();
  }
}
