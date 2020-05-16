/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Symbol.Kind;
import org.sonar.python.semantic.FunctionSymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

public class TypeShedTest {

  @Test
  public void classes() {
    ClassSymbol intClass = TypeShed.typeShedClass("int");
    assertThat(intClass.superClasses()).isEmpty();
    assertThat(intClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(intClass.usages()).isEmpty();
    assertThat(intClass.declaredMembers()).allMatch(member -> member.usages().isEmpty());
    assertThat(TypeShed.typeShedClass("bool").superClasses()).containsExactly(intClass);
  }

  @Test
  public void str() {
    ClassSymbol strClass = TypeShed.typeShedClass("str");
    assertThat(strClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(strClass.superClasses()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(tuple(Kind.CLASS, "object"), tuple(Kind.AMBIGUOUS, "Sequence"));
  }

  @Test(expected = IllegalArgumentException.class)
  public void not_a_class() {
    TypeShed.typeShedClass("repr");
  }

  @Test(expected = IllegalArgumentException.class)
  public void unknown_name() {
    TypeShed.typeShedClass("xxx");
  }

  @Test
  public void none_type() {
    ClassSymbol noneType = TypeShed.typeShedClass("NoneType");
    assertThat(noneType.superClasses()).isEmpty();
  }

  @Test
  public void typing_module() {
    Map<String, Symbol> symbols = TypeShed.typingModuleSymbols().stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    assertThat(symbols.values()).allMatch(symbol -> symbol.usages().isEmpty());
    // python3 specific
    assertThat(symbols.get("Awaitable").kind()).isEqualTo(Kind.CLASS);
    // overlap btw python2 and python3
    assertThat(symbols.get("Iterator").kind()).isEqualTo(Kind.AMBIGUOUS);
  }

  @Test
  public void stdlib_symbols() {
    Map<String, Symbol> mathSymbols = TypeShed.symbolsForModule("math").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol acosSymbol = mathSymbols.get("acos");
    assertThat(acosSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) acosSymbol).declaredReturnType().canOnlyBe("float")).isTrue();
    assertThat(TypeShed.symbolWithFQN("math", "math.acos")).isSameAs(acosSymbol);
    assertThat(mathSymbols.values()).allMatch(symbol -> symbol.usages().isEmpty());

    Map<String, Symbol> threadingSymbols = TypeShed.symbolsForModule("threading").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Kind.CLASS);
    assertThat(threadingSymbols.values()).allMatch(symbol -> symbol.usages().isEmpty());

    Map<String, Symbol> imaplibSymbols = TypeShed.symbolsForModule("imaplib").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    assertThat(imaplibSymbols).isNotEmpty();
    assertThat(imaplibSymbols.values()).allMatch(symbol -> symbol.usages().isEmpty());
  }

  @Test
  public void third_party_symbols() {
    Map<String, Symbol> emojiSymbols = TypeShed.symbolsForModule("emoji").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol emojizeSymbol = emojiSymbols.get("emojize");
    assertThat(emojizeSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) emojizeSymbol).declaredReturnType().canOnlyBe("str")).isTrue();
    assertThat(TypeShed.symbolWithFQN("emoji", "emoji.emojize")).isSameAs(emojizeSymbol);
  }

  @Test
  public void should_resolve_packages() {
    assertThat(TypeShed.symbolsForModule("urllib")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("ctypes")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("email")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("json")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("docutils")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("ctypes.util")).isNotEmpty();
    assertThat(TypeShed.symbolsForModule("lib2to3.pgen2.grammar")).isNotEmpty();
    // resolved but still empty
    assertThat(TypeShed.symbolsForModule("cryptography")).isEmpty();
    assertThat(TypeShed.symbolsForModule("kazoo")).isEmpty();
  }

  @Test
  public void package_symbols() {
    Map<String, Symbol> cursesSymbols = TypeShed.symbolsForModule("curses").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol wrapperSymbol = cursesSymbols.get("wrapper");
    assertThat(wrapperSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) wrapperSymbol).declaredReturnType()).isEqualTo(AnyType.ANY);
    assertThat(TypeShed.symbolWithFQN("curses", "curses.wrapper")).isSameAs(wrapperSymbol);
  }

  @Test
  public void package_submodules_symbols() {
    Map<String, Symbol> asciiSymbols = TypeShed.symbolsForModule("curses.ascii").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol isalnumSymbol = asciiSymbols.get("isalnum");
    assertThat(isalnumSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) isalnumSymbol).declaredReturnType().canOnlyBe("bool")).isTrue();
    assertThat(TypeShed.symbolWithFQN("curses.ascii", "curses.ascii.isalnum")).isSameAs(isalnumSymbol);
  }

  @Test
  public void package_inner_submodules_symbols() {
    Map<String, Symbol> driverSymbols = TypeShed.symbolsForModule("lib2to3.pgen2.driver").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol loadGrammarSymbol = driverSymbols.get("load_grammar");
    assertThat(loadGrammarSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(TypeShed.symbolWithFQN("lib2to3.pgen2.driver", "lib2to3.pgen2.driver.load_grammar")).isSameAs(loadGrammarSymbol);
  }
}
