/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Symbol.Kind;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
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
    // python 3.9 support
    assertThat(strClass.resolveMember("removeprefix")).isNotEmpty();
    assertThat(strClass.resolveMember("removesuffix")).isNotEmpty();
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

  @Test
  public void package_relative_import() {
    Map<String, Symbol> osSymbols = TypeShed.symbolsForModule("os").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Symbol sysSymbol = osSymbols.get("sys");
    assertThat(sysSymbol.kind()).isEqualTo(Kind.AMBIGUOUS);

    Symbol timesResult = osSymbols.get("times_result");
    assertThat(timesResult.kind()).isEqualTo(Kind.CLASS);
    assertThat(timesResult.fullyQualifiedName()).isEqualTo("posix.times_result");

    Map<String, Symbol> requestsSymbols = TypeShed.symbolsForModule("requests").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Symbol requestSymbol = requestsSymbols.get("request");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(requestSymbol.fullyQualifiedName()).isEqualTo("requests.api.request");
  }

  @Test
  public void package_member_fqn_points_to_original_fqn() {
    Map<String, Symbol> symbols = TypeShed.symbolsForModule("flask").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol targetSymbol = symbols.get("Response");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("flask.wrappers.Response");
    assertThat(TypeShed.symbolWithFQN("flask", "flask.Response")).isSameAs(targetSymbol);
  }

  @Test
  public void package_member_ambigous_symbol_common_fqn() {
    Map<String, Symbol> symbols = TypeShed.symbolsForModule("io").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol targetSymbol = symbols.get("FileIO");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("io.FileIO");
    assertThat(TypeShed.symbolWithFQN("io", "io.FileIO")).isSameAs(targetSymbol);
  }

  @Test
  public void two_exported_symbols_with_same_local_names() {
    Map<String, Symbol> osSymbols = TypeShed.symbolsForModule("os").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Map<String, Symbol> posixSymbols = TypeShed.symbolsForModule("posix").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    Symbol setupSymbolFromPosix = posixSymbols.get("stat_result");
    Symbol setupSymbolFromOs = osSymbols.get("stat_result");
    assertThat(setupSymbolFromPosix.kind()).isEqualTo(Kind.AMBIGUOUS);
    assertThat(setupSymbolFromOs.kind()).isEqualTo(Kind.AMBIGUOUS);
  }

  @Test
  public void package_django() {
    Map<String, Symbol> djangoSymbols = TypeShed.symbolsForModule("django.http").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Symbol responseSymbol = djangoSymbols.get("HttpResponse");
    assertThat(responseSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(responseSymbol.fullyQualifiedName()).isEqualTo("django.http.response.HttpResponse");
  }

  @Test
  public void return_type_hints() {
    Map<String, Symbol> symbols = TypeShed.typingModuleSymbols().stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    assertThat(((FunctionSymbolImpl) symbols.get("get_args")).annotatedReturnTypeName()).isEqualTo("typing.Tuple");
    symbols = TypeShed.symbolsForModule("flask_mail").stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    ClassSymbol mail = (ClassSymbol) symbols.get("Mail");
    assertThat(((FunctionSymbol) mail.declaredMembers().stream().iterator().next()).annotatedReturnTypeName()).isNull();
  }

  @Test
  public void package_django_class_property_type() {
    Map<String, Symbol> djangoSymbols = TypeShed.symbolsForModule("django.http.request").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Symbol requestSymbol = djangoSymbols.get("HttpRequest");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(((ClassSymbol) requestSymbol).declaredMembers().iterator().next().annotatedTypeName()).isEqualTo("django.http.request.QueryDict");
  }

  @Test
  public void package_sqlite3_connect_type_in_ambiguous_symbol() {
    Map<String, Symbol> djangoSymbols = TypeShed.symbolsForModule("sqlite3").stream().collect(Collectors.toMap(Symbol::name, Function.identity(), AmbiguousSymbolImpl::create));
    Symbol requestSymbol = djangoSymbols.get("connect");
    assertThat(((FunctionSymbolImpl) ((((AmbiguousSymbolImpl) requestSymbol).alternatives()).toArray()[0])).annotatedReturnTypeName()).isEqualTo("sqlite3.dbapi2.Connection");
  }

  @Test
  public void stub_files_symbols() {
    Set<Symbol> mathSymbols = TypeShed.symbolsForModule("math");
    Set<Symbol> djangoHttpSymbols = TypeShed.symbolsForModule("django.http");

    Collection<Symbol> symbols = TypeShed.stubFilesSymbols();
    assertThat(symbols)
      .containsAll(mathSymbols)
      .containsAll(djangoHttpSymbols);
  }
}
