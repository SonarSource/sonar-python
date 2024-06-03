/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import com.google.protobuf.TextFormat;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.Mockito;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Symbol.Kind;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.tuple;

class TypeShedTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void setPythonVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    org.sonar.python.types.TypeShed.resetBuiltinSymbols();
  }

  private void setPythonVersions(Set<PythonVersionUtils.Version> pythonVersions) {
    ProjectPythonVersion.setCurrentVersions(pythonVersions);
    org.sonar.python.types.TypeShed.resetBuiltinSymbols();
  }

  @Test
  void classes() {
    ClassSymbol intClass = org.sonar.python.types.TypeShed.typeShedClass("int");
    assertThat(intClass.superClasses()).extracting(Symbol::name).containsExactly("object");
    assertThat(intClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(intClass.usages()).isEmpty();
    assertThat(intClass.declaredMembers()).allMatch(member -> member.usages().isEmpty());
    assertThat(org.sonar.python.types.TypeShed.typeShedClass("bool").superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("int");
  }

  @Test
  void str() {
    ClassSymbol strClass = org.sonar.python.types.TypeShed.typeShedClass("str");
    assertThat(strClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(strClass.superClasses()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(tuple(Kind.CLASS, "Sequence"));
    // python 3.9 support
    assertThat(strClass.resolveMember("removeprefix")).isNotEmpty();
    assertThat(strClass.resolveMember("removesuffix")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.fromString("3.8"));
    strClass = org.sonar.python.types.TypeShed.typeShedClass("str");
    assertThat(strClass.superClasses()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(tuple(Kind.CLASS, "Sequence"));
    assertThat(strClass.resolveMember("removeprefix")).isEmpty();
    assertThat(strClass.resolveMember("removesuffix")).isEmpty();

    setPythonVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void not_a_class() {
    assertThatThrownBy(() -> org.sonar.python.types.TypeShed.typeShedClass("repr")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void unknown_name() {
    assertThatThrownBy(() -> org.sonar.python.types.TypeShed.typeShedClass("xxx")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void none_type() {
    ClassSymbol noneType = org.sonar.python.types.TypeShed.typeShedClass("NoneType");
    assertThat(noneType.superClasses()).isEmpty();
  }

  @Test
  void typing_module() {
    Map<String, Symbol> symbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("typing");
    assertThat(symbols.values()).allMatch(symbol -> symbol.usages().isEmpty());
    // python3 specific
    assertThat(symbols.get("Awaitable").kind()).isEqualTo(Kind.CLASS);

    assertThat(symbols.get("Sequence").kind()).isEqualTo(Kind.CLASS);
  }

  @Test
  void stdlib_symbols() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> mathSymbols = typeShed.symbolsForModule("math");
    Symbol symbol = mathSymbols.get("acos");
    assertThat(symbol.kind()).isEqualTo(Kind.AMBIGUOUS);
    Symbol acosSymbol = ((AmbiguousSymbolImpl) symbol).alternatives().iterator().next();
    assertThat(acosSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) acosSymbol).parameters()).hasSize(1);
    assertThat(((FunctionSymbolImpl) acosSymbol).declaredReturnType().canOnlyBe("float")).isTrue();
    assertThat(typeShed.symbolWithFQN("math", "math.acos")).isSameAs(symbol);
    assertThat(mathSymbols.values()).allMatch(s -> s.usages().isEmpty());

    Map<String, Symbol> threadingSymbols = typeShed.symbolsForModule("threading");
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Kind.CLASS);
    assertThat(threadingSymbols.values()).allMatch(s -> s.usages().isEmpty());

    Map<String, Symbol> imaplibSymbols = typeShed.symbolsForModule("imaplib");
    assertThat(imaplibSymbols).isNotEmpty();
    assertThat(imaplibSymbols.values()).allMatch(s -> s.usages().isEmpty());
  }

  @Test
  void symbols_not_retrieved_when_within_same_project() {
    ProjectLevelSymbolTable projectLevelSymbolTable = Mockito.mock(ProjectLevelSymbolTable.class);
    TypeShed typeShed = new TypeShed(projectLevelSymbolTable);

    Mockito.when(projectLevelSymbolTable.projectBasePackages()).thenReturn(Set.of("sklearn"));
    Map<String, Symbol> sklearnSymbols = typeShed.symbolsForModule("sklearn.ensemble");
    assertThat(sklearnSymbols).isEmpty();
    sklearnSymbols = typeShed.symbolsForModule("sklearn");
    assertThat(sklearnSymbols).isEmpty();
    Symbol symbol = typeShed.symbolWithFQN("sklearn.ensemble", "sklearn.ensemble.RandomForestClassifier");
    assertThat(symbol).isNull();

    Mockito.when(projectLevelSymbolTable.projectBasePackages()).thenReturn(Set.of("unrelated"));
    sklearnSymbols = typeShed.symbolsForModule("sklearn.ensemble");
    assertThat(sklearnSymbols).isNotEmpty();
    symbol = typeShed.symbolWithFQN("sklearn.ensemble", "sklearn.ensemble.RandomForestClassifier");
    assertThat(symbol).isNotNull();
  }

  @Test
  void third_party_symbols() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> flaskLoggingSymbols = typeShed.symbolsForModule("flask.helpers");
    Symbol flaskSymbol = flaskLoggingSymbols.get("get_root_path");
    assertThat(flaskSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) flaskSymbol).declaredReturnType().canOnlyBe("str")).isTrue();
    assertThat(typeShed.symbolWithFQN("flask.helpers", "flask.helpers.get_root_path")).isSameAs(flaskSymbol);
  }

  @Test
  void third_party_symbols_sklearn() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> sklearnSymbol = typeShed.symbolsForModule("sklearn.datasets._base");
    Symbol loadIrisSymbol = sklearnSymbol.get("load_iris");
    assertThat(loadIrisSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) loadIrisSymbol).declaredReturnType().canOnlyBe("tuple")).isTrue();
    assertThat(typeShed.symbolWithFQN("sklearn.datasets._base", "sklearn.datasets._base.load_iris")).isSameAs(loadIrisSymbol);
  }

  @Test
  void should_resolve_packages() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    assertThat(typeShed.symbolsForModule("urllib")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("ctypes")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("email")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("json")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("docutils")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("ctypes.util")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("lib2to3.pgen2.grammar")).isNotEmpty();
    assertThat(typeShed.symbolsForModule("cryptography")).isNotEmpty();
    // resolved but still empty
    assertThat(typeShed.symbolsForModule("kazoo")).isEmpty();
  }

  @Test
  void package_symbols() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> cursesSymbols = typeShed.symbolsForModule("curses");
    Symbol wrapperSymbol = cursesSymbols.get("wrapper");
    assertThat(wrapperSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(typeShed.symbolWithFQN("curses", "curses.wrapper")).isSameAs(wrapperSymbol);
  }

  @Test
  void package_submodules_symbols() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> asciiSymbols = typeShed.symbolsForModule("curses.ascii");
    Symbol isalnumSymbol = asciiSymbols.get("isalnum");
    assertThat(isalnumSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) isalnumSymbol).declaredReturnType().canOnlyBe("bool")).isTrue();
    assertThat(typeShed.symbolWithFQN("curses.ascii", "curses.ascii.isalnum")).isSameAs(isalnumSymbol);
  }

  @Test
  void package_inner_submodules_symbols() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> driverSymbols = typeShed.symbolsForModule("lib2to3.pgen2.driver");
    Symbol loadGrammarSymbol = driverSymbols.get("load_grammar");
    // There is a small difference between Python 2 and Python 3 symbols: Python 2 uses Text instead of str
    assertThat(loadGrammarSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(typeShed.symbolWithFQN("lib2to3.pgen2.driver", "lib2to3.pgen2.driver.load_grammar")).isSameAs(loadGrammarSymbol);
  }

  @Test
  void package_relative_import() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> osSymbols = typeShed.symbolsForModule("os");
    // The "import sys" is not part of the exported API (private import) in Typeshed
    // See: https://github.com/python/typeshed/blob/master/CONTRIBUTING.md#conventions
    assertThat(osSymbols).doesNotContainKey("sys");

    Map<String, Symbol> sqlite3Symbols = typeShed.symbolsForModule("sqlite3");
    Symbol completeStatementFunction = sqlite3Symbols.get("complete_statement");
    assertThat(completeStatementFunction.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(completeStatementFunction.fullyQualifiedName()).isEqualTo("sqlite3.dbapi2.complete_statement");
    Set<String> sqlite3Dbapi2Symbols = typeShed.symbolsForModule("sqlite3.dbapi2").keySet();
    // Python names with a leading underscore are not imported when using wildcard imports
    sqlite3Dbapi2Symbols.removeIf(s -> s.startsWith("_"));
    assertThat(sqlite3Symbols.keySet()).containsAll(sqlite3Dbapi2Symbols);

    Map<String, Symbol> requestsSymbols = typeShed.symbolsForModule("requests");
    Symbol requestSymbol = requestsSymbols.get("request");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(requestSymbol.fullyQualifiedName()).isEqualTo("requests.api.request");
  }

  @Test
  void package_member_fqn_points_to_original_fqn() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> symbols = typeShed.symbolsForModule("flask");
    Symbol targetSymbol = symbols.get("Response");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("flask.wrappers.Response");
    assertThat(typeShed.symbolWithFQN("flask", "flask.Response")).isSameAs(targetSymbol);
  }


  @Test
  void package_member_ambigous_symbol_common_fqn() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> symbols = typeShed.symbolsForModule("io");
    Symbol targetSymbol = symbols.get("FileIO");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("io.FileIO");
    assertThat(typeShed.symbolWithFQN("io", "io.FileIO")).isSameAs(targetSymbol);
  }

  @Test
  void two_exported_symbols_with_same_local_names() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> osSymbols = typeShed.symbolsForModule("os");
    Map<String, Symbol> posixSymbols = typeShed.symbolsForModule("posix");
    Symbol setupSymbolFromPosix = posixSymbols.get("stat_result");
    Symbol setupSymbolFromOs = osSymbols.get("stat_result");
    assertThat(setupSymbolFromPosix.kind()).isEqualTo(Kind.CLASS);
    assertThat(setupSymbolFromPosix.fullyQualifiedName()).isEqualTo("posix.stat_result");
    assertThat(setupSymbolFromOs.kind()).isEqualTo(Kind.CLASS);
    assertThat(setupSymbolFromOs.fullyQualifiedName()).isEqualTo("os.stat_result");
  }

  @Test
  void package_django() {
    Map<String, Symbol> djangoSymbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("django.http");
    Symbol responseSymbol = djangoSymbols.get("HttpResponse");
    assertThat(responseSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(responseSymbol.fullyQualifiedName()).isEqualTo("django.http.response.HttpResponse");
  }

  @Test
  void return_type_hints() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> symbols = typeShed.symbolsForModule("typing");
    assertThat(((FunctionSymbolImpl) symbols.get("get_args")).annotatedReturnTypeName()).isEqualTo("tuple");
    symbols = typeShed.symbolsForModule("flask_mail");
    ClassSymbol mail = (ClassSymbol) symbols.get("Mail");
    assertThat(((FunctionSymbol) mail.declaredMembers().stream().iterator().next()).annotatedReturnTypeName()).isNull();
  }

  @Test
  void package_django_class_property_type() {
    Map<String, Symbol> djangoSymbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("django.http.request");
    Symbol requestSymbol = djangoSymbols.get("HttpRequest");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(((ClassSymbol) requestSymbol).declaredMembers().iterator().next().annotatedTypeName()).isEqualTo("django.http.request" +
      ".QueryDict");
  }

  @Test
  void package_lxml_reexported_symbol_fqn() {
    Map<String, Symbol> lxmlEtreeSymbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("lxml.etree");
    Symbol elementTreeSymbol = lxmlEtreeSymbols.get("ElementTree");
    assertThat(elementTreeSymbol.kind()).isEqualTo(Kind.CLASS);
    // FIXME: Original FQN is "xml.etree.ElementTree.ElementTree" and we should be able to retrieve it somehow
    assertThat(elementTreeSymbol.fullyQualifiedName()).isEqualTo("lxml.etree.ElementTree");
  }

  @Test
  void package_sqlite3_connect_type_in_ambiguous_symbol() {
    Map<String, Symbol> sqlite3Symbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("sqlite3");
    ClassSymbol connectionSymbol = (ClassSymbol) sqlite3Symbols.get("Connection");
    AmbiguousSymbol cursorFunction =
      connectionSymbol.declaredMembers().stream().filter(m -> "cursor".equals(m.name())).findFirst().map(AmbiguousSymbol.class::cast).get();
    Set<Symbol> alternatives = cursorFunction.alternatives();
    assertThat(alternatives)
      .hasSize(2)
      .allMatch(s -> "sqlite3.dbapi2.Cursor".equals(((FunctionSymbol) s).annotatedReturnTypeName()));
  }

  @Test
  void deserialize_annoy_protobuf() {
    Map<String, Symbol> deserializedAnnoySymbols = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("annoy").values().stream()
      .collect(Collectors.toMap(Symbol::fullyQualifiedName, Function.identity()));
    assertThat(deserializedAnnoySymbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.CLASS, "annoy.annoylib.Annoy"), tuple(Kind.OTHER, "annoy.__annotations__"), tuple(Kind.OTHER,
        "annoy.__path__"));

    ClassSymbol annoyIndex = (ClassSymbol) deserializedAnnoySymbols.get("annoy.annoylib.Annoy");
    assertThat(annoyIndex.superClasses()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.CLASS, "object"));
    assertThat(annoyIndex.declaredMembers()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(
      tuple(Kind.FUNCTION, "__init__"),
      tuple(Kind.FUNCTION, "load"),
      tuple(Kind.FUNCTION, "save"),
      tuple(Kind.AMBIGUOUS, "get_nns_by_item"),
      tuple(Kind.AMBIGUOUS, "get_nns_by_vector"),
      tuple(Kind.FUNCTION, "get_item_vector"),
      tuple(Kind.FUNCTION, "add_item"),
      tuple(Kind.FUNCTION, "on_disk_build"),
      tuple(Kind.FUNCTION, "build"),
      tuple(Kind.FUNCTION, "unbuild"),
      tuple(Kind.FUNCTION, "unload"),
      tuple(Kind.FUNCTION, "get_distance"),
      tuple(Kind.FUNCTION, "get_n_items"),
      tuple(Kind.FUNCTION, "get_n_trees"),
      tuple(Kind.FUNCTION, "verbose"),
      tuple(Kind.FUNCTION, "set_seed"),
      tuple(Kind.OTHER, "f")
    );
    assertThat(annoyIndex.hasDecorators()).isFalse();
    assertThat(annoyIndex.definitionLocation()).isNull();
  }

  @Test
  void deserialize_nonexistent_or_incorrect_protobuf() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    assertThat(new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("NOT_EXISTENT")).isEmpty();
    assertThat(typeShed.getSymbolsFromProtobufModule(null)).isEmpty();
    InputStream targetStream = new ByteArrayInputStream("foo".getBytes());
    assertThat(TypeShed.deserializedModule("mod", targetStream)).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Error while deserializing protobuf for module mod");
  }

  @Test
  void class_symbols_from_protobuf() throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol moduleSymbol = moduleSymbol(
      """
        fully_qualified_name: "mod"
        classes {
          name: "Base"
          fully_qualified_name: "mod.Base"
          super_classes: "builtins.object"
        }
        classes {
          name: "C"
          fully_qualified_name: "mod.C"
          super_classes: "builtins.str"
        }
        classes {
          name: "D"
          fully_qualified_name: "mod.D"
          super_classes: "NOT_EXISTENT"
        }
        """);
    Map<String, Symbol> symbols = new TypeShed(ProjectLevelSymbolTable.empty()).getSymbolsFromProtobufModule(moduleSymbol);
    assertThat(symbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.CLASS, "mod.Base"), tuple(Kind.CLASS, "mod.C"), tuple(Kind.CLASS, "mod.D"));

    ClassSymbol C = (ClassSymbol) symbols.get("C");
    assertThat(C.superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("str");
    ClassSymbol D = (ClassSymbol) symbols.get("D");
    assertThat(D.superClasses()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactly(tuple(Kind.OTHER, "NOT_EXISTENT"));
  }

  @Test
  void function_symbols_from_protobuf() throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol moduleSymbol = moduleSymbol(
      """
        fully_qualified_name: "mod"
        functions {
          name: "foo"
          fully_qualified_name: "mod.foo"
          parameters {
            name: "p"
            kind: POSITIONAL_OR_KEYWORD
          }
        }
        overloaded_functions {
          name: "bar"
          fullname: "mod.bar"
          definitions {
            name: "bar"
            fully_qualified_name: "mod.bar"
            parameters {
              name: "x"
              kind: POSITIONAL_OR_KEYWORD
            }
            has_decorators: true
            resolved_decorator_names: "typing.overload"
            is_overload: true
          }
          definitions {
            name: "bar"
            fully_qualified_name: "mod.bar"
            parameters {
              name: "x"
              kind: POSITIONAL_OR_KEYWORD
            }
            parameters {
              name: "y"
              kind: POSITIONAL_OR_KEYWORD
            }
            has_decorators: true
            resolved_decorator_names: "typing.overload"
            is_overload: true
          }
        }
        """);
    Map<String, Symbol> symbols = new TypeShed(ProjectLevelSymbolTable.empty()).getSymbolsFromProtobufModule(moduleSymbol);
    assertThat(symbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.FUNCTION, "mod.foo"), tuple(Kind.AMBIGUOUS, "mod.bar"));
    AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) symbols.get("bar");
    assertThat(ambiguousSymbol.alternatives()).extracting(Symbol::kind).containsExactly(Kind.FUNCTION, Kind.FUNCTION);

  }

  @Test
  void pythonVersions() {
    Symbol range = new TypeShed(ProjectLevelSymbolTable.empty()).builtinSymbols().get("range");
    assertThat(((SymbolImpl) range).validForPythonVersions()).containsExactlyInAnyOrder("36", "37", "38", "39", "310", "311");
    assertThat(range.kind()).isEqualTo(Kind.CLASS);

    // python 2
    setPythonVersions(PythonVersionUtils.fromString("2.7"));
    range = new TypeShed(ProjectLevelSymbolTable.empty()).builtinSymbols().get("range");
    // Python 3 symbols are returned, as no dedicated stubs for 2.7 are available anymore
    assertThat(range).isNotNull();

    // python 3
    setPythonVersions(PythonVersionUtils.fromString("3.8"));
    range = new TypeShed(ProjectLevelSymbolTable.empty()).builtinSymbols().get("range");
    assertThat(((SymbolImpl) range).validForPythonVersions()).containsExactlyInAnyOrder("36", "37", "38", "39", "310", "311");
    assertThat(range.kind()).isEqualTo(Kind.CLASS);

    setPythonVersions(PythonVersionUtils.fromString("3.10"));
    var intSymbol = (ClassSymbol) new TypeShed(ProjectLevelSymbolTable.empty()).builtinSymbols().get("int");
    assertThat(intSymbol.resolveMember("bit_count")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.fromString("3.12"));
    intSymbol = (ClassSymbol) new TypeShed(ProjectLevelSymbolTable.empty()).builtinSymbols().get("int");
    assertThat(intSymbol.resolveMember("bit_count")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void symbolWithFQN_should_be_consistent() {
    // smtplib imports typing.Sequence only in Python3, hence typing.Sequence has kind CLASS
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    typeShed.symbolsForModule("smtplib");
    Symbol sequence = typeShed.symbolWithFQN("typing.Sequence");
    assertThat(sequence.kind()).isEqualTo(Kind.CLASS);
    Map<String, Symbol> typing = typeShed.symbolsForModule("typing");
    assertThat(sequence).isSameAs(typing.get("Sequence"));
  }

  private static SymbolsProtos.ModuleSymbol moduleSymbol(String protobuf) throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol.Builder builder = SymbolsProtos.ModuleSymbol.newBuilder();
    TextFormat.merge(protobuf, builder);
    return builder.build();
  }

  @Test
  void variables_from_protobuf() throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol moduleSymbol = moduleSymbol(
      """
        fully_qualified_name: "mod"
        vars {
          name: "foo"
          fully_qualified_name: "mod.foo"
          type_annotation {
            pretty_printed_name: "builtins.str"
            fully_qualified_name: "builtins.str"
          }
        }
        vars {
          name: "bar"
          fully_qualified_name: "mod.bar"
        }
        """);
    Map<String, Symbol> symbols = new TypeShed(ProjectLevelSymbolTable.empty()).getSymbolsFromProtobufModule(moduleSymbol);
    assertThat(symbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName, Symbol::annotatedTypeName)
      .containsExactlyInAnyOrder(tuple(Kind.OTHER, "mod.foo", "str"), tuple(Kind.OTHER, "mod.bar", null));
  }

  @Test
  void symbol_from_submodule_access() {
    var typeShed = new TypeShed(ProjectLevelSymbolTable.empty());
    Map<String, Symbol> os = typeShed.symbolsForModule("os");
    SymbolImpl path = (SymbolImpl) os.get("path");
    Symbol samefile = path.getChildrenSymbolByName().get("samefile");
    assertThat(samefile).isNotNull();
    assertThat(samefile.fullyQualifiedName()).isEqualTo("os.path.samefile");

    Map<String, Symbol> osPath = typeShed.symbolsForModule("os.path");
    Symbol samefileFromSubModule = osPath.get("samefile");
    assertThat(samefileFromSubModule).isSameAs(samefile);
  }

  @Test
  void typeshed_private_modules_should_not_affect_fqn() {
    Map<String, Symbol> socketModule = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("socket");
    ClassSymbol socket = (ClassSymbol) socketModule.get("socket");
    assertThat(socket.declaredMembers()).extracting(Symbol::name, Symbol::fullyQualifiedName).contains(tuple("connect", "socket.socket" +
      ".connect"));
    assertThat(socket.superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("object");
  }

  @Test
  void overloaded_function_alias_has_function_annotated_type() {
    Map<String, Symbol> gettextModule = new TypeShed(ProjectLevelSymbolTable.empty()).symbolsForModule("gettext");
    Symbol translation = gettextModule.get("translation");
    Symbol catalog = gettextModule.get("Catalog");
    assertThat(translation.kind()).isEqualTo(Kind.AMBIGUOUS);
    assertThat(catalog.annotatedTypeName()).isEqualTo("function");
  }
}
