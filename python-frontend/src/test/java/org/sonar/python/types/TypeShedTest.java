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
package org.sonar.python.types;

import com.google.protobuf.TextFormat;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.AfterEach;
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
import static org.sonar.python.types.TypeShed.symbolWithFQN;
import static org.sonar.python.types.TypeShed.symbolsForModule;

class TypeShedTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void setPythonVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @AfterEach
  void resetPythonVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
  }

  private void setPythonVersions(Set<PythonVersionUtils.Version> pythonVersions) {
    ProjectPythonVersion.setCurrentVersions(pythonVersions);
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void classes() {
    ClassSymbol intClass = TypeShed.typeShedClass("int");
    assertThat(intClass.superClasses()).extracting(Symbol::name).containsExactly("object");
    assertThat(intClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(intClass.usages()).isEmpty();
    assertThat(intClass.declaredMembers()).allMatch(member -> member.usages().isEmpty());
    assertThat(TypeShed.typeShedClass("bool").superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("int");
  }

  @Test
  void str() {
    ClassSymbol strClass = TypeShed.typeShedClass("str");
    assertThat(strClass.hasUnresolvedTypeHierarchy()).isFalse();
    assertThat(strClass.superClasses()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(tuple(Kind.CLASS, "Sequence"));
    // python 3.9 support
    assertThat(strClass.resolveMember("removeprefix")).isNotEmpty();
    assertThat(strClass.resolveMember("removesuffix")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.fromString("3.8"));
    strClass = TypeShed.typeShedClass("str");
    assertThat(strClass.superClasses()).extracting(Symbol::kind, Symbol::name).containsExactlyInAnyOrder(tuple(Kind.CLASS, "Sequence"));
    assertThat(strClass.resolveMember("removeprefix")).isEmpty();
    assertThat(strClass.resolveMember("removesuffix")).isEmpty();

    setPythonVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void not_a_class() {
    assertThatThrownBy(() -> TypeShed.typeShedClass("repr")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void unknown_name() {
    assertThatThrownBy(() -> TypeShed.typeShedClass("xxx")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void none_type() {
    ClassSymbol noneType = TypeShed.typeShedClass("NoneType");
    assertThat(noneType.superClasses()).isEmpty();
  }

  @Test
  void typing_module() {
    Map<String, Symbol> symbols = symbolsForModule("typing");
    assertThat(symbols.values()).allMatch(symbol -> symbol.usages().isEmpty());
    // python3 specific
    assertThat(symbols.get("Awaitable").kind()).isEqualTo(Kind.CLASS);

    assertThat(symbols.get("Sequence").kind()).isEqualTo(Kind.CLASS);
  }

  @Test
  void stdlib_symbols() {
    Map<String, Symbol> mathSymbols = symbolsForModule("os.path");
    Symbol symbol = mathSymbols.get("realpath");
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
    Symbol acosSymbol = ((AmbiguousSymbolImpl) symbol).alternatives().iterator().next();
    assertThat(acosSymbol.kind()).isEqualTo(Kind.FUNCTION);
    FunctionSymbolImpl acosFunctionsymbol = (FunctionSymbolImpl) acosSymbol;
    assertThat(acosFunctionsymbol.parameters()).hasSizeBetween(1, 2);
    assertThat(acosFunctionsymbol.declaredReturnType()).isInstanceOf(AnyType.class);
    assertThat(TypeShed.symbolWithFQN("os.path", "os.path.realpath")).isSameAs(symbol);
    assertThat(mathSymbols.values()).allMatch(s -> s.usages().isEmpty());

    Map<String, Symbol> threadingSymbols = symbolsForModule("threading");
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Kind.CLASS);
    assertThat(threadingSymbols.values()).allMatch(s -> s.usages().isEmpty());

    Map<String, Symbol> imaplibSymbols = symbolsForModule("imaplib");
    assertThat(imaplibSymbols).isNotEmpty();
    assertThat(imaplibSymbols.values()).allMatch(s -> s.usages().isEmpty());
  }

  @Test
  void symbols_not_retrieved_when_within_same_project() {
    ProjectLevelSymbolTable projectLevelSymbolTable = Mockito.mock(ProjectLevelSymbolTable.class);
    TypeShed.setProjectLevelSymbolTable(projectLevelSymbolTable);

    Mockito.when(projectLevelSymbolTable.projectBasePackages()).thenReturn(Set.of("sklearn"));
    Map<String, Symbol> sklearnSymbols = symbolsForModule("sklearn.ensemble");
    assertThat(sklearnSymbols).isEmpty();
    sklearnSymbols = symbolsForModule("sklearn");
    assertThat(sklearnSymbols).isEmpty();
    Symbol symbol = symbolWithFQN("sklearn.ensemble", "sklearn.ensemble.RandomForestClassifier");
    assertThat(symbol).isNull();

    Mockito.when(projectLevelSymbolTable.projectBasePackages()).thenReturn(Set.of("unrelated"));
    sklearnSymbols = symbolsForModule("sklearn.ensemble");
    assertThat(sklearnSymbols).isNotEmpty();
    symbol = symbolWithFQN("sklearn.ensemble", "sklearn.ensemble.RandomForestClassifier");
    assertThat(symbol).isNotNull();
  }

  @Test
  void third_party_symbols() {
    Map<String, Symbol> flaskLoggingSymbols = symbolsForModule("flask.helpers");
    Symbol flaskSymbol = flaskLoggingSymbols.get("get_root_path");
    assertThat(flaskSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) flaskSymbol).declaredReturnType().canOnlyBe("str")).isTrue();
    assertThat(TypeShed.symbolWithFQN("flask.helpers", "flask.helpers.get_root_path")).isSameAs(flaskSymbol);
  }

  @Test
  void third_party_symbols_sklearn() {
    Map<String, Symbol> sklearnSymbol = symbolsForModule("sklearn.datasets._base");
    Symbol loadIrisSymbol = sklearnSymbol.get("load_iris");
    assertThat(loadIrisSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) loadIrisSymbol).declaredReturnType().canOnlyBe("tuple")).isTrue();
    assertThat(TypeShed.symbolWithFQN("sklearn.datasets._base", "sklearn.datasets._base.load_iris")).isSameAs(loadIrisSymbol);
  }

  @Test
  void should_resolve_packages() {
    assertThat(symbolsForModule("urllib")).isNotEmpty();
    assertThat(symbolsForModule("ctypes")).isNotEmpty();
    assertThat(symbolsForModule("email")).isNotEmpty();
    assertThat(symbolsForModule("json")).isNotEmpty();
    assertThat(symbolsForModule("docutils")).isNotEmpty();
    assertThat(symbolsForModule("ctypes.util")).isNotEmpty();
    assertThat(symbolsForModule("lib2to3.pgen2.grammar")).isNotEmpty();
    assertThat(symbolsForModule("cryptography")).isNotEmpty();
    // resolved but still empty
    assertThat(symbolsForModule("kazoo")).isEmpty();
  }

  @Test
  void package_symbols() {
    Map<String, Symbol> cursesSymbols = symbolsForModule("curses");
    Symbol wrapperSymbol = cursesSymbols.get("wrapper");
    assertThat(wrapperSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) wrapperSymbol).declaredReturnType()).isEqualTo(AnyType.ANY);
    assertThat(TypeShed.symbolWithFQN("curses", "curses.wrapper")).isSameAs(wrapperSymbol);
  }

  @Test
  void package_submodules_symbols() {
    Map<String, Symbol> asciiSymbols = symbolsForModule("curses.ascii");
    Symbol isalnumSymbol = asciiSymbols.get("isalnum");
    assertThat(isalnumSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(((FunctionSymbolImpl) isalnumSymbol).declaredReturnType().canOnlyBe("bool")).isTrue();
    assertThat(TypeShed.symbolWithFQN("curses.ascii", "curses.ascii.isalnum")).isSameAs(isalnumSymbol);
  }

  @Test
  void package_inner_submodules_symbols() {
    Map<String, Symbol> driverSymbols = symbolsForModule("lib2to3.pgen2.driver");
    Symbol loadGrammarSymbol = driverSymbols.get("load_grammar");
    // There is a small difference between Python 2 and Python 3 symbols: Python 2 uses Text instead of str
    assertThat(loadGrammarSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(TypeShed.symbolWithFQN("lib2to3.pgen2.driver", "lib2to3.pgen2.driver.load_grammar")).isSameAs(loadGrammarSymbol);
  }

  @Test
  void package_relative_import() {
    Map<String, Symbol> osSymbols = symbolsForModule("os");
    // The "import sys" is not part of the exported API (private import) in Typeshed
    // See: https://github.com/python/typeshed/blob/master/CONTRIBUTING.md#conventions
    assertThat(osSymbols).doesNotContainKey("sys");

    Map<String, Symbol> sqlite3Symbols = symbolsForModule("sqlite3");
    Symbol completeStatementFunction = sqlite3Symbols.get("complete_statement");
    assertThat(completeStatementFunction.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(completeStatementFunction.fullyQualifiedName()).isEqualTo("sqlite3.dbapi2.complete_statement");
    Set<String> sqlite3Dbapi2Symbols = symbolsForModule("sqlite3.dbapi2").keySet();
    // Python names with a leading underscore are not imported when using wildcard imports
    sqlite3Dbapi2Symbols.removeIf(s -> s.startsWith("_"));
    assertThat(sqlite3Symbols.keySet()).containsAll(sqlite3Dbapi2Symbols);

    Map<String, Symbol> requestsSymbols = symbolsForModule("requests");
    Symbol requestSymbol = requestsSymbols.get("request");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.FUNCTION);
    assertThat(requestSymbol.fullyQualifiedName()).isEqualTo("requests.api.request");
  }

  @Test
  void package_member_fqn_points_to_original_fqn() {
    Map<String, Symbol> symbols = symbolsForModule("flask");
    Symbol targetSymbol = symbols.get("Response");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("flask.wrappers.Response");
    assertThat(TypeShed.symbolWithFQN("flask", "flask.Response")).isSameAs(targetSymbol);
  }


  @Test
  void package_member_ambigous_symbol_common_fqn() {
    Map<String, Symbol> symbols = symbolsForModule("io");
    Symbol targetSymbol = symbols.get("FileIO");
    assertThat(targetSymbol.fullyQualifiedName()).isEqualTo("io.FileIO");
    assertThat(TypeShed.symbolWithFQN("io", "io.FileIO")).isSameAs(targetSymbol);
  }

  @Test
  void two_exported_symbols_with_same_local_names() {
    Map<String, Symbol> osSymbols = symbolsForModule("os");
    Map<String, Symbol> posixSymbols = symbolsForModule("posix");
    Symbol setupSymbolFromPosix = posixSymbols.get("stat_result");
    Symbol setupSymbolFromOs = osSymbols.get("stat_result");
    assertThat(setupSymbolFromPosix.kind()).isEqualTo(Kind.CLASS);
    assertThat(setupSymbolFromPosix.fullyQualifiedName()).isEqualTo("posix.stat_result");
    assertThat(setupSymbolFromOs.kind()).isEqualTo(Kind.CLASS);
    assertThat(setupSymbolFromOs.fullyQualifiedName()).isEqualTo("os.stat_result");
  }

  @Test
  void package_django() {
    Map<String, Symbol> djangoSymbols = symbolsForModule("django.http");
    Symbol responseSymbol = djangoSymbols.get("HttpResponse");
    assertThat(responseSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(responseSymbol.fullyQualifiedName()).isEqualTo("django.http.response.HttpResponse");
  }

  @Test
  void return_type_hints() {
    Map<String, Symbol> symbols = symbolsForModule("typing");
    assertThat(((FunctionSymbolImpl) symbols.get("get_args")).annotatedReturnTypeName()).isEqualTo("tuple");
    symbols = symbolsForModule("flask_mail");
    ClassSymbol mail = (ClassSymbol) symbols.get("Mail");
    assertThat(((FunctionSymbol) mail.declaredMembers().stream().iterator().next()).annotatedReturnTypeName()).isNull();
  }

  @Test
  void package_django_class_property_type() {
    Map<String, Symbol> djangoSymbols = symbolsForModule("django.http.request");
    Symbol requestSymbol = djangoSymbols.get("HttpRequest");
    assertThat(requestSymbol.kind()).isEqualTo(Kind.CLASS);
    assertThat(((ClassSymbol) requestSymbol).declaredMembers().iterator().next().annotatedTypeName()).isEqualTo("django.http.request.QueryDict");
  }

  @Test
  void package_lxml_reexported_symbol_fqn() {
    Map<String, Symbol> lxmlEtreeSymbols = symbolsForModule("lxml.etree");
    Symbol elementTreeSymbol = lxmlEtreeSymbols.get("ElementTree");
    assertThat(elementTreeSymbol.kind()).isEqualTo(Kind.CLASS);
    // FIXME: Original FQN is "xml.etree.ElementTree.ElementTree" and we should be able to retrieve it somehow
    assertThat(elementTreeSymbol.fullyQualifiedName()).isEqualTo("lxml.etree.ElementTree");
  }

  @Test
  void package_sqlite3_connect_type_in_ambiguous_symbol() {
    Map<String, Symbol> sqlite3Symbols = symbolsForModule("sqlite3");
    ClassSymbol connectionSymbol = (ClassSymbol) sqlite3Symbols.get("Connection");
    AmbiguousSymbol cursorFunction = connectionSymbol.declaredMembers().stream().filter(m -> "cursor".equals(m.name())).findFirst().map(AmbiguousSymbol.class::cast).get();
    Set<Symbol> alternatives = cursorFunction.alternatives();
    assertThat(alternatives)
      .hasSize(2)
      .allMatch(s -> "sqlite3.dbapi2.Cursor".equals(((FunctionSymbol) s).annotatedReturnTypeName()));
  }

  @Test
  void stub_files_symbols() {
    Collection<Symbol> mathSymbols = symbolsForModule("math").values();
    Collection<Symbol> djangoHttpSymbols = symbolsForModule("django.http").values();

    Collection<Symbol> symbols = TypeShed.stubFilesSymbols();
    assertThat(symbols)
      .containsAll(mathSymbols)
      .containsAll(djangoHttpSymbols);
  }

  @Test
  void deserialize_annoy_protobuf() {
    Map<String, Symbol> deserializedAnnoySymbols = symbolsForModule("annoy").values().stream()
      .collect(Collectors.toMap(Symbol::fullyQualifiedName, s -> s));
    assertThat(deserializedAnnoySymbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.CLASS, "annoy.annoylib.Annoy"), tuple(Kind.OTHER, "annoy.__annotations__"), tuple(Kind.OTHER, "annoy.__path__"));

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
    assertThat(symbolsForModule("NOT_EXISTENT")).isEmpty();
    assertThat(TypeShed.getSymbolsFromProtobufModule(null)).isEmpty();
    InputStream targetStream = new ByteArrayInputStream("foo".getBytes());
    assertThat(TypeShed.deserializedModule("mod", targetStream)).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Error while deserializing protobuf for module mod");
  }

  @Test
  void class_symbols_from_protobuf() throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol moduleSymbol = moduleSymbol(
      "fully_qualified_name: \"mod\"\n" +
        "classes {\n" +
        "  name: \"Base\"\n" +
        "  fully_qualified_name: \"mod.Base\"\n" +
        "  super_classes: \"builtins.object\"\n" +
        "}\n" +
        "classes {\n" +
        "  name: \"C\"\n" +
        "  fully_qualified_name: \"mod.C\"\n" +
        "  super_classes: \"builtins.str\"\n" +
        "}\n" +
        "classes {\n" +
        "  name: \"D\"\n" +
        "  fully_qualified_name: \"mod.D\"\n" +
        "  super_classes: \"NOT_EXISTENT\"\n" +
        "}");
    Map<String, Symbol> symbols = TypeShed.getSymbolsFromProtobufModule(moduleSymbol);
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
      "fully_qualified_name: \"mod\"\n" +
        "functions {\n" +
        "  name: \"foo\"\n" +
        "  fully_qualified_name: \"mod.foo\"\n" +
        "  parameters {\n" +
        "    name: \"p\"\n" +
        "    kind: POSITIONAL_OR_KEYWORD\n" +
        "  }\n" +
        "}\n" +
        "overloaded_functions {\n" +
        "  name: \"bar\"\n" +
        "  fullname: \"mod.bar\"\n" +
        "  definitions {\n" +
        "    name: \"bar\"\n" +
        "    fully_qualified_name: \"mod.bar\"\n" +
        "    parameters {\n" +
        "      name: \"x\"\n" +
        "      kind: POSITIONAL_OR_KEYWORD\n" +
        "    }\n" +
        "    has_decorators: true\n" +
        "    resolved_decorator_names: \"typing.overload\"\n" +
        "    is_overload: true\n" +
        "  }\n" +
        "  definitions {\n" +
        "    name: \"bar\"\n" +
        "    fully_qualified_name: \"mod.bar\"\n" +
        "    parameters {\n" +
        "      name: \"x\"\n" +
        "      kind: POSITIONAL_OR_KEYWORD\n" +
        "    }\n" +
        "    parameters {\n" +
        "      name: \"y\"\n" +
        "      kind: POSITIONAL_OR_KEYWORD\n" +
        "    }\n" +
        "    has_decorators: true\n" +
        "    resolved_decorator_names: \"typing.overload\"\n" +
        "    is_overload: true\n" +
        "  }\n" +
        "}\n");
    Map<String, Symbol> symbols = TypeShed.getSymbolsFromProtobufModule(moduleSymbol);
    assertThat(symbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder(tuple(Kind.FUNCTION, "mod.foo"), tuple(Kind.AMBIGUOUS, "mod.bar"));
    AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) symbols.get("bar");
    assertThat(ambiguousSymbol.alternatives()).extracting(Symbol::kind).containsExactly(Kind.FUNCTION, Kind.FUNCTION);

  }

  @Test
  void pythonVersions() {
    Symbol range = TypeShed.builtinSymbols().get("range");
    assertThat(((SymbolImpl) range).validForPythonVersions()).containsExactlyInAnyOrder("38", "39", "310", "311", "312", "313");
    assertThat(range.kind()).isEqualTo(Kind.CLASS);

    // python 2
    setPythonVersions(PythonVersionUtils.fromString("2.7"));
    range = TypeShed.builtinSymbols().get("range");
    // Python 3 symbols are returned, as no dedicated stubs for 2.7 are available anymore
    assertThat(range).isNotNull();

    // python 3
    setPythonVersions(PythonVersionUtils.fromString("3.8"));
    range = TypeShed.builtinSymbols().get("range");
    assertThat(((SymbolImpl) range).validForPythonVersions()).containsExactlyInAnyOrder("38", "39", "310", "311", "312", "313");
    assertThat(range.kind()).isEqualTo(Kind.CLASS);

    setPythonVersions(PythonVersionUtils.fromString("3.10"));
    ClassSymbol intSymbol = TypeShed.typeShedClass("int");
    assertThat(intSymbol.resolveMember("bit_count")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.fromString("3.12"));
    intSymbol = TypeShed.typeShedClass("int");
    assertThat(intSymbol.resolveMember("bit_count")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.fromString("3.13"));
    intSymbol = TypeShed.typeShedClass("int");
    assertThat(intSymbol.resolveMember("bit_count")).isNotEmpty();

    setPythonVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void testEolVersion() {
    setPythonVersions(PythonVersionUtils.fromString("3.7"));
    var intSymbol = TypeShed.typeShedClass("int");
    assertThat(intSymbol.resolveMember("bit_length")).isNotEmpty();

    assertThat(TypeShed.builtinSymbols().get("range")).isNotNull();

    setPythonVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void symbolWithFQN_should_be_consistent() {
    // smtplib imports typing.Sequence only in Python3, hence typing.Sequence has kind CLASS
    symbolsForModule("smtplib");
    Symbol sequence = TypeShed.symbolWithFQN("typing.Sequence");
    assertThat(sequence.kind()).isEqualTo(Kind.CLASS);
    Map<String, Symbol> typing = symbolsForModule("typing");
    assertThat(sequence).isSameAs(typing.get("Sequence"));
  }

  @Test
  void stubModules() {
    TypeShed.symbolsForModule("doesnotexist");
    TypeShed.symbolsForModule("math");
    assertThat(TypeShed.stubModules()).containsExactly("math");
  }

  private static SymbolsProtos.ModuleSymbol moduleSymbol(String protobuf) throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol.Builder builder = SymbolsProtos.ModuleSymbol.newBuilder();
    TextFormat.merge(protobuf, builder);
    return builder.build();
  }

  @Test
  void variables_from_protobuf() throws TextFormat.ParseException {
    SymbolsProtos.ModuleSymbol moduleSymbol = moduleSymbol(
      "fully_qualified_name: \"mod\"\n" +
        "vars {\n" +
        "  name: \"foo\"\n" +
        "  fully_qualified_name: \"mod.foo\"\n" +
        "  type_annotation {\n" +
        "    pretty_printed_name: \"builtins.str\"\n" +
        "    fully_qualified_name: \"builtins.str\"\n" +
        "  }\n" +
        "}\n" +
        "vars {\n" +
        "  name: \"bar\"\n" +
        "  fully_qualified_name: \"mod.bar\"\n" +
        "}\n");
    Map<String, Symbol> symbols = TypeShed.getSymbolsFromProtobufModule(moduleSymbol);
    assertThat(symbols.values()).extracting(Symbol::kind, Symbol::fullyQualifiedName, Symbol::annotatedTypeName)
      .containsExactlyInAnyOrder(tuple(Kind.OTHER, "mod.foo", "str"), tuple(Kind.OTHER, "mod.bar", null));
  }

  @Test
  void stubFilesSymbols_should_not_contain_ambiguous_symbols_of_classes() {
    // populate typeshed symbols cache by importing socket module
    symbolsForModule("socket");
    Collection<Symbol> symbols = TypeShed.stubFilesSymbols();
    assertThat(symbols).noneMatch(s -> s.is(Kind.AMBIGUOUS) && ((AmbiguousSymbol) s).alternatives().stream().allMatch(a -> a.is(Kind.CLASS)));
  }

  @Test
  void symbol_from_submodule_access() {
    Map<String, Symbol> os = symbolsForModule("os");
    SymbolImpl path = (SymbolImpl) os.get("path");
    Symbol samefile = path.getChildrenSymbolByName().get("samefile");
    assertThat(samefile).isNotNull();
    assertThat(samefile.fullyQualifiedName()).isEqualTo("os.path.samefile");

    Map<String, Symbol> osPath = symbolsForModule("os.path");
    Symbol samefileFromSubModule = osPath.get("samefile");
    assertThat(samefileFromSubModule).isSameAs(samefile);
  }

  @Test
  void typeshed_private_modules_should_not_affect_fqn() {
    Map<String, Symbol> socketModule = symbolsForModule("socket");
    ClassSymbol socket = (ClassSymbol) socketModule.get("socket");
    assertThat(socket.declaredMembers()).extracting(Symbol::name, Symbol::fullyQualifiedName).contains(tuple("connect", "socket.socket.connect"));
    assertThat(socket.superClasses()).extracting(Symbol::fullyQualifiedName).containsExactly("object");
  }

  @Test
  void overloaded_function_alias_has_function_annotated_type() {
    Map<String, Symbol> gettextModule = symbolsForModule("gettext");
    Symbol translation = gettextModule.get("translation");
    Symbol catalog = gettextModule.get("Catalog");
    assertThat(translation.kind()).isEqualTo(Kind.AMBIGUOUS);
    assertThat(catalog.annotatedTypeName()).isEqualTo("function");
  }

  @Test
  void stubFilesSymbols_third_party_symbols_should_not_be_null() {
    // six modules contain ambiguous symbols that only contain class symbols
    // however third party symbols don't have validForPythonVersions field set
    symbolsForModule("six");
    assertThat(TypeShed.stubFilesSymbols()).doesNotContainNull();
  }

  @Test
  void customDbStubs() {
    var pgdb = symbolsForModule("pgdb");
    assertThat(pgdb.get("connect")).isInstanceOf(FunctionSymbol.class);

    var mysql = symbolsForModule("mysql.connector");
    assertThat(mysql.get("connect")).isInstanceOf(FunctionSymbol.class);

    var pymysql = symbolsForModule("pymysql");
    assertThat(pymysql.get("connect")).isInstanceOf(FunctionSymbol.class);
  }
}
