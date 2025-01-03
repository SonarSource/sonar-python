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
package org.sonar.python.semantic.v2.typeshed;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;

import static org.assertj.core.api.Assertions.assertThat;

class TypeShedDescriptorsProviderTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private static TypeShedDescriptorsProvider typeshedDescriptorsProvider() {
    return typeshedDescriptorsProvider(Set.of());
  }

  private static TypeShedDescriptorsProvider typeshedDescriptorsProvider(Set<String> projectBasePackages) {
    return new TypeShedDescriptorsProvider(projectBasePackages, PythonVersionUtils.allVersions());
  }
  
  @Test
  void builtinDescriptorsTest() {
    var provider = typeshedDescriptorsProvider();
    var builtinDescriptors = provider.builtinDescriptors();
    assertThat(builtinDescriptors).isNotEmpty();

    var intDescriptor = builtinDescriptors.get("int");
    assertThat(intDescriptor.fullyQualifiedName()).isEqualTo("int");

    assertThat(provider.builtinDescriptors()).isSameAs(builtinDescriptors);
  }

  @Test
  void builtin312DescriptorsTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of(), Set.of(PythonVersionUtils.Version.V_312));
    var builtinDescriptors = provider.builtinDescriptors();

    assertThat(builtinDescriptors).isNotEmpty();
  }

  @Test
  void builtinDisambiguation() {
    var provider = typeshedDescriptorsProvider();
    var builtinDescriptors = provider.builtinDescriptors();
    var floatDescriptor = builtinDescriptors.get("int");
    assertThat(floatDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);
    var newMember = ((ClassDescriptor) floatDescriptor).members().stream().filter(m -> m.name().equals("to_bytes")).findFirst().get();
    assertThat(newMember.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    assertThat(((AmbiguousDescriptor) newMember).alternatives()).hasSize(2);
  }

  @Test
  void builtinNoDisambiguation() {
    var provider = new TypeShedDescriptorsProvider(Set.of(), Set.of(PythonVersionUtils.Version.V_312));
    var builtinDescriptors = provider.builtinDescriptors();
    var floatDescriptor = builtinDescriptors.get("int");
    assertThat(floatDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);
    var newMember = ((ClassDescriptor) floatDescriptor).members().stream().filter(m -> m.name().equals("to_bytes")).findFirst().get();
    assertThat(newMember.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
  }

  @Test
  void typingDescriptorsTest() {
    var provider = typeshedDescriptorsProvider();
    var typing = provider.descriptorsForModule("typing");
    assertThat(typing).isNotEmpty();
  }

  @Test
  void moduleMatchesCurrentProjectTest() {
    var provider = typeshedDescriptorsProvider(Set.of("typing"));
    var typing = provider.descriptorsForModule("typing");
    assertThat(typing).isEmpty();
  }

  @Test
  void cacheTest() {
    var provider = typeshedDescriptorsProvider();
    var typing1 = provider.descriptorsForModule("typing");
    var typing2 = provider.descriptorsForModule("typing");
    assertThat(typing1).isSameAs(typing2);
  }


  @Test
  void stdlibDescriptors() {
    var provider = typeshedDescriptorsProvider();
    var osPathDescriptor = provider.descriptorsForModule("os.path");
    var descriptor = osPathDescriptor.get("realpath");
    assertThat(descriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    var realPathDescriptor = ((AmbiguousDescriptor) descriptor).alternatives().iterator().next();
    assertThat(realPathDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(((FunctionDescriptor) realPathDescriptor).parameters()).hasSizeBetween(1, 2);
    assertThat(((FunctionDescriptor) realPathDescriptor).annotatedReturnTypeName()).isNull();

    var threadingSymbols = provider.descriptorsForModule("threading");
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Descriptor.Kind.CLASS);

    var imaplibSymbols = provider.descriptorsForModule("imaplib");
    assertThat(imaplibSymbols).isNotEmpty();
  }

  @Test
  void testAnnotatedReturnTypeName() {
    var provider = typeshedDescriptorsProvider();
    var mathDescriptor = provider.descriptorsForModule("math");
    var acosDescriptor = mathDescriptor.get("acos");
    assertThat(acosDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(((FunctionDescriptor) acosDescriptor).annotatedReturnTypeName()).isEqualTo("float");
  }

  @Test
  void shouldResolvePackages() {
    var provider = typeshedDescriptorsProvider();
    assertThat(provider.descriptorsForModule("urllib")).isNotEmpty();
    assertThat(provider.descriptorsForModule("ctypes")).isNotEmpty();
    assertThat(provider.descriptorsForModule("email")).isNotEmpty();
    assertThat(provider.descriptorsForModule("json")).isNotEmpty();
    assertThat(provider.descriptorsForModule("docutils")).isNotEmpty();
    assertThat(provider.descriptorsForModule("ctypes.util")).isNotEmpty();
    assertThat(provider.descriptorsForModule("lib2to3.pgen2.grammar")).isNotEmpty();
    assertThat(provider.descriptorsForModule("cryptography")).isNotEmpty();
    // resolved but still empty
    assertThat(provider.descriptorsForModule("kazoo")).isEmpty();
  }

  @Test
  void customDbStubs() {
    var provider = typeshedDescriptorsProvider();
    var pgdb = provider.descriptorsForModule("pgdb");
    assertThat(pgdb.get("connect")).isInstanceOf(FunctionDescriptor.class);

    var mysql = provider.descriptorsForModule("mysql.connector");
    assertThat(mysql.get("connect")).isInstanceOf(FunctionDescriptor.class);

    var pymysql = provider.descriptorsForModule("pymysql");
    assertThat(pymysql.get("connect")).isInstanceOf(FunctionDescriptor.class);
  }

  @Test
  void unknownModule() {
    var provider = typeshedDescriptorsProvider();
    var unknownModule = provider.descriptorsForModule("unknown_module");
    assertThat(unknownModule).isEmpty();
  }

  @Test
  void readExceptionTest() {
    InputStream targetStream = new ByteArrayInputStream("foo".getBytes());
    assertThat(TypeShedDescriptorsProvider.deserializedModule("mod", targetStream)).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Error while deserializing protobuf for module mod");
  }

}
