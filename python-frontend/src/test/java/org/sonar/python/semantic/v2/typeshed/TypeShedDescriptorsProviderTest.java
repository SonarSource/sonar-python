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
package org.sonar.python.semantic.v2.typeshed;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;

import static org.assertj.core.api.Assertions.assertThat;

class TypeShedDescriptorsProviderTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void setup() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void builtinSymbolsTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var builtinSymbols = provider.builtinSymbols();
    Assertions.assertThat(builtinSymbols).isNotEmpty();

    var intDescriptor = builtinSymbols.get("int");
    Assertions.assertThat(intDescriptor.fullyQualifiedName()).isEqualTo("int");
  }

  @Test
  void builtin312SymbolsTest() {
    ProjectPythonVersion.setCurrentVersions(Set.of(PythonVersionUtils.Version.V_311));
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var builtinSymbols = provider.builtinSymbols();

    Assertions.assertThat(builtinSymbols).isNotEmpty();
  }

  @Test
  void typingSymbolsTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var typing = provider.symbolsForModule("typing");

    Assertions.assertThat(typing).isNotEmpty();
  }

  @Test
  void stdlib_symbols() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var mathDescriptors = provider.symbolsForModule("math");
    var descriptor = mathDescriptors.get("acos");
    assertThat(descriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    var acosDescriptor = ((AmbiguousDescriptor) descriptor).alternatives().iterator().next();
    assertThat(acosDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(((FunctionDescriptor) acosDescriptor).parameters()).hasSize(1);
    assertThat(((FunctionDescriptor) acosDescriptor).annotatedReturnTypeName()).isEqualTo("float");

    var threadingSymbols = provider.symbolsForModule("threading");
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Descriptor.Kind.CLASS);

    var imaplibSymbols = provider.symbolsForModule("imaplib");
    assertThat(imaplibSymbols).isNotEmpty();
  }

  @Test
  void should_resolve_packages() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    assertThat(provider.symbolsForModule("urllib")).isNotEmpty();
    assertThat(provider.symbolsForModule("ctypes")).isNotEmpty();
    assertThat(provider.symbolsForModule("email")).isNotEmpty();
    assertThat(provider.symbolsForModule("json")).isNotEmpty();
    assertThat(provider.symbolsForModule("docutils")).isNotEmpty();
    assertThat(provider.symbolsForModule("ctypes.util")).isNotEmpty();
    assertThat(provider.symbolsForModule("lib2to3.pgen2.grammar")).isNotEmpty();
    assertThat(provider.symbolsForModule("cryptography")).isNotEmpty();
    // resolved but still empty
    assertThat(provider.symbolsForModule("kazoo")).isEmpty();
  }

  @Test
  void unknown_module() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var unknownModule = provider.symbolsForModule("unknown_module");
    assertThat(unknownModule).isEmpty();
  }

  @Test
  void readExceptionTest() {
    InputStream targetStream = new ByteArrayInputStream("foo".getBytes());
    assertThat(TypeShedDescriptorsProvider.deserializedModule("mod", targetStream)).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Error while deserializing protobuf for module mod");
  }

}
