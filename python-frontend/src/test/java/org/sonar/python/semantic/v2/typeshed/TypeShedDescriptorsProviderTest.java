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
  void builtinDescriptorsTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var builtinDescriptors = provider.builtinDescriptors();
    Assertions.assertThat(builtinDescriptors).isNotEmpty();

    var intDescriptor = builtinDescriptors.get("int");
    Assertions.assertThat(intDescriptor.fullyQualifiedName()).isEqualTo("int");

    Assertions.assertThat(provider.builtinDescriptors()).isSameAs(builtinDescriptors);
  }

  @Test
  void builtin312DescriptorsTest() {
    ProjectPythonVersion.setCurrentVersions(Set.of(PythonVersionUtils.Version.V_311));
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var builtinDescriptors = provider.builtinDescriptors();

    Assertions.assertThat(builtinDescriptors).isNotEmpty();
  }

  @Test
  void typingDescriptorsTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var typing = provider.descriptorsForModule("typing");
    Assertions.assertThat(typing).isNotEmpty();
  }

  @Test
  void moduleMatchesCurrentProjectTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of("typing"));
    var typing = provider.descriptorsForModule("typing");
    Assertions.assertThat(typing).isEmpty();
  }

  @Test
  void cacheTest() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var typing1 = provider.descriptorsForModule("typing");
    var typing2 = provider.descriptorsForModule("typing");
    Assertions.assertThat(typing1).isSameAs(typing2);
  }


  @Test
  void stdlibDescriptors() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
    var mathDescriptors = provider.descriptorsForModule("math");
    var descriptor = mathDescriptors.get("acos");
    assertThat(descriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    var acosDescriptor = ((AmbiguousDescriptor) descriptor).alternatives().iterator().next();
    assertThat(acosDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(((FunctionDescriptor) acosDescriptor).parameters()).hasSize(1);
    assertThat(((FunctionDescriptor) acosDescriptor).annotatedReturnTypeName()).isEqualTo("float");

    var threadingSymbols = provider.descriptorsForModule("threading");
    assertThat(threadingSymbols.get("Thread").kind()).isEqualTo(Descriptor.Kind.CLASS);

    var imaplibSymbols = provider.descriptorsForModule("imaplib");
    assertThat(imaplibSymbols).isNotEmpty();
  }

  @Test
  void shouldResolvePackages() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
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
  void unknownModule() {
    var provider = new TypeShedDescriptorsProvider(Set.of());
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