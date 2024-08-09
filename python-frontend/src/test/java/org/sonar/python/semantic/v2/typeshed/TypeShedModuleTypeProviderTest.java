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

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;

import static org.assertj.core.api.Assertions.assertThat;

class TypeShedModuleTypeProviderTest {


  @Test
  void getBuiltinsModuleType() {
    var provider = new TypeShedModuleTypeProvider();
    var builtins = provider.getBuiltinModuleType();
    Assertions.assertThat(builtins).isNotNull();
  }

  @Test
  void getModuleType() {
    var provider = new TypeShedModuleTypeProvider();
    var moduleType1 = provider.getModuleType("babel", null);
    var moduleType2 = provider.getModuleType("babel", null);
    Assertions.assertThat(moduleType1).isNotNull().isSameAs(moduleType2);
  }

  @Test
  void stdlib_symbols() {
    var provider = new TypeShedModuleTypeProvider();
    var mathModule = provider.getModuleType("math", null);
    var acosType = mathModule.resolveMember("acos").get();

    Assertions.assertThat(acosType).isInstanceOf(FunctionType.class);

    var threadingModule = provider.getModuleType("threading", null);
    var threadType = threadingModule.resolveMember("Thread").get();
    Assertions.assertThat(threadType).isInstanceOf(ClassType.class);
    Assertions.assertThat(threadType.resolveMember("run").get()).isInstanceOf(FunctionType.class);

    var imaplibModule = provider.getModuleType("imaplib", null);
    var imapType = imaplibModule.resolveMember("IMAP4").get();
    Assertions.assertThat(imapType).isInstanceOf(ClassType.class);
    Assertions.assertThat(imapType.resolveMember("open").get()).isInstanceOf(FunctionType.class);
  }

  @Test
  void third_party_symbols() {
    var provider = new TypeShedModuleTypeProvider();
    var flaskHelpersModule = provider.getModuleType("flask.helpers", null);
    var flaskSymbol = flaskHelpersModule.resolveMember("get_root_path").get();
    assertThat(flaskSymbol).isInstanceOf(FunctionType.class);
  }

  @Test
  void should_resolve_packages() {
    var typeShed = new TypeShedModuleTypeProvider();
    assertThat(typeShed.getModuleType("urllib", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("ctypes", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("email", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("json", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("docutils", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("ctypes.util", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("lib2to3.pgen2.grammar", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("cryptography", null).members()).isNotEmpty();
    assertThat(typeShed.getModuleType("kazoo", null)).isNull();
  }
}
