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
package org.sonar.python.types.json;

import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.types.PyTypeDetailedInfo;
import org.sonar.python.types.PyTypeDetailedInfoDeserializer;
import org.sonar.python.types.PyTypeInfo;
import org.sonar.python.types.pytype.BaseType;

class PolymorphDeserializerTest {

  @Test
  void test() throws FileNotFoundException {
    var gson = new GsonBuilder()
      .registerTypeAdapter(PyTypeDetailedInfo.class, new PyTypeDetailedInfoDeserializer())
      .registerTypeAdapter(BaseType.class, new PolymorphDeserializer<>())
      .create();
    var type = new TypeToken<Map<String, List<PyTypeInfo>>>() {}.getType();

    var types = gson.<Map<String, List<PyTypeInfo>>>fromJson(new FileReader("../python-checks/src/test/resources/checks.json"), type);

    Assertions.assertNotNull(types);
  }

}
