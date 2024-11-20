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
package org.sonar.plugins.python.xunit;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertEquals;

class TestCaseTest {
  @Test
  void rendersRightDetails() {
    Map<String, TestCase> ioMap = new HashMap<>();

    ioMap.put("<testcase status=\"ok\" time=\"1\" name=\"name\"/>",
              new TestCase("name", 1, "ok", "", "", null, null));
    ioMap.put("<testcase status=\"error\" time=\"1\" name=\"name\"><error message=\"errmsg\"><![CDATA[stack]]></error></testcase>",
              new TestCase("name", 1, "error", "stack", "errmsg", null, null));
    ioMap.put("<testcase status=\"failure\" time=\"1\" name=\"name\"><failure message=\"errmsg\"><![CDATA[stack]]></failure></testcase>",
              new TestCase("name", 1, "failure", "stack", "errmsg","file", "testClassname"));

    for(Map.Entry<String, TestCase> entry: ioMap.entrySet()) {
      assertEquals(entry.getKey(), entry.getValue().getDetails());
    }
  }
}
