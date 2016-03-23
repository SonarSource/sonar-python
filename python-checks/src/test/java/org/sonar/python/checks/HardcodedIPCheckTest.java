/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class HardcodedIPCheckTest {

  @Test
  public void test() {
    HardcodedIPCheck check = new HardcodedIPCheck();
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/hardcodedIP.py"), check);
    String message = "Make this IP \"%s\" address configurable.";
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .next().atLine(3).withMessage(String.format(message, "127.0.0.1"))
      .next().atLine(5).withMessage(String.format(message, "123.1.1.1"))
      .next().atLine(7).withMessage(String.format(message, "255.1.1.1"))
      .next().atLine(19).withMessage(String.format(message, "2001:0db8:11a3:09d7:1f34:8a2e:07a0:765d"))
      .next().atLine(20).withMessage(String.format(message, "::1f34:8a2e:07a0:765d"))
      .next().atLine(21).withMessage(String.format(message, "1f34:8a2e:07a0:765d::"))
      .next().atLine(22).withMessage(String.format(message, "1f34:2e:7a0:765d::"))
      .next().atLine(25).withMessage(String.format(message, "1.2.3.4"))
      .noMore();
  }

}
