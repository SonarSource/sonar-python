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
package org.sonar.python.checks;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.ClassOrderer;
import org.junit.jupiter.api.ClassOrdererContext;

public class ChecksClassTestOrder implements ClassOrderer {
  private List<String> classOrder = new ArrayList<>();

  ChecksClassTestOrder() {
    try(BufferedReader classOrderStream = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream("/classOrder.txt")))) {
      classOrder.addAll(classOrderStream.lines().toList());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void orderClasses(ClassOrdererContext context) {
    context.getClassDescriptors().sort((a, b) -> {
      int aIndex = classOrder.indexOf(a.getTestClass().getName());
      int bIndex = classOrder.indexOf(b.getTestClass().getName());
      if (aIndex == -1) {
        return 1;
      }
      if (bIndex == -1) {
        return -1;
      }
      return Integer.compare(aIndex, bIndex);
    });
  }
}
