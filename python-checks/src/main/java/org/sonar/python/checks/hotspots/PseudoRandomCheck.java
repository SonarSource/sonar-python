/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.checks.AbstractCallExpressionCheck;

@Rule(key = "S2245")
public class PseudoRandomCheck extends AbstractCallExpressionCheck {

  private static final Set<String> FUNCTIONS_TO_CHECK = new HashSet<>(Arrays.asList(
    "random.random",
    "random.getrandbits",
    "random.randint",
    "random.sample",
    "random.choice",
    "random.choices"));

  @Override
  protected Set<String> functionsToCheck() {
    return FUNCTIONS_TO_CHECK;
  }

  @Override
  protected String message() {
    return "Make sure that using this pseudorandom number generator is safe here.";
  }

}
