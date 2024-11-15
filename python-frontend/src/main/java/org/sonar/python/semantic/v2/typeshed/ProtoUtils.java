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

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;

public class ProtoUtils {
  private ProtoUtils() {
    // utility class
  }

  static Map<String, Descriptor> disambiguateByName(Stream<Stream<? extends Descriptor>> input) {
    return input.flatMap(i -> i)
      .collect(Collectors.groupingBy(
        Descriptor::name, Collectors.collectingAndThen(
          Collectors.toSet(),
          ProtoUtils::disambiguateSymbolsWithSameName)));
  }

  private static Descriptor disambiguateSymbolsWithSameName(Set<Descriptor> descriptors) {
    if (descriptors.size() > 1) {
      return AmbiguousDescriptor.create(descriptors);
    }
    return descriptors.iterator().next();
  }

  static boolean isValidForPythonVersion(List<String> validForPythonVersions, Set<String> supportedPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    // TODO: SONARPY-1522 - remove this workaround when we will have all the stubs for Python 3.12.
    Set<String> notSerializedVersions =
      PythonVersionUtils.getNotSerializedVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
    if (notSerializedVersions.containsAll(supportedPythonVersions)
        && validForPythonVersions.contains(PythonVersionUtils.Version.V_311.serializedValue())) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions);
    return !intersection.isEmpty();
  }
}
