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
package org.sonar.plugins.python.api;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_310;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_311;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_312;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_313;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_36;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_37;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_39;

public class PythonVersionUtils {

  public enum Version {
    V_36(3, 6, "36"),
    V_37(3, 7, "37"),
    V_38(3, 8, "38"),
    V_39(3, 9, "39"),
    V_310(3, 10, "310"),
    V_311(3, 11, "311"),
    V_312(3, 12, "312"),
    V_313(3, 13, "313");

    private final int major;
    private final int minor;
    private final String serializedValue;

    Version(int major, int minor, String serializedValue) {
      this.major = major;
      this.minor = minor;
      this.serializedValue = serializedValue;
    }

    public int major() {
      return major;
    }

    public int minor() {
      return minor;
    }

    public String serializedValue() {
      return serializedValue;
    }

    public int compare(int major, int minor) {
      if (major() == major) {
        return Integer.compare(minor(), minor);
      }
      return Integer.compare(major(), major);
    }

    @Override
    public String toString() {
      return major + "." + minor;
    }
  }

  /**
   * Note that versions between 3 and 3.6 are currently mapped to 3.6 because
   * we don't take into account those version during typeshed symbols serialization
   */
  private static final Map<String, Version> STRING_VERSION_MAP = Map.ofEntries(
    Map.entry("3.0", V_36),
    Map.entry("3.1", V_36),
    Map.entry("3.2", V_36),
    Map.entry("3.3", V_36),
    Map.entry("3.4", V_36),
    Map.entry("3.5", V_36),
    Map.entry("3.6", V_36),
    Map.entry("3.7", V_37),
    Map.entry("3.8", V_38),
    Map.entry("3.9", V_39),
    Map.entry("3.10", V_310),
    Map.entry("3.11", V_311),
    Map.entry("3.12", V_312),
    Map.entry("3.13", V_313));
  private static final Version MIN_SUPPORTED_VERSION = V_36;
  public static final Version MAX_SUPPORTED_VERSION = V_313;
  private static final Logger LOG = LoggerFactory.getLogger(PythonVersionUtils.class);
  public static final String PYTHON_VERSION_KEY = "sonar.python.version";

  private PythonVersionUtils() {
  }

  public static Set<Version> fromString(String propertyValue) {
    return fromStringArray(propertyValue.split(","));
  }

  public static Set<Version> fromStringArray(String[] versions) {
    if (versions.length == 0) {
      return allVersions();
    }
    Set<Version> pythonVersions = EnumSet.noneOf(Version.class);
    for (String versionValue : versions) {
      versionValue = versionValue.trim();
      if ("3".equals(versionValue)) {
        // Only 3.x stubs are supported
        return allVersions();
      }
      Version version = STRING_VERSION_MAP.get(versionValue);
      if (version != null) {
        pythonVersions.add(version);
      } else {
        boolean isGuessSuccessful = guessPythonVersion(pythonVersions, versionValue);
        if (!isGuessSuccessful) {
          return allVersions();
        }
      }
    }
    return pythonVersions;
  }

  public static Set<Version> allVersions() {
    return EnumSet.allOf(Version.class);
  }

  private static boolean guessPythonVersion(Set<Version> pythonVersions, String versionValue) {
    String[] version = versionValue.split("\\.");
    try {
      int major = Integer.parseInt(version[0]);
      int minor = version.length > 1 ? Integer.parseInt(version[1]) : 0;
      Version guessedVersion = STRING_VERSION_MAP.get(major + "." + minor);
      if (guessedVersion != null) {
        pythonVersions.add(guessedVersion);
        logWarningGuessVersion(versionValue, guessedVersion);
        return true;
      }
      if (major < 3) {
        logWarningPython2(versionValue);
        return false;
      }
      if (MIN_SUPPORTED_VERSION.compare(major, minor) > 0) {
        pythonVersions.add(MIN_SUPPORTED_VERSION);
        logWarningGuessVersion(versionValue, MIN_SUPPORTED_VERSION);
      } else if (MAX_SUPPORTED_VERSION.compare(major, minor) < 0) {
        pythonVersions.add(MAX_SUPPORTED_VERSION);
        logWarningGuessVersion(versionValue, MAX_SUPPORTED_VERSION);
      } else {
        logErrorMessage(versionValue);
        return false;
      }
    } catch (NumberFormatException nfe) {
      logErrorMessage(versionValue);
      return false;
    }
    return true;
  }

  public static boolean areSourcePythonVersionsGreaterOrEqualThan(Set<Version> sourcePythonVersions, Version required) {
    // All versions must be greater than or equal to the required version.
    return !sourcePythonVersions.isEmpty() && sourcePythonVersions.stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }

  /**
   * @return the set of versions which are supported but not serialized due to SONARPY-1522
   */
  public static Set<Version> getNotSerializedVersions() {
    return EnumSet.of(V_312, V_313);
  }

  private static void logErrorMessage(String propertyValue) {
    LOG.warn(
      "Error while parsing value of parameter '{}' ({}). Versions must be specified as MAJOR_VERSION.MINOR_VERSION (e.g. \"3.7, 3.8\")",
      PYTHON_VERSION_KEY,
      propertyValue);
  }

  private static void logWarningGuessVersion(String propertyValue, Version guessedVersion) {
    LOG.warn("No explicit support for version {}. Python version has been set to {}.", propertyValue, guessedVersion);
  }

  private static void logWarningPython2(String propertyValue) {
    LOG.warn("No explicit support for version {}. Support for Python versions prior to 3 is deprecated.", propertyValue);
  }
}
