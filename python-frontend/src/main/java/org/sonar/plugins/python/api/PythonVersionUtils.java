/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_310;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_311;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_312;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_313;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_314;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersionUtils.Version.V_39;

public class PythonVersionUtils {

  /**
   * Normalized Python source-version compatibility buckets derived from project configuration and exposed to
   * version-sensitive rules. Configured versions older than 3.8 are represented by {@link #V_38}.
   * These values do not identify the serialized semantic models used during analysis.
   */
  public enum Version {
    V_38(3, 8),
    V_39(3, 9),
    V_310(3, 10),
    V_311(3, 11),
    V_312(3, 12),
    V_313(3, 13),
    V_314(3, 14);

    private final int major;
    private final int minor;

    Version(int major, int minor) {
      this.major = major;
      this.minor = minor;
    }

    public int major() {
      return major;
    }

    public int minor() {
      return minor;
    }

    /**
     * @deprecated Source versions no longer have a one-to-one correspondence with serialized semantic models.
     * Use {@link SemanticVersion#serializedValue()} after calling
     * {@link PythonVersionUtils#toSupportedSemanticVersions(Set)} when selecting semantic data.
     */
    @Deprecated(since = "5.31")
    public String serializedValue() {
      return Integer.toString(major) + minor;
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
   * Python versions for which serialized semantic-model data is available.
   * These values are internal analysis targets and are distinct from configured source-version compatibility.
   * Values must be declared from oldest to newest so that the supported range can be derived from their order.
   */
  public enum SemanticVersion {
    V_310(3, 10),
    V_311(3, 11),
    V_312(3, 12),
    V_313(3, 13),
    V_314(3, 14);

    private final int major;
    private final int minor;

    SemanticVersion(int major, int minor) {
      this.major = major;
      this.minor = minor;
    }

    /**
     * Returns the version identifier stored in serialized semantic-model data.
     */
    public String serializedValue() {
      return Integer.toString(major) + minor;
    }

    @Override
    public String toString() {
      return major + "." + minor;
    }
  }

  private static final Version MIN_RECOGNIZED_SOURCE_VERSION = V_38;
  private static final Version MIN_SUPPORTED_SOURCE_VERSION = V_310;
  public static final Version MAX_SUPPORTED_VERSION = V_314;

  private static final Map<String, Version> STRING_VERSION_MAP = Map.ofEntries(
    Map.entry("3.0", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.1", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.2", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.3", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.4", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.5", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.6", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.7", MIN_RECOGNIZED_SOURCE_VERSION),
    Map.entry("3.8", V_38),
    Map.entry("3.9", V_39),
    Map.entry("3.10", V_310),
    Map.entry("3.11", V_311),
    Map.entry("3.12", V_312),
    Map.entry("3.13", V_313),
    Map.entry("3.14", V_314));
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
    if (PythonVersionSpecifierParser.containsVersionSpecifier(versions)) {
      return fromVersionSpecifiers(versions);
    }
    return fromExactVersions(versions);
  }

  private static Set<Version> fromVersionSpecifiers(String[] versions) {
    String propertyValue = String.join(",", Arrays.stream(versions).map(String::trim).toArray(String[]::new));
    Optional<Set<Version>> parsedVersions = PythonVersionSpecifierParser.parse(versions);
    if (parsedVersions.isEmpty()) {
      logErrorMessage(propertyValue);
      return allVersions();
    }
    Set<Version> matchingVersions = parsedVersions.get();
    if (matchingVersions.isEmpty()) {
      logWarningNoMatchingRange(propertyValue);
      return allVersions();
    }
    return matchingVersions;
  }

  private static Set<Version> fromExactVersions(String[] versions) {
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

  /**
   * Returns the source versions targeted by default when no project version is configured.
   */
  public static Set<Version> allVersions() {
    return EnumSet.range(MIN_SUPPORTED_SOURCE_VERSION, MAX_SUPPORTED_VERSION);
  }

  /**
   * Maps declared source versions to versions for which serialized semantic data is available.
   * Source versions older than 3.10 use the Python 3.10 semantic model.
   */
  public static Set<SemanticVersion> toSupportedSemanticVersions(Set<Version> sourcePythonVersions) {
    Set<SemanticVersion> semanticVersions = EnumSet.noneOf(SemanticVersion.class);
    sourcePythonVersions.stream()
      .map(PythonVersionUtils::toSupportedSemanticVersion)
      .forEach(semanticVersions::add);
    return semanticVersions;
  }

  /**
   * Returns all versions for which serialized semantic-model data is available.
   */
  public static Set<SemanticVersion> allSupportedSemanticVersions() {
    return EnumSet.allOf(SemanticVersion.class);
  }

  /**
   * Returns the oldest version for which serialized semantic-model data is available.
   */
  public static SemanticVersion minSupportedSemanticVersion() {
    return SemanticVersion.values()[0];
  }

  /**
   * Returns the newest version for which serialized semantic-model data is available.
   */
  public static SemanticVersion maxSupportedSemanticVersion() {
    SemanticVersion[] versions = SemanticVersion.values();
    return versions[versions.length - 1];
  }

  private static SemanticVersion toSupportedSemanticVersion(Version sourcePythonVersion) {
    return switch (sourcePythonVersion) {
      case V_38, V_39, V_310 -> minSupportedSemanticVersion();
      case V_311 -> SemanticVersion.V_311;
      case V_312 -> SemanticVersion.V_312;
      case V_313 -> SemanticVersion.V_313;
      case V_314 -> SemanticVersion.V_314;
    };
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
        pythonVersions.add(MIN_RECOGNIZED_SOURCE_VERSION);
        logWarningPython2(versionValue);
        return true;
      }
      if (MIN_RECOGNIZED_SOURCE_VERSION.compare(major, minor) > 0) {
        pythonVersions.add(MIN_RECOGNIZED_SOURCE_VERSION);
        logWarningGuessVersion(versionValue, MIN_RECOGNIZED_SOURCE_VERSION);
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


  private static void logErrorMessage(String propertyValue) {
    LOG.warn(
      "Error while parsing value of parameter '{}' ({}). Use comma-separated Python versions (e.g. \"3.10,3.11\") or numeric version specifiers (e.g. \">=3.10,<3.13\").",
      PYTHON_VERSION_KEY,
      propertyValue);
  }

  private static void logWarningGuessVersion(String propertyValue, Version guessedVersion) {
    LOG.warn("No explicit support for version {}. Python version has been set to {}.", propertyValue, guessedVersion);
  }

  private static void logWarningPython2(String propertyValue) {
    LOG.warn("No explicit support for version {}. Support for Python versions prior to 3 is deprecated.", propertyValue);
  }

  private static void logWarningNoMatchingRange(String propertyValue) {
    LOG.warn("No supported Python version matches version range {}. Analysis will target all supported Python versions.", propertyValue);
  }
}
