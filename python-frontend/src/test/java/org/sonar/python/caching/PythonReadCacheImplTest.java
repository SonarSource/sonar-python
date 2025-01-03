/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.caching;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.batch.sensor.cache.ReadCache;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PythonReadCacheImplTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void read_bytes() {
    byte[] bytes = "hello".getBytes();
    ByteArrayInputStream inputStream = new ByteArrayInputStream(bytes);

    ReadCache readCache = mock(ReadCache.class);
    when(readCache.read("key")).thenReturn(inputStream);
    when(readCache.contains("key")).thenReturn(true);

    PythonReadCacheImpl pythonReadCache = new PythonReadCacheImpl(readCache);
    byte[] result = pythonReadCache.readBytes("key");

    assertThat(result).isEqualTo(bytes);
  }

  @Test
  void read_bytes_no_such_key() {
    ReadCache readCache = mock(ReadCache.class);
    when(readCache.contains("key")).thenReturn(false);

    PythonReadCacheImpl pythonReadCache = new PythonReadCacheImpl(readCache);
    byte[] result = pythonReadCache.readBytes("key");

    assertThat(result).isNull();
  }

  @Test
  void read_bytes_io_exception() throws IOException {
    InputStream inputStream = mock(InputStream.class);
    when(inputStream.readAllBytes()).thenThrow(IOException.class);

    ReadCache readCache = mock(ReadCache.class);
    when(readCache.read("key")).thenReturn(inputStream);
    when(readCache.contains("key")).thenReturn(true);

    PythonReadCacheImpl pythonReadCache = new PythonReadCacheImpl(readCache);
    byte[] result = pythonReadCache.readBytes("key");

    assertThat(result)
      .isNull();

    assertThat(logTester.logs(Level.DEBUG))
      .containsExactly("Unable to read data for key: \"key\"");
  }

  @Test
  void contains() {
    ReadCache readCache = mock(ReadCache.class);
    when(readCache.contains("exists")).thenReturn(true);
    when(readCache.contains("doesNotExist")).thenReturn(false);

    PythonReadCacheImpl pythonReadCache = new PythonReadCacheImpl(readCache);

    assertThat(pythonReadCache.contains("exists")).isTrue();
    assertThat(pythonReadCache.contains("doesNotExists")).isFalse();
  }
}
