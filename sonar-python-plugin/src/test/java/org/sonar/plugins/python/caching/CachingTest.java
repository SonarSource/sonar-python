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
package org.sonar.plugins.python.caching;


import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.Mockito;
import org.slf4j.event.Level;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.PythonReadCacheImpl;
import org.sonar.python.caching.PythonWriteCacheImpl;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorsToProtobuf;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.DescriptorsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.plugins.python.caching.Caching.IMPORTS_MAP_CACHE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;

class CachingTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private final static String CACHE_VERSION = "dummyVersion";


  @Test
  void writeProjectLevelSymbolTableEntry() throws InvalidProtocolBufferException {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    Set<Descriptor> initialDescriptors = Set.of(
      new ClassDescriptor("C", "mod.C", Collections.emptyList(), Collections.emptySet(), false, null, false, false, null, false),
      new FunctionDescriptor("foo", "mod.foo", Collections.emptyList(), false, false, Collections.emptyList(), false, null, null),
      new VariableDescriptor("x", "mod.x", null)
    );
    caching.writeProjectLevelSymbolTableEntry("mod", initialDescriptors);
    Map<String, byte[]> data = writeCache.getData();
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    Set<Descriptor> retrievedDescriptors = fromProtobuf(DescriptorsProtos.ModuleDescriptor.parseFrom(data.get(cacheKey)));
    assertThat(cacheContext.isCacheEnabled()).isTrue();
    assertThat(retrievedDescriptors).usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }


  @Test
  void readProjectLevelSymbolTableEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    Set<Descriptor> initialDescriptors = Set.of(
      new ClassDescriptor("C", "mod.C", Collections.emptyList(), Collections.emptySet(), false, null, false, false, null, false),
      new FunctionDescriptor("foo", "mod.foo", Collections.emptyList(), false, false, Collections.emptyList(), false, null, null),
      new VariableDescriptor("x", "mod.x", null)
    );
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, DescriptorsToProtobuf.toProtobufModuleDescriptor(initialDescriptors).toByteArray());
    Set<Descriptor> retrievedDescriptorsOptional = caching.readProjectLevelSymbolTableEntry("mod");
    assertThat(retrievedDescriptorsOptional).isNotNull().usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }

  @Test
  void readProjectLevelSymbolTableMissingEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), new PythonReadCacheImpl(readCache));

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    assertThat(caching.readProjectLevelSymbolTableEntry("unknown")).isNull();
  }

  @Test
  void readProjectLevelSymbolTableIOException() throws IOException {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    InputStream inputStream = mock(InputStream.class);
    when(inputStream.readAllBytes()).thenThrow(new IOException("Boom!"));
    PythonReadCacheImpl pythonReadCache = Mockito.spy(new PythonReadCacheImpl(readCache));
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, new byte[0]);
    Mockito.when(pythonReadCache.read(cacheKey)).thenReturn(inputStream);

    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), pythonReadCache);
    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    assertThat(caching.readProjectLevelSymbolTableEntry("mod")).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Unable to read data for key: \"python:descriptors:mod\"");
  }

  @Test
  void writeImportsMapEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(new TestReadCache());
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    Set<String> imports = Set.of("mod2", "pkg1.mod3", "pkg2.pkg3.mod4");

    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + "mod";
    caching.writeImportsMapEntry("mod", imports);
    Map<String, byte[]> data = writeCache.getData();
    Set<String> retrievedDescriptors = Arrays.stream(new String(data.get(cacheKey), StandardCharsets.UTF_8).split(";")).collect(Collectors.toSet());
    assertThat(retrievedDescriptors).containsExactlyInAnyOrderElementsOf(imports);
  }

  @Test
  void readImportsMapEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    writeCache.bind(readCache);
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    Set<String> imports = Set.of("mod2", "pkg1.mod3", "pkg2.pkg3.mod4");
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, String.join(";", imports).getBytes(StandardCharsets.UTF_8));
    assertThat(caching.readImportMapEntry("mod")).containsExactlyInAnyOrderElementsOf(imports);
  }

  @Test
  void readImportsMissingEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), new PythonReadCacheImpl(readCache));

    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    assertThat(caching.readImportMapEntry("unknown")).isNull();
  }

  @Test
  void corruptedDataInCache() {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);


    Caching caching = new Caching(cacheContext, CACHE_VERSION);
    String module = "mod";
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod", new byte[] {42});
    assertThat(caching.readProjectLevelSymbolTableEntry(module)).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Failed to deserialize project level symbol table entry for module: \"mod\"");
  }
}
