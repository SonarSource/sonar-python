/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.caching;

import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.DescriptorsProtos;

import static org.sonar.python.index.DescriptorsToProtobuf.toProtobuf;

public class Caching {

  private final CacheContext cacheContext;

  public static final String IMPORTS_MAP_CACHE_KEY_PREFIX = "python:imports:";
  public static final String PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX = "python:descriptors:";

  private static final Logger LOG = Loggers.get(Caching.class);

  public Caching(CacheContext cacheContext) {
    this.cacheContext = cacheContext;
  }

  public void writeImportsMapEntry(String moduleFqn, Set<String> imports) {
    byte[] importData = String.join(";", imports).getBytes(StandardCharsets.UTF_8);
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, importData);
  }

  public void writeProjectLevelSymbolTableEntry(String moduleFqn, Set<Descriptor> descriptors) {
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, moduleDescriptor(descriptors).toByteArray());
  }

  @CheckForNull
  public Set<Descriptor> readProjectLevelSymbolTableEntry(String module) {
    String key = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + module;
    if (cacheContext.getReadCache().contains(key)) {
      byte[] bytes = cacheContext.getReadCache().readBytes(key);
      if (bytes != null) {
        try {
          return DescriptorUtils.deserializeProtobufDescriptors(bytes);
        } catch (InvalidProtocolBufferException e) {
          LOG.debug("Failed to deserialize project level symbol table entry for module: \"{}\"", module);
        }
      }
    }
    return null;
  }

  @CheckForNull
  public Set<String> readImportMapEntry(String moduleFqn) {
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + moduleFqn;
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes != null) {
      return new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    }
    return null;
  }


  // Visible for testing
  public static DescriptorsProtos.ModuleDescriptor moduleDescriptor(Set<Descriptor> descriptors) {
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> varDescriptors = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousDescriptors = new ArrayList<>();
    for (Descriptor descriptor : descriptors) {
      Descriptor.Kind kind = descriptor.kind();
      if (kind == Descriptor.Kind.CLASS) {
        classDescriptors.add(toProtobuf(((ClassDescriptor) descriptor)));
      } else if (kind == Descriptor.Kind.FUNCTION) {
        functionDescriptors.add(toProtobuf((FunctionDescriptor) descriptor));
      } else if (kind == Descriptor.Kind.VARIABLE) {
        varDescriptors.add(toProtobuf((VariableDescriptor) descriptor));
      } else {
        ambiguousDescriptors.add(toProtobuf((AmbiguousDescriptor) descriptor));
      }
    }
    return DescriptorsProtos.ModuleDescriptor.newBuilder()
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(varDescriptors)
      .addAllAmbiguousDescriptors(ambiguousDescriptors)
      .build();
  }
}
