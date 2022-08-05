set(MemoryPoolSources
        register_memory_pool.cpp
        )

add_library(MtKaHyParMemoryPoolDefault STATIC ${MemoryPoolSources})

add_library(MtKaHyParMemoryPoolQuality STATIC ${MemoryPoolSources})
target_compile_definitions(MtKaHyParMemoryPoolQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParMemoryPoolGraph STATIC ${MemoryPoolSources})
target_compile_definitions(MtKaHyParMemoryPoolGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParMemoryPoolGraphQuality STATIC ${MemoryPoolSources})
target_compile_definitions(MtKaHyParMemoryPoolGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParMemoryPoolGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND MEMORY_POOL_LIBRARIES
      MtKaHyParMemoryPoolDefault
      MtKaHyParMemoryPoolQuality
      MtKaHyParMemoryPoolGraph
      MtKaHyParMemoryPoolGraphQuality)

foreach(mem_pool_lib ${MEMORY_POOL_LIBRARIES})
  set_property(TARGET ${mem_pool_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${mem_pool_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
endforeach()

foreach(modtarget IN LISTS TARGETS_WANTING_ALL_SOURCES)
  get_target_property(compile_defs ${modtarget} COMPILE_DEFINITIONS)
  SET(STRONG FALSE)
  SET(GRAPH FALSE)
  foreach( property ${compile_defs} )
    if(${property} MATCHES ".*STRONG.*")
      SET(STRONG TRUE)
    elseif(${property} MATCHES ".*GRAPH.*")
      SET(GRAPH TRUE)
    endif()
  endforeach()

  if ( NOT GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParMemoryPoolDefault)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParMemoryPoolGraph)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParMemoryPoolQuality)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParMemoryPoolGraphQuality)
  endif()
endforeach()