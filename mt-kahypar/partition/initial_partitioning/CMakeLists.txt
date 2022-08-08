set(InitialPartitioningSources
        flat/pool_initial_partitioner.cpp
        flat/random_initial_partitioner.cpp
        flat/bfs_initial_partitioner.cpp
        flat/label_propagation_initial_partitioner.cpp
        recursive_bipartitioning_initial_partitioner.cpp
        deep_initial_partitioner.cpp
        )

set(RegistrySources
        register_initial_partitioning_algorithms.cpp
        )

add_library(MtKaHyParIPDefault STATIC ${InitialPartitioningSources})

add_library(MtKaHyParIPQuality STATIC ${InitialPartitioningSources})
target_compile_definitions(MtKaHyParIPQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParIPGraph STATIC ${InitialPartitioningSources})
target_compile_definitions(MtKaHyParIPGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParIPGraphQuality STATIC ${InitialPartitioningSources})
target_compile_definitions(MtKaHyParIPGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParIPGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND IP_LIBRARIES
      MtKaHyParIPDefault
      MtKaHyParIPQuality
      MtKaHyParIPGraph
      MtKaHyParIPGraphQuality)

foreach(ip_lib ${IP_LIBRARIES})
  set_property(TARGET ${ip_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${ip_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
endforeach()

foreach(modtarget IN LISTS TARGETS_WANTING_ALL_SOURCES)
  target_sources(${modtarget} PRIVATE ${RegistrySources})
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
    target_link_libraries(${modtarget} MtKaHyParIPDefault)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParIPGraph)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParIPQuality)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParIPGraphQuality)
  endif()
endforeach()