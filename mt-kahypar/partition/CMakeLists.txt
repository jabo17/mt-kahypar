add_subdirectory(preprocessing/)
add_subdirectory(coarsening)
add_subdirectory(refinement)
add_subdirectory(initial_partitioning)

set(PartitionSources
        partitioner.cpp
        multilevel.cpp
        metrics.cpp
        recursive_bipartitioning.cpp
        deep_multilevel.cpp
        )

set(ContextSources
      context.cpp
      context_enum_classes.cpp)

add_library(MtKaHyParContext STATIC ${ContextSources})

add_library(MtKaHyParPartitionDefault STATIC ${PartitionSources})

add_library(MtKaHyParPartitionQuality STATIC ${PartitionSources})
target_compile_definitions(MtKaHyParPartitionQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParPartitionGraph STATIC ${PartitionSources})
target_compile_definitions(MtKaHyParPartitionGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParPartitionGraphQuality STATIC ${PartitionSources})
target_compile_definitions(MtKaHyParPartitionGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParPartitionGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND PARTITION_LIBRARIES
      MtKaHyParContext
      MtKaHyParPartitionDefault
      MtKaHyParPartitionQuality
      MtKaHyParPartitionGraph
      MtKaHyParPartitionGraphQuality)

foreach(partitioner_lib ${PARTITION_LIBRARIES})
  set_property(TARGET ${partitioner_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${partitioner_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
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
    target_link_libraries(${modtarget} MtKaHyParPartitionDefault MtKaHyParContext)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParPartitionGraph MtKaHyParContext)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParPartitionQuality MtKaHyParContext)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParPartitionGraphQuality MtKaHyParContext)
  endif()
endforeach()