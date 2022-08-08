set(PreprocessingSources
        community_detection/parallel_louvain.cpp
        community_detection/local_moving_modularity.cpp
        ${PROJECT_SOURCE_DIR}/mt-kahypar/datastructures/graph.cpp)

set(RegistrySources
        register_preprocessing_algorithms.cpp
        )

add_library(MtKaHyParPreprocessingDefault STATIC ${PreprocessingSources})

add_library(MtKaHyParPreprocessingQuality STATIC ${PreprocessingSources})
target_compile_definitions(MtKaHyParPreprocessingQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParPreprocessingGraph STATIC ${PreprocessingSources})
target_compile_definitions(MtKaHyParPreprocessingGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParPreprocessingGraphQuality STATIC ${PreprocessingSources})
target_compile_definitions(MtKaHyParPreprocessingGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParPreprocessingGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND PREPROCESSING_LIBRARIES
      MtKaHyParPreprocessingDefault
      MtKaHyParPreprocessingQuality
      MtKaHyParPreprocessingGraph
      MtKaHyParPreprocessingGraphQuality)

foreach(preprocessing_lib ${PREPROCESSING_LIBRARIES})
  set_property(TARGET ${preprocessing_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${preprocessing_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
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
    target_link_libraries(${modtarget} MtKaHyParPreprocessingDefault)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParPreprocessingGraph)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParPreprocessingQuality)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParPreprocessingGraphQuality)
  endif()
endforeach()