set(CoarseningSources
        deterministic_multilevel_coarsener.cpp
        multilevel_uncoarsener.cpp
        nlevel_uncoarsener.cpp
        )

set(RegistrySources
        register_coarsening_algorithms.cpp
        )

add_library(MtKaHyParCoarseningDefault STATIC ${CoarseningSources})

add_library(MtKaHyParCoarseningQuality STATIC ${CoarseningSources})
target_compile_definitions(MtKaHyParCoarseningQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParCoarseningGraph STATIC ${CoarseningSources})
target_compile_definitions(MtKaHyParCoarseningGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParCoarseningGraphQuality STATIC ${CoarseningSources})
target_compile_definitions(MtKaHyParCoarseningGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParCoarseningGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND COARSENING_LIBRARIES
      MtKaHyParCoarseningDefault
      MtKaHyParCoarseningQuality
      MtKaHyParCoarseningGraph
      MtKaHyParCoarseningGraphQuality)

foreach(coarsening_lib ${COARSENING_LIBRARIES})
  set_property(TARGET ${coarsening_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${coarsening_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
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
    target_link_libraries(${modtarget} MtKaHyParCoarseningDefault)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParCoarseningGraph)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParCoarseningQuality)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParCoarseningGraphQuality)
  endif()
endforeach()
