set(IOSources
        csv_output.cpp
        hypergraph_io.cpp
        sql_plottools_serializer.cpp
        partitioning_output.cpp
        command_line_options.cpp)

add_library(MtKaHyParIODefault STATIC ${IOSources})

add_library(MtKaHyParIOQuality STATIC ${IOSources})
target_compile_definitions(MtKaHyParIOQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParIOGraph STATIC ${IOSources})
target_compile_definitions(MtKaHyParIOGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParIOGraphQuality STATIC ${IOSources})
target_compile_definitions(MtKaHyParIOGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParIOGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND IO_LIBRARIES
      MtKaHyParIODefault
      MtKaHyParIOQuality
      MtKaHyParIOGraph
      MtKaHyParIOGraphQuality)

foreach(io_lib ${IO_LIBRARIES})
  target_link_libraries(${io_lib} ${Boost_LIBRARIES})
  set_property(TARGET ${io_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${io_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
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
    target_link_libraries(${modtarget} MtKaHyParIODefault)
  elseif( GRAPH AND NOT STRONG )
    target_link_libraries(${modtarget} MtKaHyParIOGraph)
  elseif( NOT GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParIOQuality)
  elseif( GRAPH AND STRONG )
    target_link_libraries(${modtarget} MtKaHyParIOGraphQuality)
  endif()
endforeach()