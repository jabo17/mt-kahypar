set(LPSources
        label_propagation/label_propagation_refiner.cpp
        rebalancing/rebalancer.cpp
        deterministic/deterministic_label_propagation.cpp
        )

set(FMSources
        fm/multitry_kway_fm.cpp
        fm/localized_kway_fm_core.cpp
        fm/global_rollback.cpp
        fm/sequential_twoway_fm_refiner.cpp
        )

set(FlowSources
        flows/refiner_adapter.cpp
        flows/problem_construction.cpp
        flows/scheduler.cpp
        flows/quotient_graph.cpp
        flows/flow_refiner.cpp
        flows/sequential_construction.cpp
        flows/parallel_construction.cpp
        flows/flow_hypergraph_builder.cpp
        )

set(FlowSources
    flows/refiner_adapter.cpp
    flows/problem_construction.cpp
    flows/scheduler.cpp
    flows/quotient_graph.cpp
    flows/flow_refiner.cpp
    flows/sequential_construction.cpp
    flows/parallel_construction.cpp
    flows/flow_hypergraph_builder.cpp
    )

set(RegistrySources
    register_refinement_algorithms.cpp
    )

add_library(MtKaHyParLPDefault STATIC ${LPSources})
add_library(MtKaHyParFMDefault STATIC ${FMSources})
add_library(MtKaHyParFlowDefault STATIC ${FlowSources})
add_library(MtKaHyParRegistryDefault STATIC ${RegistrySources})

add_library(MtKaHyParLPQuality STATIC ${LPSources})
add_library(MtKaHyParFMQuality STATIC ${FMSources})
add_library(MtKaHyParFlowQuality STATIC ${FlowSources})
add_library(MtKaHyParRegistryQuality STATIC ${RegistrySources})
target_compile_definitions(MtKaHyParLPQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParFMQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParFlowQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParRegistryQuality PUBLIC USE_STRONG_PARTITIONER)

add_library(MtKaHyParLPGraph STATIC ${LPSources})
add_library(MtKaHyParFMGraph STATIC ${FMSources})
add_library(MtKaHyParFlowGraph STATIC ${FlowSources})
add_library(MtKaHyParRegistryGraph STATIC ${RegistrySources})
target_compile_definitions(MtKaHyParLPGraph PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParFMGraph PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParFlowGraph PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParRegistryGraph PUBLIC USE_GRAPH_PARTITIONER)

add_library(MtKaHyParLPGraphQuality STATIC ${LPSources})
add_library(MtKaHyParFMGraphQuality STATIC ${FMSources})
add_library(MtKaHyParFlowGraphQuality STATIC ${FlowSources})
add_library(MtKaHyParRegistryGraphQuality STATIC ${RegistrySources})
target_compile_definitions(MtKaHyParLPGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParLPGraphQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParFMGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParFMGraphQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParFlowGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParFlowGraphQuality PUBLIC USE_STRONG_PARTITIONER)
target_compile_definitions(MtKaHyParRegistryGraphQuality PUBLIC USE_GRAPH_PARTITIONER)
target_compile_definitions(MtKaHyParRegistryGraphQuality PUBLIC USE_STRONG_PARTITIONER)

list(APPEND REFINEMENT_LIBRARIES
      MtKaHyParLPDefault
      MtKaHyParFMDefault
      MtKaHyParFlowDefault
      MtKaHyParRegistryDefault
      MtKaHyParLPQuality
      MtKaHyParFMQuality
      MtKaHyParFlowQuality
      MtKaHyParRegistryQuality
      MtKaHyParLPGraph
      MtKaHyParFMGraph
      MtKaHyParFlowGraph
      MtKaHyParRegistryGraph
      MtKaHyParLPGraphQuality
      MtKaHyParFMGraphQuality
      MtKaHyParFlowGraphQuality
      MtKaHyParRegistryGraphQuality)

foreach(refinement_lib ${REFINEMENT_LIBRARIES})
  set_property(TARGET ${refinement_lib} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${refinement_lib} PROPERTY CXX_STANDARD_REQUIRED ON)
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
      target_link_libraries(${modtarget} MtKaHyParLPDefault MtKaHyParFMDefault MtKaHyParFlowDefault MtKaHyParRegistryDefault)
    elseif( GRAPH AND NOT STRONG )
      target_link_libraries(${modtarget} MtKaHyParLPGraph MtKaHyParFMGraph MtKaHyParFlowGraph MtKaHyParRegistryGraph)
    elseif( NOT GRAPH AND STRONG )
      target_link_libraries(${modtarget} MtKaHyParLPQuality MtKaHyParFMQuality MtKaHyParFlowQuality MtKaHyParRegistryQuality)
    elseif( GRAPH AND STRONG )
      target_link_libraries(${modtarget} MtKaHyParLPGraphQuality MtKaHyParFMGraphQuality MtKaHyParFlowGraphQuality MtKaHyParRegistryGraphQuality)
    endif()
endforeach()