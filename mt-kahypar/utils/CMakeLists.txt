set(UtilSources
        memory_tree.cpp
        )

add_library(MtKaHyParUtils STATIC ${UtilSources})
set_property(TARGET MtKaHyParUtils PROPERTY CXX_STANDARD 17)
set_property(TARGET MtKaHyParUtils PROPERTY CXX_STANDARD_REQUIRED ON)

foreach(modtarget IN LISTS TARGETS_WANTING_ALL_SOURCES)
  target_link_libraries(${modtarget} MtKaHyParUtils)
endforeach()