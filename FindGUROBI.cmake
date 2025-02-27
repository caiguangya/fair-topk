find_path(GUROBI_HEADER_DIR
    NAMES gurobi_c.h
    PATH_SUFFIXES gurobi)

find_library(GUROBI_LIBRARY
    NAMES gurobi100 gurobi110 gurobi120
    PATH_SUFFIXES gurobi)

if(CMAKE_CXX_COMPILER_LOADED)
    find_path(GUROBI_CXX_HEADER_DIR
        NAMES gurobi_c++.h
        PATH_SUFFIXES gurobi)
    
    find_library(GUROBI_CXX_WRAPPER_LIBRARY
        NAMES gurobi_c++
        PATH_SUFFIXES gurobi)

    if (GUROBI_CXX_HEADER_DIR AND GUROBI_CXX_WRAPPER_LIBRARY)
        set(GUROBI_CXX_FOUND TRUE)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI REQUIRED_VARS GUROBI_LIBRARY GUROBI_HEADER_DIR HANDLE_COMPONENTS)

if(GUROBI_FOUND AND NOT TARGET GUROBI)
    if(CMAKE_VERSION VERSION_GREATER 3.20.0)
        cmake_path(GET GUROBI_HEADER_DIR PARENT_PATH GUROBI_INCLUDE_DIR)
    else()
        get_filename_component(GUROBI_INCLUDE_DIR ${GUROBI_HEADER_DIR} DIRECTORY)
    endif()
    add_library(GUROBI::SOLVER SHARED IMPORTED)
    set_target_properties(GUROBI::SOLVER PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${GUROBI_INCLUDE_DIR}
        IMPORTED_LOCATION ${GUROBI_LIBRARY})
    
    if(GUROBI_CXX_FOUND)
        if(CMAKE_VERSION VERSION_GREATER 3.20.0)
            cmake_path(GET GUROBI_CXX_HEADER_DIR PARENT_PATH GUROBI_CXX_INCLUDE_DIR)
        else()
            get_filename_component(GUROBI_CXX_INCLUDE_DIR ${GUROBI_CXX_HEADER_DIR} DIRECTORY)
        endif()
        add_library(GUROBI::CXX STATIC IMPORTED)
        set_target_properties(GUROBI::CXX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${GUROBI_CXX_INCLUDE_DIR}
            IMPORTED_LOCATION ${GUROBI_CXX_WRAPPER_LIBRARY})
    endif()

    if(GUROBI_CXX_FOUND)
        add_library(GUROBI INTERFACE IMPORTED)
        set_property(TARGET GUROBI PROPERTY INTERFACE_LINK_LIBRARIES GUROBI::CXX GUROBI::SOLVER)
    else()
        add_library(GUROBI ALIAS GUROBI::SOLVER)
    endif()
endif()