get_filename_component(_TEST_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

add_executable(${_TEST_NAME}_tests
    ${_TEST_NAME}_tests.cpp
    test_3mf.cpp
    test_aabbindirect.cpp
    test_appconfig.cpp
    test_arachne_walls.cpp
    test_arrange.cpp
    test_bambu_networking.cpp
    test_calib.cpp
    test_clipper_offset.cpp
    test_clipper_utils.cpp
    test_config.cpp
    test_config_variant_expansion.cpp
    test_toolordering_nozzle_group.cpp
    test_preset_bundle_loading.cpp
    test_preset_setting_id.cpp
    test_preset_diff.cpp
    test_elephant_foot_compensation.cpp
    test_fill_corner_smoothing.cpp
    test_fill_plane_path.cpp
    test_geometry.cpp
    test_multimaterial_segmentation.cpp
    test_placeholder_parser.cpp
    test_polygon.cpp
    test_mutable_polygon.cpp
    test_mutable_priority_queue.cpp
    test_nozzle_volume_type.cpp
    test_stl.cpp
    test_meshboolean.cpp
    test_marchingsquares.cpp
    test_model.cpp
    test_utils.cpp
    test_timeutils.cpp
    test_voronoi.cpp
    test_optimizers.cpp
    test_ordering_strategies.cpp
    # test_png_io.cpp
    test_indexed_triangle_set.cpp
    ../libnest2d/printer_parts.cpp
    )

if (TARGET OpenVDB::openvdb)
    target_sources(${_TEST_NAME}_tests PRIVATE test_hollowing.cpp)
endif()

target_link_libraries(${_TEST_NAME}_tests test_common libslic3r Catch2::Catch2WithMain)
target_sources(${_TEST_NAME}_tests PRIVATE test_usd.cpp)
target_link_libraries(${_TEST_NAME}_tests usd_whole_archive)

# The CI test job unpacks build/**/tests and nothing else, so deps/ is not
# there; stage the tree inside what gets packed.
add_custom_command(TARGET ${_TEST_NAME}_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ORCA_USD_PLUGIN_DIR}" "$<TARGET_FILE_DIR:${_TEST_NAME}_tests>/usd"
    COMMENT "Staging OpenUSD plugInfo tree beside the test binary")
target_include_directories(${_TEST_NAME}_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_property(TARGET ${_TEST_NAME}_tests PROPERTY FOLDER "tests")

# Optional, dev-only: a user OSL shader driving PrintMan displacement through slice_scene end to end.
# OFF by default so the normal build and CI never need OSL/LLVM. When ON, it links a from-source
# OpenShadingLanguage (see spikes/osl in the printman repo) into this test binary only -- the engine
# and libslic3r are untouched, because slice_scene already takes a DisplacementField. Paths default
# to the local build tree and Homebrew and are overridable.
option(SLIC3R_OSL "Build the local OSL displacement end-to-end test (needs a from-source OSL)" OFF)
if (SLIC3R_OSL)
    set(OSL_ROOT "$ENV{HOME}/tmp/osl/OpenShadingLanguage" CACHE PATH "From-source OpenShadingLanguage tree")
    set(OSL_BREW "/opt/homebrew" CACHE PATH "Homebrew prefix providing OpenImageIO/Imath/fmt")
    find_library(OSL_EXEC_LIB oslexec       PATHS "${OSL_ROOT}/build/lib" NO_DEFAULT_PATH REQUIRED)
    find_library(OSL_OIIO_LIB OpenImageIO   PATHS "${OSL_BREW}/lib" REQUIRED)
    find_library(OSL_OIIOU_LIB OpenImageIO_Util PATHS "${OSL_BREW}/lib" REQUIRED)
    find_program(OSLC_BIN oslc              PATHS "${OSL_ROOT}/build/bin" NO_DEFAULT_PATH REQUIRED)

    # Compile each test shader to bytecode beside the test binary's build dir; the test reads them there.
    set(_osl_oso_dir "${CMAKE_CURRENT_BINARY_DIR}/osl_shaders")
    set(_osl_shaders printman_test_disp printman_test_normal printman_test_hifreq printman_grid deriv_probe)
    set(_osl_oso_files "")
    foreach(_sh ${_osl_shaders})
        add_custom_command(
            OUTPUT "${_osl_oso_dir}/${_sh}.oso"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_osl_oso_dir}"
            COMMAND "${OSLC_BIN}" -I"${OSL_ROOT}/src/shaders"
                    -o "${_osl_oso_dir}/${_sh}.oso"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../data/test_osl/${_sh}.osl"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../data/test_osl/${_sh}.osl"
            COMMENT "Compiling OSL test shader ${_sh} (oslc)")
        list(APPEND _osl_oso_files "${_osl_oso_dir}/${_sh}.oso")
    endforeach()
    add_custom_target(printman_osl_test_shaders DEPENDS ${_osl_oso_files})
    add_dependencies(${_TEST_NAME}_tests printman_osl_test_shaders)

    target_sources(${_TEST_NAME}_tests PRIVATE test_osl_displace.cpp)
    target_include_directories(${_TEST_NAME}_tests PRIVATE
        "${OSL_ROOT}/build/include" "${OSL_ROOT}/src/include"
        "${OSL_BREW}/include" "${OSL_BREW}/opt/fmt/include")
    target_link_libraries(${_TEST_NAME}_tests "${OSL_EXEC_LIB}" "${OSL_OIIO_LIB}" "${OSL_OIIOU_LIB}")
    target_compile_definitions(${_TEST_NAME}_tests PRIVATE
        SLIC3R_OSL PRINTMAN_OSL_TEST_DIR="${_osl_oso_dir}")
    set_property(TARGET ${_TEST_NAME}_tests APPEND PROPERTY
        BUILD_RPATH "${OSL_ROOT}/build/lib;${OSL_BREW}/lib")
    message(STATUS "SLIC3R_OSL ON: OSL end-to-end test links ${OSL_EXEC_LIB}")
endif()

orcaslicer_copy_test_dlls()

orcaslicer_discover_tests(${_TEST_NAME}_tests)
