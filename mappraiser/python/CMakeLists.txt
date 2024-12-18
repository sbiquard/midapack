# Determine our python install location and set up a rule to install the modules
if(DEFINED PYTHON_MAJORMINOR)
    message(STATUS "Python version set to: ${PYTHON_MAJORMINOR}")
else()
    # Try to find an interpreter
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "if True:
            import sys
            print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))"
        OUTPUT_VARIABLE PYTHON_MAJORMINOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

# Hardcode this to "lib" for now, since lib32/lib64 is a pain
# to use on many systems.
#set(PYTHON_SITE
#    "${CMAKE_INSTALL_LIBDIR}/python${PYTHON_MAJORMINOR}/site-packages")
set(PYTHON_SITE "lib/python${PYTHON_MAJORMINOR}/site-packages/mappraiser")
message(STATUS "Python files will be installed to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE}")

# Install the top-level __init__.py file
install(FILES
    __init__.py
    tests.py
    DESTINATION ${PYTHON_SITE}
)

add_subdirectory(toast_op)
add_subdirectory(workflow)
add_subdirectory(wrapper)
