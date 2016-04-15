# Install Python package
find_program(PYTHON "python")
if(BUILD_PYTHON AND PYTHON)

    set(PYTHONLIBRARYPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pguresvt/__init__.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pguresvt/pguresvt.py.in 
                   ${CMAKE_CURRENT_BINARY_DIR}/pguresvt/pguresvt.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pguresvt/__init__.py 
                   ${CMAKE_CURRENT_BINARY_DIR}/pguresvt/__init__.py)               
    configure_file(${SETUP_PY_IN} ${SETUP_PY})
        
    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON}
                       ARGS setup.py build
                       DEPENDS ${DEPS})
                       
    add_custom_target(pythontarget ALL DEPENDS ${OUTPUT})
        
    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user)")
endif()
