TRIBITS_SUBPACKAGE(SharedOnly)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

TRIBITS_ADD_LIBRARY(
  shared_only_lib
  SHARED
  HEADERS SharedOnly.hpp
  SOURCES SharedOnly.cpp
  )

TRIBITS_ADD_TEST_DIRECTORIES(test)

TRIBITS_SUBPACKAGE_POSTPROCESS()
