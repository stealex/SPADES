add_library(radial SHARED radial.f)
add_library(dhfs SHARED DHFS.f)

target_link_libraries(dhfs PUBLIC radial)