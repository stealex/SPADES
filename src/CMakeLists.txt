set(FORTRAIN_FILES radial.f
                   DHFS.f)

add_library(dhfs SHARED "${FORTRAIN_FILES}")