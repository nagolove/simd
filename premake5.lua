workspace 'SIMD'
    configurations { "DEFAULT" }
    targetdir "."
    buildoptions {
        "-mavx",
        "-mavx512f",
    }

project 'simd_1'
    kind "ConsoleApp"
    files("**.c")

    filter "configurations:DEFAULT"
        symbols "On"
