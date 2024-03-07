load("@rules_cc//cc:defs.bzl", "cc_test")

def generate_test(name, src):
    cc_test(
        name = name,
        srcs = src,
        includes = ["//src/core"],
        deps = [
            "//src/core:core",
            "@com_google_googletest//:gtest_main",
        ],
    )
