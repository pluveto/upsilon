# BUILD

load("//tests:test_generator.bzl", "generate_test")

# Find all files ending with "_test.cc" in the current directory
test_files = glob(["*_test.cc"])

# Generate test rules for each test file found using list comprehension
[generate_test(name = test_file[:-len("_test.cc")] + "_test", src = [test_file]) for test_file in test_files]
