load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

release = "1.10.0"
http_archive(
  name = "googletest",
  urls = ["https://github.com/google/googletest/archive/release-" + release + ".tar.gz"],
  strip_prefix = "googletest-release-" + release,
)

http_file(
  name = "cpplint_build",
  urls = ["https://raw.githubusercontent.com/nicmcd/pkgbuild/master/cpplint.BUILD"],
)

release = "1.5.2"
http_archive(
    name = "cpplint",
    urls = ["https://github.com/cpplint/cpplint/archive/" + release + ".tar.gz"],
    strip_prefix = "cpplint-" + release,
    build_file = "@cpplint_build//file:downloaded",
)

hash = "0a1151f"
http_archive(
  name = "paragraph",
  urls = ["https://github.com/paragraph-sim/paragraph-core/tarball/" + hash],
  type = "tar.gz",
  strip_prefix = "paragraph-sim-paragraph-core-" + hash,
)

# Tensorflow rules

#hash = "582c8d2"
hash = "e98b052"
http_archive(
  name = "org_tensorflow",
  urls = [
      "https://github.com/tensorflow/tensorflow/tarball/" + hash,
  ],
  type = "tar.gz",
  strip_prefix = "tensorflow-tensorflow-" + hash,
  patch_args = ["-p1"],
  patches = [
      "//tensorflow_patches:build.patch",
      "//tensorflow_patches:tf.patch",
  ],
)

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and workspace() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
