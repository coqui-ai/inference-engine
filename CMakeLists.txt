cmake_minimum_required(VERSION 3.10)

project(coqui-inference-engine)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_SKIP_BUILD_RPATH False)

add_executable(main src/main.cc
    src/stt.cc
    src/onnxmodelstate.cc
    src/modelstate.cc
    src/alphabet.cc
    src/stt_errors.cc
    src/ctcdecode/ctc_beam_search_decoder.cpp
    src/ctcdecode/decoder_utils.cpp
    src/ctcdecode/path_trie.cpp
    src/ctcdecode/scorer.cpp
    third_party/kenlm/util/bit_packing.cc
    third_party/kenlm/util/ersatz_progress.cc
    third_party/kenlm/util/exception.cc
    third_party/kenlm/util/file.cc
    third_party/kenlm/util/file_piece.cc
    third_party/kenlm/util/float_to_string.cc
    third_party/kenlm/util/integer_to_string.cc
    third_party/kenlm/util/mmap.cc
    third_party/kenlm/util/murmur_hash.cc
    third_party/kenlm/util/parallel_read.cc
    third_party/kenlm/util/pool.cc
    third_party/kenlm/util/read_compressed.cc
    third_party/kenlm/util/scoped.cc
    third_party/kenlm/util/spaces.cc
    third_party/kenlm/util/string_piece.cc
    third_party/kenlm/util/usage.cc
    third_party/kenlm/lm/bhiksha.cc
    third_party/kenlm/lm/binary_format.cc
    third_party/kenlm/lm/config.cc
    third_party/kenlm/lm/lm_exception.cc
    third_party/kenlm/lm/model.cc
    third_party/kenlm/lm/quantize.cc
    third_party/kenlm/lm/read_arpa.cc
    third_party/kenlm/lm/search_hashed.cc
    third_party/kenlm/lm/search_trie.cc
    third_party/kenlm/lm/sizes.cc
    third_party/kenlm/lm/trie.cc
    third_party/kenlm/lm/trie_sort.cc
    third_party/kenlm/lm/value_build.cc
    third_party/kenlm/lm/virtual_interface.cc
    third_party/kenlm/lm/vocab.cc
    third_party/kenlm/util/double-conversion/bignum-dtoa.cc
    third_party/kenlm/util/double-conversion/bignum.cc
    third_party/kenlm/util/double-conversion/cached-powers.cc
    third_party/kenlm/util/double-conversion/diy-fp.cc
    third_party/kenlm/util/double-conversion/double-conversion.cc
    third_party/kenlm/util/double-conversion/fast-dtoa.cc
    third_party/kenlm/util/double-conversion/fixed-dtoa.cc
    third_party/kenlm/util/double-conversion/strtod.cc
    third_party/openfst-1.6.7/src/lib/compat.cc
    third_party/openfst-1.6.7/src/lib/flags.cc
    third_party/openfst-1.6.7/src/lib/fst-types.cc
    third_party/openfst-1.6.7/src/lib/fst.cc
    third_party/openfst-1.6.7/src/lib/mapped-file.cc
    third_party/openfst-1.6.7/src/lib/properties.cc
    third_party/openfst-1.6.7/src/lib/symbol-table-ops.cc
    third_party/openfst-1.6.7/src/lib/symbol-table.cc
    third_party/openfst-1.6.7/src/lib/util.cc
    third_party/openfst-1.6.7/src/lib/weight.cc
    third_party/fft2d/fftsg.c
    third_party/fft2d/fftsg2d.c
    third_party/tensorflow/mfcc.cc
    third_party/tensorflow/mfcc_dct.cc
    third_party/tensorflow/mfcc_mel_filterbank.cc
    third_party/tensorflow/spectrogram.cc
)

target_include_directories(main PUBLIC
    third_party/onnxruntime/include/onnxruntime/core/session
    third_party/kenlm
    third_party/openfst-1.6.7/src/include
    src
    third_party/object_pool
    third_party/ThreadPool
    third_party/cereal/include
    third_party/fft2d
    third_party/tensorflow
)

target_compile_definitions(main PUBLIC KENLM_MAX_ORDER=6)

target_link_libraries(main
    ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.1.8.0.dylib
)
