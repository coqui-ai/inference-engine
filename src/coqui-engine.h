#ifndef COQUI_ENGINE_H
#define COQUI_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SWIG
    #if defined _MSC_VER
        #define ENGINE_EXPORT __declspec(dllexport)
    #else
        #define ENGINE_EXPORT __attribute__ ((visibility("default")))
    #endif /*End of _MSC_VER*/
#else
    #define ENGINE_EXPORT
#endif

typedef struct CoquiEngine CoquiEngine;

typedef struct CoquiModelPackage CoquiModelPackage;

typedef struct CoquiStreamingState CoquiStreamingState;

/**
 * @brief Stores text of an individual token, along with its timing information
 */
typedef struct TokenMetadata {
  /** The text corresponding to this token */
  const char* const text;

  /** Position of the token in units of 20ms */
  const unsigned int timestep;

  /** Position of the token in seconds */
  const float start_time;
} TokenMetadata;

/**
 * @brief A single transcript computed by the model, including a confidence
 *        value and the metadata for its constituent tokens.
 */
typedef struct CandidateTranscript {
  /** Array of TokenMetadata objects */
  const TokenMetadata* const tokens;
  /** Size of the tokens array */
  const unsigned int num_tokens;
  /** Approximated confidence value for this transcript. This is roughly the
   * sum of the acoustic model logit values for each timestep/character that
   * contributed to the creation of this transcript.
   */
  const double confidence;
} CandidateTranscript;

/**
 * @brief An array of CandidateTranscript objects computed by the model.
 */
typedef struct Metadata {
  /** Array of CandidateTranscript objects */
  const CandidateTranscript* const transcripts;
  /** Size of the transcripts array */
  const unsigned int num_transcripts;
} Metadata;

// sphinx-doc: error_code_listing_start

#define COQUI_FOR_EACH_ERROR(APPLY) \
  APPLY(COQUI_ERR_OK,                      0x0000, "No error.") \
  APPLY(COQUI_ERR_NO_MODEL,                0x1000, "Missing model information.") \
  APPLY(COQUI_ERR_INVALID_ALPHABET,        0x2000, "Invalid alphabet embedded in model. (Data corruption?)") \
  APPLY(COQUI_ERR_INVALID_SHAPE,           0x2001, "Invalid model shape.") \
  APPLY(COQUI_ERR_INVALID_SCORER,          0x2002, "Invalid scorer file.") \
  APPLY(COQUI_ERR_MODEL_INCOMPATIBLE,      0x2003, "Incompatible model.") \
  APPLY(COQUI_ERR_SCORER_NOT_ENABLED,      0x2004, "External scorer is not enabled.") \
  APPLY(COQUI_ERR_SCORER_UNREADABLE,       0x2005, "Could not read scorer file.") \
  APPLY(COQUI_ERR_SCORER_INVALID_LM,       0x2006, "Could not recognize language model header in scorer.") \
  APPLY(COQUI_ERR_SCORER_NO_TRIE,          0x2007, "Reached end of scorer file before loading vocabulary trie.") \
  APPLY(COQUI_ERR_SCORER_INVALID_TRIE,     0x2008, "Invalid magic in trie header.") \
  APPLY(COQUI_ERR_SCORER_VERSION_MISMATCH, 0x2009, "Scorer file version does not match expected version.") \
  APPLY(COQUI_ERR_FAIL_INIT_MMAP,          0x3000, "Failed to initialize memory mapped model.") \
  APPLY(COQUI_ERR_FAIL_INIT_SESS,          0x3001, "Failed to initialize the session.") \
  APPLY(COQUI_ERR_FAIL_INTERPRETER,        0x3002, "Interpreter failed.") \
  APPLY(COQUI_ERR_FAIL_RUN_SESS,           0x3003, "Failed to run the session.") \
  APPLY(COQUI_ERR_FAIL_CREATE_STREAM,      0x3004, "Error creating the stream.") \
  APPLY(COQUI_ERR_FAIL_READ_PROTOBUF,      0x3005, "Error reading the proto buffer model file.") \
  APPLY(COQUI_ERR_FAIL_CREATE_SESS,        0x3006, "Failed to create session.") \
  APPLY(COQUI_ERR_FAIL_CREATE_MODEL,       0x3007, "Could not allocate model state.") \
  APPLY(COQUI_ERR_FAIL_INSERT_HOTWORD,     0x3008, "Could not insert hot-word.") \
  APPLY(COQUI_ERR_FAIL_CLEAR_HOTWORD,      0x3009, "Could not clear hot-words.") \
  APPLY(COQUI_ERR_FAIL_ERASE_HOTWORD,      0x3010, "Could not erase hot-word.")

// sphinx-doc: error_code_listing_end

enum Coqui_Error_Codes
{
#define DEFINE(NAME, VALUE, DESC) NAME = VALUE,
COQUI_FOR_EACH_ERROR(DEFINE)
#undef DEFINE
};

/**
 * @brief Performs global initialization tasks before all other APIs can be used.
 * 
 * @param[out] retval a CoquiEngine pointer.
 * 
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_InitEngine(CoquiEngine** retval);

/**
 * @brief Frees CoquiEngine object created by {@link Coqui_InitEngine}
 * 
 * @param engine a CoquiEngine pointer.
 * 
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_FreeEngine(CoquiEngine* engine);

/**
 * @brief An object providing an interface to a trained Coqui model package.
 *
 * @param model_path The path to the frozen model graph.
 * @param[out] retval a CoquiModelPackage pointer
 *
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_LoadModelPackage(CoquiEngine* engine,
                           const char* model_path,
                           CoquiModelPackage** retval);

/**
 * @brief Get beam width value used by the model. If {@link Coqui_SetModelBeamWidth}
 *        was not called before, will return the default value loaded from the
 *        model file.
 *
 * @param ctx A CoquiModelPackage pointer created with {@link Coqui_CreateModel}.
 *
 * @return Beam width value used by the model.
 */
ENGINE_EXPORT
unsigned int Coqui_GetModelBeamWidth(const CoquiModelPackage* ctx);

/**
 * @brief Set beam width value used by the model.
 *
 * @param ctx A CoquiModelPackage pointer created with {@link Coqui_CreateModel}.
 * @param beam_width The beam width used by the model. A larger beam width value
 *                   generates better results at the cost of decoding time.
 *
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_SetModelBeamWidth(CoquiModelPackage* ctx,
                            unsigned int beam_width);

/**
 * @brief Return the sample rate expected by a model.
 *
 * @param ctx A CoquiModelPackage pointer created with {@link Coqui_CreateModel}.
 *
 * @return Sample rate expected by the model for its input.
 */
ENGINE_EXPORT
int Coqui_GetModelSampleRate(const CoquiModelPackage* ctx);

/**
 * @brief Frees associated resources and destroys model object.
 */
ENGINE_EXPORT
void Coqui_FreeModel(CoquiModelPackage* ctx);

/**
 * @brief Enable decoding using an external scorer.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 * @param scorer_path The path to the external scorer file.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
ENGINE_EXPORT
int Coqui_EnableExternalScorer(CoquiModelPackage* ctx,
                               const char* scorer_path);

/**
 * @brief Add a hot-word and its boost.
 *
 * Words that don't occur in the scorer (e.g. proper nouns) or strings that contain spaces won't be taken into account.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 * @param word The hot-word.
 * @param boost The boost. Positive value increases and negative reduces chance of a word occuring in a transcription. Excessive positive boost might lead to splitting up of letters of the word following the hot-word.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
ENGINE_EXPORT
int Coqui_AddHotWord(CoquiModelPackage* ctx,
                     const char* word,
                     float boost);

/**
 * @brief Remove entry for a hot-word from the hot-words map.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 * @param word The hot-word.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
ENGINE_EXPORT
int Coqui_EraseHotWord(CoquiModelPackage* ctx,
                       const char* word);

/**
 * @brief Removes all elements from the hot-words map.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
ENGINE_EXPORT
int Coqui_ClearHotWords(CoquiModelPackage* ctx);

/**
 * @brief Disable decoding using an external scorer.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 *
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_DisableExternalScorer(CoquiModelPackage* ctx);

/**
 * @brief Set hyperparameters alpha and beta of the external scorer.
 *
 * @param ctx The CoquiModelPackage pointer for the model being changed.
 * @param alpha The alpha hyperparameter of the decoder. Language model weight.
 * @param beta The beta hyperparameter of the decoder. Word insertion weight.
 *
 * @return Zero on success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_SetScorerAlphaBeta(CoquiModelPackage* ctx,
                             float alpha,
                             float beta);

/**
 * @brief Use the Coqui model package to convert speech to text.
 *
 * @param ctx The CoquiModelPackage pointer for the model to use.
 * @param buffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate (matching what the model was trained on).
 * @param buffer_size The number of samples in the audio signal.
 *
 * @return The STT result. The user is responsible for freeing the string using
 *         {@link Coqui_FreeString()}. Returns NULL on error.
 */
ENGINE_EXPORT
char* Coqui_SpeechToText(CoquiModelPackage* ctx,
                         const short* buffer,
                         unsigned int buffer_size);

/**
 * @brief Use the Coqui model package to convert speech to text and output results
 * including metadata.
 *
 * @param ctx The CoquiModelPackage pointer for the model to use.
 * @param buffer A 16-bit, mono raw audio signal at the appropriate
 *                sample rate (matching what the model was trained on).
 * @param buffer_size The number of samples in the audio signal.
 * @param num_results The maximum number of CandidateTranscript structs to return. Returned value might be smaller than this.
 *
 * @return Metadata struct containing multiple CandidateTranscript structs. Each
 *         transcript has per-token metadata including timing information. The
 *         user is responsible for freeing Metadata by calling {@link Coqui_FreeMetadata()}.
 *         Returns NULL on error.
 */
ENGINE_EXPORT
Metadata* Coqui_SpeechToTextWithMetadata(CoquiModelPackage* ctx,
                                         const short* buffer,
                                         unsigned int buffer_size,
                                         unsigned int num_results);

/**
 * @brief Create a new streaming inference state. The streaming state returned
 *        by this function can then be passed to {@link Coqui_FeedAudioContent()}
 *        and {@link Coqui_FinishStream()}.
 *
 * @param ctx The CoquiModelPackage pointer for the model to use.
 * @param[out] retval an opaque pointer that represents the streaming state. Can
 *                    be NULL if an error occurs.
 *
 * @return Zero for success, non-zero on failure.
 */
ENGINE_EXPORT
int Coqui_CreateStream(CoquiModelPackage* ctx,
                       CoquiStreamingState** retval);

/**
 * @brief Feed audio samples to an ongoing streaming inference.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 * @param buffer An array of 16-bit, mono raw audio samples at the
 *                appropriate sample rate (matching what the model was trained on).
 * @param buffer_size The number of samples in @p buffer.
 */
ENGINE_EXPORT
void Coqui_FeedAudioContent(CoquiStreamingState* sctx,
                            const short* buffer,
                            unsigned int buffer_size);

/**
 * @brief Compute the intermediate decoding of an ongoing streaming inference.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 *
 * @return The STT intermediate result. The user is responsible for freeing the
 *         string using {@link Coqui_FreeString()}.
 */
ENGINE_EXPORT
char* Coqui_IntermediateDecode(const CoquiStreamingState* sctx);

/**
 * @brief Compute the intermediate decoding of an ongoing streaming inference,
 *        return results including metadata.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 * @param num_results The number of candidate transcripts to return.
 *
 * @return Metadata struct containing multiple candidate transcripts. Each transcript
 *         has per-token metadata including timing information. The user is
 *         responsible for freeing Metadata by calling {@link Coqui_FreeMetadata()}.
 *         Returns NULL on error.
 */
ENGINE_EXPORT
Metadata* Coqui_IntermediateDecodeWithMetadata(const CoquiStreamingState* sctx,
                                               unsigned int num_results);

/**
 * @brief Compute the final decoding of an ongoing streaming inference and return
 *        the result. Signals the end of an ongoing streaming inference.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 *
 * @return The STT result. The user is responsible for freeing the string using
 *         {@link Coqui_FreeString()}.
 *
 * @note This method will free the state pointer (@p sctx).
 */
ENGINE_EXPORT
char* Coqui_FinishStream(CoquiStreamingState* sctx);

/**
 * @brief Compute the final decoding of an ongoing streaming inference and return
 *        results including metadata. Signals the end of an ongoing streaming
 *        inference.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 * @param num_results The number of candidate transcripts to return.
 *
 * @return Metadata struct containing multiple candidate transcripts. Each transcript
 *         has per-token metadata including timing information. The user is
 *         responsible for freeing Metadata by calling {@link Coqui_FreeMetadata()}.
 *         Returns NULL on error.
 *
 * @note This method will free the state pointer (@p sctx).
 */
ENGINE_EXPORT
Metadata* Coqui_FinishStreamWithMetadata(CoquiStreamingState* sctx,
                                         unsigned int num_results);

/**
 * @brief Destroy a streaming state without decoding the computed logits. This
 *        can be used if you no longer need the result of an ongoing streaming
 *        inference and don't want to perform a costly decode operation.
 *
 * @param sctx A streaming state pointer returned by {@link Coqui_CreateStream()}.
 *
 * @note This method will free the state pointer (@p sctx).
 */
ENGINE_EXPORT
void Coqui_FreeStream(CoquiStreamingState* sctx);

/**
 * @brief Free memory allocated for metadata information.
 */
ENGINE_EXPORT
void Coqui_FreeMetadata(Metadata* m);

/**
 * @brief Free a char* string returned by the Coqui STT API.
 */
ENGINE_EXPORT
void Coqui_FreeString(char* str);

/**
 * @brief Returns the version of this library. The returned version is a semantic
 *        version (SemVer 2.0.0). The string returned must be freed with {@link Coqui_FreeString()}.
 *
 * @return The version string.
 */
ENGINE_EXPORT
char* Coqui_Version();

/**
 * @brief Returns a textual description corresponding to an error code.
 *        The string returned must be freed with @{link Coqui_FreeString()}.
 *
 * @return The error description.
 */
ENGINE_EXPORT
char* Coqui_ErrorCodeToErrorMessage(int aErrorCode);

#undef ENGINE_EXPORT

#ifdef __cplusplus
}
#endif

#endif /* COQUI_ENGINE_H */
