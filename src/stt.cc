#include <algorithm>
#ifdef _MSC_VER
  #define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "coqui-engine.h"
#include "alphabet.h"
#include "modelstate.h"

#include "onnxmodelstate.h"

#include "ctcdecode/ctc_beam_search_decoder.h"

#ifdef __ANDROID__
#include <android/log.h>
#define  LOG_TAG    "coqui-engine"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define  LOGD(...)
#define  LOGE(...)
#endif // __ANDROID__

using std::vector;

/* This is the implementation of the streaming inference API.

   The streaming process uses three buffers that are fed eagerly as audio data
   is fed in. The buffers only hold the minimum amount of data needed to do a
   step in the acoustic model. The three buffers which live in CoquiStreamingState
   are:

   - audio_buffer, used to buffer audio samples until there's enough data to
     compute input features for a single window.

   - mfcc_buffer, used to buffer input features until there's enough data for
     a single timestep. Remember there's overlap in the features, each timestep
     contains n_context past feature frames, the current feature frame, and
     n_context future feature frames, for a total of 2*n_context + 1 feature
     frames per timestep.

   - batch_buffer, used to buffer timesteps until there's enough data to compute
     a batch of n_steps.

   Data flows through all three buffers as audio samples are fed via the public
   API. When audio_buffer is full, features are computed from it and pushed to
   mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
   When batch_buffer is full, we do a single step through the acoustic model
   and accumulate the intermediate decoding state in the DecoderState structure.

   When finishStream() is called, we return the corresponding transcript from
   the current decoder state.
*/
struct CoquiStreamingState {
  vector<float> audio_buffer_;
  vector<float> mfcc_buffer_;
  vector<float> batch_buffer_;
  vector<float> previous_state_c_;
  vector<float> previous_state_h_;

  CoquiModelPackage* model_;
  DecoderState decoder_state_;

  CoquiStreamingState();
  ~CoquiStreamingState();

  void feedAudioContent(const short* buffer, unsigned int buffer_size);
  char* intermediateDecode() const;
  Metadata* intermediateDecodeWithMetadata(unsigned int num_results) const;
  void finalizeStream();
  char* finishStream();
  Metadata* finishStreamWithMetadata(unsigned int num_results);

  void processAudioWindow(const vector<float>& buf);
  void processMfccWindow(const vector<float>& buf);
  void pushMfccBuffer(const vector<float>& buf);
  void addZeroMfccWindow();
  void processBatch(vector<float>& buf, unsigned int n_steps);
};

CoquiStreamingState::CoquiStreamingState()
{
}

CoquiStreamingState::~CoquiStreamingState()
{
}

template<typename T>
void
shift_buffer_left(vector<T>& buf, int shift_amount)
{
  std::rotate(buf.begin(), buf.begin() + shift_amount, buf.end());
  buf.resize(buf.size() - shift_amount);
}

void
CoquiStreamingState::feedAudioContent(const short* buffer,
                                 unsigned int buffer_size)
{
  // Consume all the data that was passed in, processing full buffers if needed
  while (buffer_size > 0) {
    while (buffer_size > 0 && audio_buffer_.size() < model_->audio_win_len_) {
      // Convert i16 sample into f32
      float multiplier = 1.0f / (1 << 15);
      audio_buffer_.push_back((float)(*buffer) * multiplier);
      ++buffer;
      --buffer_size;
    }

    // If the buffer is full, process and shift it
    if (audio_buffer_.size() == model_->audio_win_len_) {
      processAudioWindow(audio_buffer_);
      // Shift data by one step
      shift_buffer_left(audio_buffer_, model_->audio_win_step_);
    }

    // Repeat until buffer empty
  }
}

char*
CoquiStreamingState::intermediateDecode() const
{
  return model_->decode(decoder_state_);
}

Metadata*
CoquiStreamingState::intermediateDecodeWithMetadata(unsigned int num_results) const
{
  return model_->decode_metadata(decoder_state_, num_results);
}

char*
CoquiStreamingState::finishStream()
{
  finalizeStream();
  return model_->decode(decoder_state_);
}

Metadata*
CoquiStreamingState::finishStreamWithMetadata(unsigned int num_results)
{
  finalizeStream();
  return model_->decode_metadata(decoder_state_, num_results);
}

void
CoquiStreamingState::processAudioWindow(const vector<float>& buf)
{
  // Compute MFCC features
  vector<float> mfcc;
  mfcc.reserve(model_->n_features_);
  model_->compute_mfcc(buf, mfcc);
  pushMfccBuffer(mfcc);
}

void
CoquiStreamingState::finalizeStream()
{
  // Flush audio buffer
  processAudioWindow(audio_buffer_);

  // Add empty mfcc vectors at end of sample
  for (int i = 0; i < model_->n_context_; ++i) {
    addZeroMfccWindow();
  }

  // Process final batch
  if (batch_buffer_.size() > 0) {
    processBatch(batch_buffer_, batch_buffer_.size()/model_->mfcc_feats_per_timestep_);
  }
}

void
CoquiStreamingState::addZeroMfccWindow()
{
  vector<float> zero_buffer(model_->n_features_, 0.f);
  pushMfccBuffer(zero_buffer);
}

template<typename InputIt, typename OutputIt>
InputIt
copy_up_to_n(InputIt from_begin, InputIt from_end, OutputIt to_begin, int max_elems)
{
  int next_copy_amount = std::min<int>(std::distance(from_begin, from_end), max_elems);
  std::copy_n(from_begin, next_copy_amount, to_begin);
  return from_begin + next_copy_amount;
}

void
CoquiStreamingState::pushMfccBuffer(const vector<float>& buf)
{
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to mfcc_buffer, stopping if we have a full context window
    start = copy_up_to_n(start, end, std::back_inserter(mfcc_buffer_),
                         model_->mfcc_feats_per_timestep_ - mfcc_buffer_.size());
    assert(mfcc_buffer_.size() <= model_->mfcc_feats_per_timestep_);

    // If we have a full context window
    if (mfcc_buffer_.size() == model_->mfcc_feats_per_timestep_) {
      processMfccWindow(mfcc_buffer_);
      // Shift data by one step of one mfcc feature vector
      shift_buffer_left(mfcc_buffer_, model_->n_features_);
    }
  }
}

void
CoquiStreamingState::processMfccWindow(const vector<float>& buf)
{
  auto start = buf.begin();
  auto end = buf.end();
  while (start != end) {
    // Copy from input buffer to batch_buffer, stopping if we have a full batch
    start = copy_up_to_n(start, end, std::back_inserter(batch_buffer_),
                         model_->n_steps_ * model_->mfcc_feats_per_timestep_ - batch_buffer_.size());
    assert(batch_buffer_.size() <= model_->n_steps_ * model_->mfcc_feats_per_timestep_);

    // If we have a full batch
    if (batch_buffer_.size() == model_->n_steps_ * model_->mfcc_feats_per_timestep_) {
      processBatch(batch_buffer_, model_->n_steps_);
      batch_buffer_.resize(0);
    }
  }
}

void
CoquiStreamingState::processBatch(vector<float>& buf, unsigned int n_steps)
{
  fprintf(stderr, "mfccs: [");
  for (int i = 0; i < model_->n_features_; ++i) {
    fprintf(stderr, "%.2f ", buf[i]);
  }
  fprintf(stderr, "%s]\n", n_steps > 1 ? "..." : "");

  vector<float> logits;
  model_->infer(buf,
                n_steps,
                previous_state_c_,
                previous_state_h_,
                logits,
                previous_state_c_,
                previous_state_h_);

  const size_t num_classes = model_->alphabet_.GetSize() + 1; // +1 for blank
  const int n_frames = logits.size() / (CoquiModelPackage::BATCH_SIZE * num_classes);

  fprintf(stderr, "logits: [");
  for (int i = 0; i < num_classes; ++i) {
    fprintf(stderr, "%.2f ", logits[i]);
  }
  fprintf(stderr, "%s]\n", n_frames > 1 ? "..." : "");

  // Convert logits to double
  vector<double> inputs(logits.begin(), logits.end());

  decoder_state_.next(inputs.data(),
                      n_frames,
                      num_classes);
}

struct CoquiEngine
{
  std::unique_ptr<Ort::Env> ort_env_;
  Ort::MemoryInfo minfo_;

  CoquiEngine()
    : minfo_(nullptr)
  {
    OrtThreadingOptions* options;
    Ort::GetApi().CreateThreadingOptions(&options);
    ort_env_.reset(new Ort::Env(options, ORT_LOGGING_LEVEL_VERBOSE, "coqui-engine"));
    minfo_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::ArenaCfg cfg(1073741824, -1, -1, -1);
    ort_env_->CreateAndRegisterAllocator(minfo_, cfg);
  }
};

int
Coqui_InitEngine(CoquiEngine** retval)
{
  Ort::InitApi();
  *retval = new CoquiEngine();
  return COQUI_ERR_OK;
}

int
Coqui_FreeEngine(CoquiEngine* engine)
{
  delete engine;
  return 0;
}

int
Coqui_LoadModelPackage(CoquiEngine* engine,
                       const char* aModelPath,
                       CoquiModelPackage** retval)
{
  *retval = nullptr;

//   std::cerr << "TensorFlow: " << tf_local_git_version() << std::endl;
//   std::cerr << " Coqui STT: " << ds_git_version() << std::endl;
// #ifdef __ANDROID__
//   LOGE("TensorFlow: %s", tf_local_git_version());
//   LOGD("TensorFlow: %s", tf_local_git_version());
//   LOGE(" Coqui STT: %s", ds_git_version());
//   LOGD(" Coqui STT: %s", ds_git_version());
// #endif

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, cannot continue." << std::endl;
    return COQUI_ERR_NO_MODEL;
  }

  //TODO: init from memory, directly from within loaded/parsed package
  std::unique_ptr<CoquiModelPackage> model(new ONNXModelPackage(engine->ort_env_.get()));

  if (!model) {
    std::cerr << "Could not allocate model state." << std::endl;
    return COQUI_ERR_FAIL_CREATE_MODEL;
  }

  int err = model->init(aModelPath);
  if (err != COQUI_ERR_OK) {
    return err;
  }

  *retval = model.release();
  return COQUI_ERR_OK;
}

unsigned int
Coqui_GetModelBeamWidth(const CoquiModelPackage* ctx)
{
  return ctx->beam_width_;
}

int
Coqui_SetModelBeamWidth(CoquiModelPackage* ctx, unsigned int beam_width)
{
  ctx->beam_width_ = beam_width;
  return 0;
}

int
Coqui_GetModelSampleRate(const CoquiModelPackage* ctx)
{
  return ctx->sample_rate_;
}

void
Coqui_FreeModel(CoquiModelPackage* ctx)
{
  delete ctx;
}

int
Coqui_EnableExternalScorer(CoquiModelPackage* ctx,
                           const char* scorer_path)
{
  std::unique_ptr<Scorer> scorer(new Scorer());
  int err = scorer->init(scorer_path, ctx->alphabet_);
  if (err != 0) {
    return COQUI_ERR_INVALID_SCORER;
  }
  ctx->scorer_ = std::move(scorer);
  return COQUI_ERR_OK;
}

int
Coqui_AddHotWord(CoquiModelPackage* ctx,
                 const char* word,
                 float boost)
{
  if (ctx->scorer_) {
    const int size_before = ctx->hot_words_.size();
    ctx->hot_words_.insert( std::pair<std::string,float> (word, boost) );
    const int size_after = ctx->hot_words_.size();
    if (size_before == size_after) {
      return COQUI_ERR_FAIL_INSERT_HOTWORD;
    }
    return COQUI_ERR_OK;
  }
  return COQUI_ERR_SCORER_NOT_ENABLED;
}

int
Coqui_EraseHotWord(CoquiModelPackage* ctx,
                   const char* word)
{
  if (ctx->scorer_) {
    const int size_before = ctx->hot_words_.size();
    int err = ctx->hot_words_.erase(word);
    const int size_after = ctx->hot_words_.size();
    if (size_before == size_after) {
      return COQUI_ERR_FAIL_ERASE_HOTWORD;
    }
    return COQUI_ERR_OK;
  }
  return COQUI_ERR_SCORER_NOT_ENABLED;
}

int
Coqui_ClearHotWords(CoquiModelPackage* ctx)
{
  if (ctx->scorer_) {
    ctx->hot_words_.clear();
    const int size_after = ctx->hot_words_.size();
    if (size_after != 0) {
      return COQUI_ERR_FAIL_CLEAR_HOTWORD;
    }
    return COQUI_ERR_OK;
  }
  return COQUI_ERR_SCORER_NOT_ENABLED;
}

int
Coqui_DisableExternalScorer(CoquiModelPackage* ctx)
{
  if (ctx->scorer_) {
    ctx->scorer_.reset();
    return COQUI_ERR_OK;
  }
  return COQUI_ERR_SCORER_NOT_ENABLED;
}

int Coqui_SetScorerAlphaBeta(CoquiModelPackage* ctx,
                             float alpha,
                             float beta)
{
  if (ctx->scorer_) {
    ctx->scorer_->reset_params(alpha, beta);
    return COQUI_ERR_OK;
  }
  return COQUI_ERR_SCORER_NOT_ENABLED;
}

int
Coqui_CreateStream(CoquiModelPackage* ctx,
                   CoquiStreamingState** retval)
{
  *retval = nullptr;

  std::unique_ptr<CoquiStreamingState> sctx(new CoquiStreamingState());
  if (!sctx) {
    std::cerr << "Could not allocate streaming state." << std::endl;
    return COQUI_ERR_FAIL_CREATE_STREAM;
  }

  sctx->audio_buffer_.reserve(ctx->audio_win_len_);
  sctx->mfcc_buffer_.reserve(ctx->mfcc_feats_per_timestep_);
  sctx->mfcc_buffer_.resize(ctx->n_features_*ctx->n_context_, 0.f);
  sctx->batch_buffer_.reserve(ctx->n_steps_ * ctx->mfcc_feats_per_timestep_);
  sctx->previous_state_c_.resize(ctx->state_size_, 0.f);
  sctx->previous_state_h_.resize(ctx->state_size_, 0.f);
  sctx->model_ = ctx;

  const int cutoff_top_n = 40;
  const double cutoff_prob = 1.0;

  sctx->decoder_state_.init(ctx->alphabet_,
                            ctx->beam_width_,
                            cutoff_prob,
                            cutoff_top_n,
                            ctx->scorer_,
                            ctx->hot_words_);

  *retval = sctx.release();
  return COQUI_ERR_OK;
}

void
Coqui_FeedAudioContent(CoquiStreamingState* sctx,
                       const short* buffer,
                       unsigned int buffer_size)
{
  sctx->feedAudioContent(buffer, buffer_size);
}

char*
Coqui_IntermediateDecode(const CoquiStreamingState* sctx)
{
  return sctx->intermediateDecode();
}

Metadata*
Coqui_IntermediateDecodeWithMetadata(const CoquiStreamingState* sctx,
                                     unsigned int num_results)
{
  return sctx->intermediateDecodeWithMetadata(num_results);
}

char*
Coqui_FinishStream(CoquiStreamingState* sctx)
{
  char* str = sctx->finishStream();
  Coqui_FreeStream(sctx);
  return str;
}

Metadata*
Coqui_FinishStreamWithMetadata(CoquiStreamingState* sctx,
                               unsigned int num_results)
{
  Metadata* result = sctx->finishStreamWithMetadata(num_results);
  Coqui_FreeStream(sctx);
  return result;
}

CoquiStreamingState*
CreateStreamAndFeedAudioContent(CoquiModelPackage* ctx,
                                const short* buffer,
                                unsigned int buffer_size)
{
  CoquiStreamingState* sctx;
  int status = Coqui_CreateStream(ctx, &sctx);
  if (status != COQUI_ERR_OK) {
    return nullptr;
  }
  Coqui_FeedAudioContent(sctx, buffer, buffer_size);
  return sctx;
}

char*
Coqui_SpeechToText(CoquiModelPackage* ctx,
                   const short* buffer,
                   unsigned int buffer_size)
{
  CoquiStreamingState* sctx = CreateStreamAndFeedAudioContent(ctx, buffer, buffer_size);
  return Coqui_FinishStream(sctx);
}

Metadata*
Coqui_SpeechToTextWithMetadata(CoquiModelPackage* ctx,
                               const short* buffer,
                               unsigned int buffer_size,
                               unsigned int num_results)
{
  CoquiStreamingState* sctx = CreateStreamAndFeedAudioContent(ctx, buffer, buffer_size);
  return Coqui_FinishStreamWithMetadata(sctx, num_results);
}

void
Coqui_FreeStream(CoquiStreamingState* sctx)
{
  delete sctx;
}

void
Coqui_FreeMetadata(Metadata* m)
{
  if (m) {
    for (int i = 0; i < m->num_transcripts; ++i) {
      for (int j = 0; j < m->transcripts[i].num_tokens; ++j) {
        free((void*)m->transcripts[i].tokens[j].text);
      }

      free((void*)m->transcripts[i].tokens);
    }

    free((void*)m->transcripts);
    free(m);
  }
}

void
Coqui_FreeString(char* str)
{
  free(str);
}

char*
Coqui_Version()
{
  return strdup("Coqui Engine v0.0.1");
}
