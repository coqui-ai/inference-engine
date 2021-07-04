#include "onnxmodelstate.h"

#include "onnxruntime_session_options_config_keys.h"

#include "spectrogram.h"
#include "mfcc.h"

using namespace Ort;
using std::vector;

#include <chrono>
#include <iostream>
using namespace std;

ONNXModelPackage::ONNXModelPackage(Ort::Env* ort_env)
  : CoquiModelPackage()
  , ort_env_(ort_env)
  , ort_session_(nullptr)
{
}

ONNXModelPackage::~ONNXModelPackage()
{
}

int
ONNXModelPackage::init(const char* model_path)
{
  int err = CoquiModelPackage::init(model_path);
  if (err != COQUI_ERR_OK) {
    return err;
  }

  // initialize session options if needed
  SessionOptions session_options;

  // If onnxruntime lib is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Use global thread pool from environment
  session_options.DisablePerSessionThreads();

  // Use environment allocators
  session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();
  ort_session_.reset(new Session(*ort_env_, model_path, session_options));
  auto t2 = high_resolution_clock::now();
  cerr << "Session initialized in " << duration_cast<milliseconds>(t2 - t1).count() << "ms\n";

  AllocatorWithDefaultOptions allocator;
  ModelMetadata meta = ort_session_->GetModelMetadata();

  char* serialized_alphabet = meta.LookupCustomMetadataMap("alphabet", allocator);
  if (!serialized_alphabet) {
    std::cerr << "Unable to load alphabet from model metadata." << std::endl;
    return COQUI_ERR_MODEL_INCOMPATIBLE;
  }
  err = alphabet_.Deserialize(serialized_alphabet, strlen(serialized_alphabet));
  allocator.Free(serialized_alphabet);

  //TODO: figure out engine versioning (detached from STT repo structure)
  // int graph_version = meta.GetVersion();
  // if (graph_version < ds_graph_version()) {
  //   std::cerr << "Specified model file version (" << graph_version << ") is "
  //             << "incompatible with minimum version supported by this client ("
  //             << ds_graph_version() << "). See "
  //             << "https://stt.readthedocs.io/en/latest/USING.html#model-compatibility "
  //             << "for more information" << std::endl;
  //   return COQUI_ERR_MODEL_INCOMPATIBLE;
  // }

  char* md_sample_rate = meta.LookupCustomMetadataMap("sample_rate", allocator);
  if (!md_sample_rate) {
    std::cerr << "Unable to load sample_rate from model metadata." << std::endl;
    return COQUI_ERR_MODEL_INCOMPATIBLE;
  }

  char* md_feature_win_len = meta.LookupCustomMetadataMap("feature_win_len", allocator);
  if (!md_feature_win_len) {
    std::cerr << "Unable to load feature_win_len from model metadata." << std::endl;
    return COQUI_ERR_MODEL_INCOMPATIBLE;
  }

  char* md_feature_win_step = meta.LookupCustomMetadataMap("feature_win_step", allocator);
  if (!md_feature_win_step) {
    std::cerr << "Unable to load feature_win_step from model metadata." << std::endl;
    return COQUI_ERR_MODEL_INCOMPATIBLE;
  }

  char* md_beam_width = meta.LookupCustomMetadataMap("beam_width", allocator);
  if (!md_beam_width) {
    std::cerr << "Unable to load beam_width from model metadata." << std::endl;
    return COQUI_ERR_MODEL_INCOMPATIBLE;
  }

  sample_rate_ = std::stoi(md_sample_rate);
  int win_len_ms = std::stoi(md_feature_win_len);
  int win_step_ms = std::stoi(md_feature_win_step);
  audio_win_len_ = sample_rate_ * (win_len_ms / 1000.0);
  audio_win_step_ = sample_rate_ * (win_step_ms / 1000.0);
  beam_width_ = (unsigned int)(std::stoi(md_beam_width));

  assert(sample_rate_ > 0);
  assert(audio_win_len_ > 0);
  assert(audio_win_step_ > 0);
  assert(beam_width_ > 0);
  assert(alphabet_.GetSize() > 0);

  for (int i = 0; i < ort_session_->GetInputCount(); ++i) {
    std::string input_name = ort_session_->GetInputName(i, allocator);
    auto shape = ort_session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    if (input_name == "input_node") {
      n_steps_ = shape[1];
      n_context_ = (shape[2] - 1) / 2;
      n_features_ = shape[3];
      mfcc_feats_per_timestep_ = shape[2] * shape[3];
    } else if (input_name == "previous_state_c") {
      state_size_ = shape[1];
    }
  }

  if (n_context_ == -1 || n_features_ == -1) {
    std::cerr << "Error: Could not infer input shape from model file. "
              << "Make sure input_node is a 4D tensor with shape "
              << "[batch_size=1, time, window_size, n_features]."
              << std::endl;
    return COQUI_ERR_INVALID_SHAPE;
  }

  for (int i = 0; i < ort_session_->GetOutputCount(); ++i) {
    std::string output_name = ort_session_->GetOutputName(i, allocator);
    auto shape = ort_session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

    if (output_name == "logits") {
      int final_dim_size = shape[1] - 1;
      if (final_dim_size != alphabet_.GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << alphabet_.GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        return COQUI_ERR_INVALID_ALPHABET;
      }
    }
  }

  return COQUI_ERR_OK;
}

Value
tensor_from_vector(std::vector<float>& vec, const std::vector<int64_t>& shape)
{
  AllocatorWithDefaultOptions allocator;
  Value tensor = Value::CreateTensor<float>(
    allocator,
    shape.data(),
    shape.size()
  );
  assert(tensor.IsTensor());
  float* rawarr = tensor.GetTensorMutableData<float>();

  int64_t num_elements = 1;
  for (auto dim : shape) {
    num_elements *= dim;
  }

  assert(vec.size() <= num_elements);

  int i = 0;
  for (int i = 0; i < vec.size(); ++i) {
    rawarr[i] = vec[i];
  }
  for (; i < num_elements; ++i) {
    rawarr[i] = 0.f;
  }

  return tensor;
}

void
copy_tensor_to_vector(const Value& tensor, vector<float>& vec, int num_elements = -1)
{
  const float* rawarr = tensor.GetTensorData<float>();
  if (num_elements == -1) {
    num_elements = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
  }
  for (int i = 0; i < num_elements; ++i) {
    vec.push_back(rawarr[i]);
  }
}

void
ONNXModelPackage::infer(std::vector<float>& mfcc,
                        unsigned int n_frames,
                        std::vector<float>& previous_state_c,
                        std::vector<float>& previous_state_h,
                        vector<float>& logits_output,
                        vector<float>& state_c_output,
                        vector<float>& state_h_output)
{
  const size_t num_classes = alphabet_.GetSize() + 1; // +1 for blank

  Value input_t = tensor_from_vector(mfcc, {BATCH_SIZE, (int64_t)n_steps_, (int64_t)2*n_context_+1, (int64_t)n_features_});
  Value previous_state_c_t = tensor_from_vector(previous_state_c, {BATCH_SIZE, (int64_t)state_size_});
  Value previous_state_h_t = tensor_from_vector(previous_state_h, {BATCH_SIZE, (int64_t)state_size_});

  const std::vector<const char*> input_node_names = {
    "input_node", "previous_state_c", "previous_state_h"
  };
  const std::vector<const char*> output_node_names = {
    "logits", "new_state_c", "new_state_h"
  };

  Value input_tensors[] = {
    std::move(input_t),
    std::move(previous_state_c_t),
    std::move(previous_state_h_t)
  };

  vector<Value> outputs = ort_session_->Run(
    RunOptions{nullptr},
    input_node_names.data(),
    input_tensors,
    sizeof(input_tensors) / sizeof(Value),
    output_node_names.data(),
    output_node_names.size()
  );

  assert(outputs.size() == output_node_names.size());

  copy_tensor_to_vector(outputs[0], logits_output, n_frames * BATCH_SIZE * num_classes);

  state_c_output.clear();
  state_c_output.reserve(state_size_);
  copy_tensor_to_vector(outputs[1], state_c_output);

  state_h_output.clear();
  state_h_output.reserve(state_size_);
  copy_tensor_to_vector(outputs[2], state_h_output);
}

void
ONNXModelPackage::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output)
{
  std::vector<std::vector<float>> spectrogram_output;
  tensorflow::Spectrogram spec;
  spec.Initialize(audio_win_len_, audio_win_step_);
  spec.ComputeSquaredMagnitudeSpectrogram(samples, &spectrogram_output);

  if (spectrogram_output.size() == 0) {
    return;
  }

  tensorflow::Mfcc mfcc;
  mfcc.set_upper_frequency_limit(sample_rate_ / 2);
  mfcc.set_dct_coefficient_count(n_features_);
  mfcc.Initialize(spectrogram_output[0].size(), sample_rate_);
  for (int i = 0; i < spectrogram_output.size(); ++i) {
    mfcc.Compute(spectrogram_output[i], &mfcc_output);
  }
}
