#ifndef ONNXMODELSTATE_H
#define ONNXMODELSTATE_H

#include <vector>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include "modelstate.h"

struct ONNXModelPackage : public CoquiModelPackage
{
  // weak
  Ort::Env* ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;

  ONNXModelPackage(Ort::Env* ort_env);
  virtual ~ONNXModelPackage();

  virtual int init(const char* model_path) override;

  virtual void infer(std::vector<float>& mfcc,
                     unsigned int n_frames,
                     std::vector<float>& previous_state_c,
                     std::vector<float>& previous_state_h,
                     std::vector<float>& logits_output,
                     std::vector<float>& state_c_output,
                     std::vector<float>& state_h_output) override;

  virtual void compute_mfcc(const std::vector<float>& audio_buffer,
                            std::vector<float>& mfcc_output) override;
};

#endif // ONNXMODELSTATE_H
