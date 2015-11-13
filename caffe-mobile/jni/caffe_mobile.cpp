#include <string>
#include <android/log.h>
#include <sstream>
#include "caffe_mobile.hpp"

using std::string;
using std::static_pointer_cast;
using std::clock;
using std::clock_t;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::shared_ptr;
using caffe::vector;
using caffe::MemoryDataLayer;

#define  LOG_TAG    "CaffeMobile"
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG, __VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)

namespace caffe {

namespace {
template <typename T>
vector<size_t> sort_to_index(vector<T> const& values) {
  vector<size_t> indices(values.size());
  std::iota(begin(indices), end(indices), static_cast<size_t>(0));

  std::sort(
    begin(indices), end(indices),
    [&](size_t a, size_t b) { return values[a] > values[b]; }
  );
  return indices;
}
}

CaffeMobile::CaffeMobile(string model_path, string weights_path)
  : output_height_(0), output_num_(0), test_time_(0.0)
{
  CHECK_GT(model_path.size(), 0) << "Need a model definition to score.";
  CHECK_GT(weights_path.size(), 0) << "Need model weights to score.";

  Caffe::set_mode(Caffe::CPU);

  clock_t t_start = clock();
  caffe_net_ = new Net<float>(model_path, caffe::TEST);
  caffe_net_->CopyTrainedLayersFrom(weights_path);
  clock_t t_end = clock();
  LOGI("%s: %g ms.", "Loading time", 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC);
}

CaffeMobile::~CaffeMobile() {
  free(caffe_net_);
  caffe_net_ = NULL;
}

int CaffeMobile::set_images(const vector<string> &img_paths) {
  CHECK(caffe_net_ != NULL);
  CHECK_GT(img_paths.size(), 0);

  Datum datum;
  vector<Datum> datumVector(img_paths.size());
  const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
    static_pointer_cast<MemoryDataLayer<float>>(caffe_net_->layer_by_name("data"));
  const bool has_crop = memory_data_layer->layer_param().transform_param().crop_size() != 0,
    is_color = memory_data_layer->channels() > 1;
  const int height = has_crop ? 0 : memory_data_layer->height(),
    width = has_crop ? 0 : memory_data_layer->width();
  for (int i = 0; i < img_paths.size(); i++) {
    CHECK(ReadImageToDatum(img_paths[i], 0, height, width, is_color, &datumVector[i]));
  }
  memory_data_layer->AddDatumVector(datumVector);

  return img_paths.size();
}

vector<float> CaffeMobile::predict() {
  CHECK(caffe_net_ != NULL);

  const vector<Blob<float>*> dummy_bottom_vec;
  float loss;

  clock_t time = clock();
  const Blob<float>& result = *caffe_net_->Forward(dummy_bottom_vec, &loss)[0];
  time = clock() - time;
  test_time_ = 1000.0 * time / CLOCKS_PER_SEC;
  LOGI("%s: %g ms.", "Prediction time: ", test_time_);

  const vector<float> probs = vector<float>(result.cpu_data(), result.cpu_data() + result.count());
  output_num_ = result.num();
  output_height_ = result.count(1);

  const float* result_data = result.cpu_data();
  if (output_height_ <= 20) {
    for (int i = 0; i < (output_num_ < 10 ? output_num_ : 10); i++) {
      ostringstream log;
      log << "  Image#"<< i << ":";
      for (int j = 0; j < output_height_; j++) {
        log << " " << result_data[i * output_height_ + j];
      }
      LOGV("%s", log.str().c_str());
    }
  }
  
  vector<float> result_copy(result.count());
  caffe_copy(result.count(), result_data, result_copy.data());
  return result_copy;
}

vector<int> CaffeMobile::predict_top_k(const string &img_path, int k) {
  CHECK(caffe_net_ != NULL);
  CHECK_GE(k, 1);

  set_images(vector<string>({img_path}));
  const vector<float> &result = predict();
  CHECK_GE(output_num_, 1);
  CHECK_GE(output_height_, 1);
  k = std::min(output_height_, k);

  vector<size_t> sorted_index = sort_to_index(result);
  return vector<int>(sorted_index.begin(), sorted_index.begin() + k);
}

} // namespace caffe
