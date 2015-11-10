#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
#include "caffe/caffe.hpp"

using std::string;

namespace caffe {

class CaffeMobile
{
public:
  CaffeMobile(string model_path, string weights_path);
  ~CaffeMobile();

  vector<float> predict();

  vector<int> predict_top_k(const string &img_path, int k = 3);

  int set_images(const vector<string> &img_paths);

  inline int output_height() { return output_height_; }
  inline int output_num() { return output_num_; }
  inline double test_time() { return test_time_; } // unit: second

protected:
  Net<float> *caffe_net_;
  int output_height_, output_num_;
  double test_time_;
};

} // namespace caffe

#endif
