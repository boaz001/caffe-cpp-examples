// The MIT License (MIT)
//
// Copyright (c) 2015 Boaz Stolk
//
// For full license view project root directory

// This program classifies two values using a network and a trained model
// Usage:
//  classify-xor NET MODEL VALUE1 VALUE2
//

#include <gflags/gflags.h>
#include <caffe/util/db.hpp>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include <algorithm>

int
main(int argc, char* argv[])
{
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Classifies two values using a network and a trained model\n"
                          "Usage:\n"
                          " classify-xor NET MODEL VALUE1 VALUE2\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 5)
  {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "classify-xor");
    return 1;
  }

  // get the net
  const std::string sNetwork = argv[1];
  std::cout << "loading " << sNetwork << std::endl;
  caffe::Net<float> caffe_test_net(sNetwork, caffe::TEST);

  // get trained model
  const std::string sModel = argv[2];
  std::cout << "loading " << sModel << std::endl;
  caffe_test_net.CopyTrainedLayersFrom(sModel);

  // read input values
  const int iVal1 = std::atoi(argv[3]);
  const int iVal2 = std::atoi(argv[4]);
  std::cout << "Input value 1: " << iVal1 << " Input value 2: " << iVal2 << std::endl;

  // two outputs; either 0 or 1
  const int iNumOfOutputs = 2;

  // feed input to the net
  // create input blob
  caffe::BlobProto blob_proto;
  caffe::BlobShape* shape = blob_proto.mutable_shape();
  shape->add_dim(1);
  shape->add_dim(iNumOfOutputs);
  blob_proto.add_data((float)iVal1);
  blob_proto.add_data((float)iVal2);
  std::vector<int> vShape;
  vShape.push_back(1);
  vShape.push_back(iNumOfOutputs);
  caffe::Blob<float> blob(vShape);

  // set data into blob
  blob.FromProto(blob_proto);

  // fill the bottom vector
  std::vector<caffe::Blob<float>*> bottom;
  bottom.push_back(&blob);

  // forward pass
  float loss = 0.0;
  const std::vector<caffe::Blob<float>*>& result = caffe_test_net.Forward(bottom, &loss);

  // find maximum
  float max = -1;
  int max_i = -1;
  for (int i = 0; i < iNumOfOutputs; ++i)
  {
    const float value = (result[0]->cpu_data())[i];
    std::cout << "index: " << i << " value: " << value << std::endl;
    if (value > max)
    {
      max = value;
      max_i = i;
    }
  }

  std::cout << "Result is: " << max_i << " classified    ";
  if ((iVal1^iVal2) == max_i)
  {
    std::cout << "GOOD" << std::endl;
  }
  else
  {
    std::cout << "BAD" << std::endl;
  }

  return 0;
}
