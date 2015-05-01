// The MIT License (MIT)
//
// Copyright (c) 2015 Boaz Stolk
//
// For full license view project root directory

// This program generates random training data samples and puts it in
// a database as serialized datum proto buffers.
// Usage:
//  generate-random-shape-training-data [FLAGS] DB_NAME
//

#include <gflags/gflags.h>
#include <boost/scoped_ptr.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>

#include <string>
#include <vector>
#include <algorithm>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// define gflags FLAGS and default values
DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(split, 1, "Number of samples {nr} used for TRAIN before a sample is used for TEST, use negative value to do the inverse");
DEFINE_bool(shuffle, true, "Randomly shuffle the order of samples");
DEFINE_bool(balance, false, "Create a balanced set");

int
main(int argc, char* argv[])
{
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Generates random training data samples and puts it in\n"
                          "the leveldb/lmdb format used as input for Caffe.\n"
                          "Usage:\n"
                          " generate-random-shape-training-data [FLAGS] DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 2)
  {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "generate-random-shape-training-data");
    return 1;
  }

  // seed random generator
  std::srand(std::time(NULL));

  // generate random data
  typedef float tInput;
  typedef std::vector<tInput> tData;
  typedef int tLabel;
  typedef std::pair<tData, tLabel> tSample;
  typedef std::vector<tSample> tSamples;
  tSamples samples;

  { // create random input
    // generate image
    const int rows = 200;
    const int cols = 200;
    cv::Mat in_image_bgr = cv::Mat::zeros(rows, cols, CV_8UC3);

    // rectangle (red)
    for (int i = 0; i < 3; i++)
    {
      const int x1 = std::min(std::max(((float)std::rand() / RAND_MAX) * in_image_bgr.cols, (float)20), (float)rows-20);
      const int y1 = std::min(std::max(((float)std::rand() / RAND_MAX) * in_image_bgr.rows, (float)20), (float)cols-20);
      const int x2 = x1 + 8;
      const int y2 = y1 + 8;

      cv::rectangle(in_image_bgr,
        cv::Point(x1, y1),
        cv::Point(x2, y2),
        cv::Scalar(0, 0, 255),
        -1,
        8);
    }

    // circle (green)
    for (int i = 0; i < 3; i++)
    {
      const int x1 = std::min(std::max(((float)std::rand() / RAND_MAX) * in_image_bgr.cols, (float)20), (float)rows-20);
      const int y1 = std::min(std::max(((float)std::rand() / RAND_MAX) * in_image_bgr.rows, (float)20), (float)cols-20);
      cv::circle(in_image_bgr,
        cv::Point(x1, y1),
        5,
        cv::Scalar(0, 255, 0),
        -1,
        8);
    }

    cv::namedWindow("in_image_bgr", CV_WINDOW_AUTOSIZE);
    cv::moveWindow("in_image_bgr", 20, 20);
    cv::imshow("in_image_bgr", in_image_bgr);
    cv::waitKey(0);

    std::cout << "generating training data from image..." << std::endl;

    // generate training data from input image
    const int kernel = 15;
    const int h_kernel = kernel / 2;
    int iPatternCount = 0;
    for (int y = h_kernel; y < in_image_bgr.rows - h_kernel - 1; y++)
    {
      for (int x = h_kernel; x < in_image_bgr.cols - h_kernel - 1; x++)
      {
        const cv::Vec3b vec = in_image_bgr.at<cv::Vec3b>(y, x);
        if (vec[1] > 0) // circle
        {
          const tLabel label = 1;
          tData data;
          for (int yk = y - h_kernel; yk < y + h_kernel + 1; yk++)
          {
            for (int xk = x - h_kernel; xk < x + h_kernel + 1; xk++)
            {
              const cv::Vec3b veck = in_image_bgr.at<cv::Vec3b>(yk, xk);
              if (veck[1] > 0 || veck[2] > 0 || veck[0] > 0) // binarize
                data.push_back(1.0f);
              else
                data.push_back(0.0f);
            }
          }
          samples.push_back(std::make_pair(data, label));
        }
        else if (vec[2] > 0) // square
        {
          const tLabel label = 2;
          tData data;
          for (int yk = y - h_kernel; yk < y + h_kernel + 1; yk++)
          {
            for (int xk = x - h_kernel; xk < x + h_kernel + 1; xk++)
            {
              const cv::Vec3b veck = in_image_bgr.at<cv::Vec3b>(yk, xk);
              if (veck[1] > 0 || veck[2] > 0 || veck[0] > 0) // binarize
                data.push_back(1.0f);
              else
                data.push_back(0.0f);
            }
          }
          samples.push_back(std::make_pair(data, label));
        }
        else // background
        {
          // if balance is true, the background samples that are added are mostly around the
          // squares and circles and some (but not all) others
          if (FLAGS_balance == true) 
          {
            bool bHasNeighbourObject = false;
            const cv::Vec3b vec_h1 = in_image_bgr.at<cv::Vec3b>(y, x-1);
            const cv::Vec3b vec_h2 = in_image_bgr.at<cv::Vec3b>(y, x+1);
            const cv::Vec3b vec_v1 = in_image_bgr.at<cv::Vec3b>(y-1, x);
            const cv::Vec3b vec_v2 = in_image_bgr.at<cv::Vec3b>(y+1, x);
            if (vec_h1[1] > 0 || vec_h1[2] > 0 || vec_v1[1] > 0 || vec_v1[2] > 0 ||
                vec_h2[1] > 0 || vec_h2[2] > 0 || vec_v2[1] > 0 || vec_v2[2] > 0)
            {
              bHasNeighbourObject = true;
            }

            iPatternCount++;
            // only take a part of the background and when background is near objects
            if (bHasNeighbourObject == false && iPatternCount % 50 != 0)
              continue;
          }
          const tLabel label = 0;
          tData data;
          for (int yk = y - h_kernel; yk < y + h_kernel + 1; yk++)
          {
            for (int xk = x - h_kernel; xk < x + h_kernel + 1; xk++)
            {
              const cv::Vec3b veck = in_image_bgr.at<cv::Vec3b>(yk, xk);
              if (veck[1] > 0 || veck[2] > 0 || veck[0] > 0) // binarize
                data.push_back(1.0f);
              else
                data.push_back(0.0f);
            }
          }
          samples.push_back(std::make_pair(data, label));
        }
      }
    }
  }

  // count classes and number of occurences
  typedef std::map<int, int> tCounts;
  tCounts counts;
  for (tSamples::const_iterator itrSample = samples.begin()
                              ; itrSample != samples.end()
                              ; ++itrSample)
  {
    counts[itrSample->second]++;
  }

  // show counts
  for (tCounts::const_iterator itr = counts.begin()
                             ; itr != counts.end()
                             ; ++itr)
  {
    std::cout << "class: " << itr->first << " count: " << itr->second << std::endl;
  }

  // shuffle the data
  if (FLAGS_shuffle == true)
  {
    // randomly shuffle samples
    std::random_shuffle(samples.begin(), samples.end());
  }

  // Create new train and test DB
  boost::scoped_ptr<caffe::db::DB> train_db(caffe::db::GetDB(FLAGS_backend));
  std::string dbTrainName = argv[1];
  dbTrainName += "_train";
  train_db->Open(dbTrainName.c_str(), caffe::db::NEW);
  boost::scoped_ptr<caffe::db::Transaction> train_txn(train_db->NewTransaction());

  boost::scoped_ptr<caffe::db::DB> test_db(caffe::db::GetDB(FLAGS_backend));
  std::string dbTestName = argv[1];
  dbTestName += "_test";
  test_db->Open(dbTestName.c_str(), caffe::db::NEW);
  boost::scoped_ptr<caffe::db::Transaction> test_txn(test_db->NewTransaction());

  // divide the train/test data, determine spliting tactic
  const int iSplitRate = FLAGS_split;
  int iNumberPutToTrain = 0;
  int iNumberPutToTest = 0;
  enum eNextSample {eNSTrain, eNSTest};
  eNextSample nextSample = eNSTrain; // always start with train samples

  // convert samples to caffe::Datum
  int iCount = 0, iCountTrain = 0, iCountTest = 0;
  for (tSamples::const_iterator itrSample = samples.begin()
                              ; itrSample != samples.end()
                              ; ++itrSample)
  {
    // extract label from sample
    const int iLabel = itrSample->second;

    // convert sample to protobuf Datum
    caffe::Datum datum;
    datum.set_channels(itrSample->first.size());
    datum.set_height(1);
    datum.set_width(1);
    datum.set_label(iLabel);
    for (tData::const_iterator itrInputData = itrSample->first.begin()
                             ; itrInputData != itrSample->first.end()
                             ; ++itrInputData)
    {
      datum.add_float_data(*itrInputData);
    }

    // write datum to db use the sample number as key for db
    std::string out;
    CHECK(datum.SerializeToString(&out));
    std::stringstream ss;
    ss << iCount;

    // put sample
    if (nextSample == eNSTrain) // always start with train samples
    {
      train_txn->Put(ss.str(), out);
      iNumberPutToTrain++;
      iCountTrain++;
    }
    else if (nextSample == eNSTest)
    {
      test_txn->Put(ss.str(), out);
      iNumberPutToTest++;
      iCountTest++;
    }

    // determine where next sample should go
    if (iSplitRate == 0)
    {
      nextSample = eNSTrain;
    }
    else if (iSplitRate < 0)
    {
      if (iNumberPutToTest == std::abs(iSplitRate))
      {
        nextSample = eNSTrain;
        iNumberPutToTest = 0;
      }
      else
      {
        nextSample = eNSTest;
      }
    }
    else if (iSplitRate > 0)
    {
      if (iNumberPutToTrain == iSplitRate)
      {
        nextSample = eNSTest;
        iNumberPutToTrain = 0;
      }
      else
      {
        nextSample = eNSTrain;
      }
    }

    // every 1000 samples commit to db
    if (iCountTrain % 1000 == 0)
    {
      train_txn->Commit();
      train_txn.reset(train_db->NewTransaction());
    }
    if (iCountTest % 1000 == 0)
    {
      test_txn->Commit();
      test_txn.reset(test_db->NewTransaction());
    }

    iCount++;
  }

  // commit the last unwritten batch
  if (iCountTrain % 1000 != 0)
  {
    train_txn->Commit();
  }
  if (iCountTest % 1000 != 0)
  {
    test_txn->Commit();
  }

  std::cout << "Total of " << iCount << " samples generated, put " << iCountTrain << " to TRAIN DB and " << iCountTest << " to TEST DB" << std::endl;
  cv::waitKey(0);
  return 0;
}
