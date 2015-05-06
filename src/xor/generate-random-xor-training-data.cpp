// The MIT License (MIT)
//
// Copyright (c) 2015 Boaz Stolk
//
// For full license view project root directory

// This program generates random training data samples and puts it in
// a database as serialized datum proto buffers.
// Usage:
//  generate-random-xor-training-data [FLAGS] NR_OF_SAMPLES DB_NAME
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

// define gflags FLAGS and default values
DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(split, 1, "Number of samples {nr} used for TRAIN before a sample is used for TEST, use negative value to do the opposite");
DEFINE_bool(shuffle, true, "Randomly shuffle the order of samples");

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
                          " generate-random-xor-training-data [FLAGS] NR_OF_SAMPLES DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 3)
  {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "generate-random-xor-training-data");
    return 1;
  }

  // seed random generator
  std::srand(std::time(NULL));

  const int iNumberOfSamples = std::atoi(argv[1]);

  // generate random data
  typedef int tInput;
  typedef std::pair<tInput, tInput> tData;
  typedef int tLabel;
  typedef std::pair<tData, tLabel> tSample;
  typedef std::vector<tSample> tSamples;
  tSamples samples;
  {
    for (int i = 0; i < iNumberOfSamples; i++)
    {
      const double dRandom1 = (double)std::rand() / RAND_MAX;
      const double dRandom2 = (double)std::rand() / RAND_MAX;
      const tData data = std::make_pair(dRandom1 > 0.5 ? 1 : 0, dRandom2 > 0.5 ? 1 : 0);
      // XOR data to get label
      const tLabel label = data.first ^ data.second;
      const tSample sample = std::make_pair(data, label);
      samples.push_back(sample);
    }
  }

  // shuffle the data
  if (FLAGS_shuffle == true)
  {
    // randomly shuffle samples
    std::random_shuffle(samples.begin(), samples.end());
  }

  // Create new train and test DB
  boost::scoped_ptr<caffe::db::DB> train_db(caffe::db::GetDB(FLAGS_backend));
  std::string dbTrainName = argv[2];
  dbTrainName += "_train";
  train_db->Open(dbTrainName.c_str(), caffe::db::NEW);
  boost::scoped_ptr<caffe::db::Transaction> train_txn(train_db->NewTransaction());

  boost::scoped_ptr<caffe::db::DB> test_db(caffe::db::GetDB(FLAGS_backend));
  std::string dbTestName = argv[2];
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

    // convert data to protobuf Datum
    caffe::Datum datum;
    datum.set_channels(2);
    datum.set_height(1);
    datum.set_width(1);
    datum.set_label(iLabel);
    datum.add_float_data(itrSample->first.first);
    datum.add_float_data(itrSample->first.second);

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
  return 0;
}
