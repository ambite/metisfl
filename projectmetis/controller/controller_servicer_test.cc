// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "projectmetis/controller/controller_mock.h"
#include "projectmetis/controller/controller_servicer.h"
#include "projectmetis/core/macros.h"
#include "projectmetis/core/matchers/proto_matchers.h"
#include "projectmetis/proto/controller.grpc.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {
namespace {
using ::grpc::ServerContext;
using ::proto::ParseTextOrDie;
using ::testing::Exactly;
using ::testing::Return;
using ::testing::proto::EqualsProto;
using ::testing::_;

const char kLearnerState[] = R"pb(
  learner {
    id: "localhost:1991"
    auth_token: "token"
    service_spec {
      hostname: "localhost"
      port: 1991
    }
    dataset_spec {
      num_training_examples: 1
      num_validation_examples: 1
      num_test_examples: 1
    }
  }
)pb";

class ControllerServicerImplTest : public ::testing::Test {
 protected:
  ServerContext ctx_;

  MockController controller_;
  std::unique_ptr<ControllerServicer> service_ =
      ControllerServicer::New(&controller_);
};

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_EmptyRequest) {
  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;

  EXPECT_CALL(controller_, GetLearners())
        .Times(Exactly(1))
        .WillOnce(Return(std::vector<LearnerDescriptor>()));

  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_EmptyVector) {
  EXPECT_CALL(controller_, GetLearners())
      .Times(Exactly(1))
      .WillOnce(Return(std::vector<LearnerDescriptor>()));

  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;
  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res_.server_entity().empty());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_NotEmptyVector) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto &learner = learner_state.learner();

  EXPECT_CALL(controller_, GetLearners())
      .Times(Exactly(1))
      .WillOnce(Return(std::vector({learner})));

  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;
  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(res_.server_entity_size(), 1);
  EXPECT_EQ(res_.server_entity(0).hostname(), "localhost");
  EXPECT_EQ(res_.server_entity(0).port(), 1991);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_EmptyRequest) {
  JoinFederationRequest req_;
  JoinFederationResponse res_;
  auto status = service_->JoinFederation(&ctx_, &req_, &res_);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_NewLearner) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto& learner = learner_state.learner();

  EXPECT_CALL(controller_,
              AddLearner(EqualsProto(learner_state.learner().service_spec()),
                         EqualsProto(learner_state.learner().dataset_spec())))
      .Times(Exactly(1))
      .WillOnce(Return(learner));

  JoinFederationRequest req_;
  JoinFederationResponse res_;
  *req_.mutable_server_entity() = learner_state.learner().service_spec();
  *req_.mutable_local_dataset_spec() = learner_state.learner().dataset_spec();

  auto status = service_->JoinFederation(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(res_.learner_id().empty());
  EXPECT_FALSE(res_.auth_token().empty());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_LearnerServiceUnreachable) {
  // TODO(canastas): Implement this after we have the service alive check functionality.
  bool is_learner_reachable = true;
  EXPECT_TRUE(is_learner_reachable);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_LearnerCollision) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto& learner = learner_state.learner();

  EXPECT_CALL(controller_,
              AddLearner(EqualsProto(learner.service_spec()),
                         EqualsProto(learner.dataset_spec())))
      .Times(Exactly(2))
      .WillOnce(Return(learner))
      .WillOnce(Return(absl::AlreadyExistsError("Learner has already joined.")));

  JoinFederationRequest req_;
  JoinFederationResponse res_;
  *req_.mutable_server_entity() = learner.service_spec();
  *req_.mutable_local_dataset_spec() = learner.dataset_spec();

  // First time, learner joins successfully.
  service_->JoinFederation(&ctx_, &req_, &res_);

  // Second time, must return and AlreadyExists error.
  auto status = service_->JoinFederation(&ctx_, &req_, &res_);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_EmptyRequest) {
  LeaveFederationRequest req_;
  LeaveFederationResponse res_;
  auto status = service_->LeaveFederation(&ctx_, &req_, &res_);

  EXPECT_FALSE(status.ok());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerExists) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto& learner = learner_state.learner();

  EXPECT_CALL(controller_,RemoveLearner(learner.id(), learner.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::OkStatus()));

  LeaveFederationRequest req;
  req.set_auth_token(learner.auth_token());
  req.set_learner_id(learner.id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res.ack().status());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerNotExists) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto& learner = learner_state.learner();

  EXPECT_CALL(controller_,RemoveLearner(learner.id(), learner.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::NotFoundError("No such learner.")));

  LeaveFederationRequest req;
  req.set_auth_token(learner.auth_token());
  req.set_learner_id(learner.id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerInvalidCredentials) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);
  const auto& learner = learner_state.learner();

  EXPECT_CALL(controller_,RemoveLearner(learner.id(), learner.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::UnauthenticatedError("Incorrect token.")));

  LeaveFederationRequest req;
  req.set_auth_token(learner.auth_token());
  req.set_learner_id(learner.id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetEvaluationLineage_RequestIsNullptr){

  GetLocalModelEvaluationLineageRequest request;
  GetLocalModelEvaluationLineageResponse response;

  auto status = service_->GetLocalModelEvaluationLineage(&ctx_, nullptr, &response);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// TODO(aasghar) Need to make the tests more robust with respect to ModelEvaluation values.
TEST_F(ControllerServicerImplTest, GetEvaluationLineage_EvaluationLineage) {

  LearnerState learnerState = ParseTextOrDie<LearnerState>(kLearnerState);
  const LearnerDescriptor &learnerDescriptor = learnerState.learner();
  std::string learner_id = learnerDescriptor.id();
  GetLocalModelEvaluationLineageRequest request;
  GetLocalModelEvaluationLineageResponse response;

  // We are testing without initializing the request object with wildcard values defined by ::testing::_
  ON_CALL(controller_, GetEvaluationLineage(learner_id, ::testing::_))
    .WillByDefault(Return(std::vector<ModelEvaluation>()));

  auto status = service_->GetLocalModelEvaluationLineage(&ctx_, &request, &response);

  EXPECT_EQ(status.error_code(),grpc::StatusCode::OK);
  EXPECT_TRUE((*response.mutable_learner_evaluations()).empty());
}

// TODO(aasghar) Could make the tests more robust with respect to ModelEvaluation values.
TEST_F(ControllerServicerImplTest, GetCommunityEvaluationLineage_EvaluationLineage) {


  GetCommunityModelEvaluationLineageRequest request;
  GetCommunityModelEvaluationLineageResponse response;

  request.set_num_backtracks(1);
  // We are testing without initializing the request object. Passing wildcard values defined by ::testing::_
  EXPECT_CALL(controller_, GetEvaluationLineage(::testing::_))
        .Times(Exactly(1))
        .WillOnce(Return(std::vector<ModelEvaluation>()));

  auto status = service_->GetCommunityModelEvaluationLineage(&ctx_, &request, &response);

  EXPECT_EQ(status.error_code(),grpc::StatusCode::OK);
  ASSERT_TRUE((*response.mutable_evaluations()).IsInitialized());
}

// TODO(aasghar) Can make the test more useful by initializing the request object.
TEST_F(ControllerServicerImplTest, MarkTaskCompleted_ReturnOk){

  MarkTaskCompletedRequest request;
  MarkTaskCompletedResponse response;

  // We are testing without initializing the request object. Passing wildcard values defined by ::testing::_
  ON_CALL(controller_, LearnerCompletedTask(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(Return(absl::OkStatus()));

  auto status = service_->MarkTaskCompleted(&ctx_, &request, &response);

  EXPECT_TRUE(status.ok());

}

// TODO(aasghar) Can make the test more useful by initializing the request object.
TEST_F(ControllerServicerImplTest, MarkTaskCompleted_ReturnError){

  MarkTaskCompletedRequest request;
  MarkTaskCompletedResponse response;

  // We are testing without initializing the request object. Passing wildcard values defined by ::testing::_
  ON_CALL(controller_, LearnerCompletedTask(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(Return(absl::NotFoundError("Learner does not exist.")));

  auto status = service_->MarkTaskCompleted(&ctx_, &request, &response);

  EXPECT_FALSE(status.ok());

}
TEST_F(ControllerServicerImplTest, StartService_WithSSL){

  projectmetis::ControllerParams params = ParseTextOrDie<projectmetis::ControllerParams>(R"pb2(
    server_entity {
      hostname: "0.0.0.0"
      port: 50051
      ssl_config {
            server_key: "/resources/ssl/server-key.pem"
            server_cert: "/resources/ssl/server-cert.pem"
        }
    }
    global_model_specs {
      learners_participation_ratio: 1
      aggregation_rule: FED_AVG
    }
    communication_specs {
      protocol: SYNCHRONOUS
    }
    model_hyperparams {
      batch_size: 1
      epochs: 1
      optimizer {
        vanilla_sgd {
          learning_rate: 0.05
          L2_reg: 0.001
        }
      }
      percent_validation: 0
    }
    )pb2");

  EXPECT_CALL(controller_, GetParams)
            .Times(Exactly(2))
            .WillRepeatedly(::testing::ReturnRef(params));

  service_->StartService();
  bool is_enabled = service_->GetController()->GetParams().server_entity().has_ssl_config();

  ASSERT_TRUE(is_enabled);
}

TEST_F(ControllerServicerImplTest, StartService_WithoutSSL){

  projectmetis::ControllerParams params = ParseTextOrDie<projectmetis::ControllerParams>(R"pb2(
    server_entity {
      hostname: "0.0.0.0"
      port: 50051
    }
    global_model_specs {
      learners_participation_ratio: 1
      aggregation_rule: FED_AVG
    }
    communication_specs {
      protocol: SYNCHRONOUS
    }
    model_hyperparams {
      batch_size: 1
      epochs: 1
      optimizer {
        vanilla_sgd {
          learning_rate: 0.05
          L2_reg: 0.001
        }
      }
      percent_validation: 0
    }
  )pb2");

  EXPECT_CALL(controller_, GetParams)
            .Times(Exactly(2))
            .WillRepeatedly(::testing::ReturnRef(params));

  service_->StartService();
  bool is_enabled = service_->GetController()->GetParams().server_entity().has_ssl_config();

  ASSERT_FALSE(is_enabled);
}

} // namespace
} // namespace projectmetis::controller
