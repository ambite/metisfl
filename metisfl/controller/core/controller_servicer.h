
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/proto/controller.grpc.pb.h"

namespace metisfl::controller {
class ControllerServicer : public ControllerService::Service {
 public:
  virtual const Controller *GetController() const = 0;

  virtual void StartService() = 0;

  virtual void WaitService() = 0;

  virtual void StopService() = 0;

  virtual bool ShutdownRequestReceived() = 0;

  static std::unique_ptr<ControllerServicer> New(ServerParams &server_params,
                                                 Controller *controller);

 private:
  std::unique_ptr<Server> server_;
  ServerParams server_params_;
  BS::thread_pool pool_;
  Controller *controller_;
  bool shutdown_ = false;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
