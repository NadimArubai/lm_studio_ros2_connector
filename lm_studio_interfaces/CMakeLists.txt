cmake_minimum_required(VERSION 3.8)
project(lm_studio_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ChatCompletion.action"
  "action/TextCompletion.action"
  "srv/ListModels.srv"
  "srv/GetEmbeddings.srv"
  "srv/SetModel.srv"
  "srv/ResetChat.srv"
  DEPENDENCIES
  action_msgs
  std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(action_msgs)
ament_export_dependencies(std_msgs)

ament_package()