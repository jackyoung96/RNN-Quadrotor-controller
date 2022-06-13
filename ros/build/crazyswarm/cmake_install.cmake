# Install script for directory: /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/jack/crazyflie/sim-to-real-TD3/ros/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE PROGRAM FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/_setup_util.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE PROGRAM FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/env.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/setup.bash;/home/jack/crazyflie/sim-to-real-TD3/ros/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/setup.bash"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/local_setup.bash"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/setup.sh;/home/jack/crazyflie/sim-to-real-TD3/ros/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/setup.sh"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/local_setup.sh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/setup.zsh;/home/jack/crazyflie/sim-to-real-TD3/ros/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/setup.zsh"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/jack/crazyflie/sim-to-real-TD3/ros/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/jack/crazyflie/sim-to-real-TD3/ros/install" TYPE FILE FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/.rosinstall")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm/srv" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/GoTo.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/Land.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/NotifySetpointsStop.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/SetGroupMask.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/StartTrajectory.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/Stop.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/Takeoff.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/UpdateParams.srv"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/srv/UploadTrajectory.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm/msg" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/LogBlock.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/GenericLogData.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/FullState.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/VelocityWorld.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/TrajectoryPolynomialPiece.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/Hover.msg"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/msg/Position.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm/cmake" TYPE FILE FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/crazyswarm-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/include/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/share/roseus/ros/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/share/common-lisp/ros/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/share/gennodejs/ros/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/lib/python3/dist-packages/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/crazyswarm/lib/python3/dist-packages/crazyswarm")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/crazyswarm.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm/cmake" TYPE FILE FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/crazyswarm-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm/cmake" TYPE FILE FILES
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/crazyswarmConfig.cmake"
    "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/catkin_generated/installspace/crazyswarmConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/crazyswarm" TYPE FILE FILES "/home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/gtest/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
