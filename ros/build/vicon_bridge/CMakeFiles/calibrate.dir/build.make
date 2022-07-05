# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge

# Include any dependencies generated for this target.
include CMakeFiles/calibrate.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calibrate.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calibrate.dir/flags.make

CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o: CMakeFiles/calibrate.dir/flags.make
CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o: /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge/src/calibrate_segment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o -c /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge/src/calibrate_segment.cpp

CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge/src/calibrate_segment.cpp > CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.i

CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge/src/calibrate_segment.cpp -o CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.s

# Object files for target calibrate
calibrate_OBJECTS = \
"CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o"

# External object files for target calibrate
calibrate_EXTERNAL_OBJECTS =

/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: CMakeFiles/calibrate.dir/src/calibrate_segment.cpp.o
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: CMakeFiles/calibrate.dir/build.make
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libdynamic_reconfigure_config_init_mutex.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libtf.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libtf2_ros.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libactionlib.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libmessage_filters.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libtf2.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libdiagnostic_updater.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libroscpp.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/librosconsole.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libxmlrpcpp.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libroscpp_serialization.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/librostime.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /opt/ros/noetic/lib/libcpp_common.so
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate: CMakeFiles/calibrate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calibrate.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calibrate.dir/build: /home/jack/crazyflie/sim-to-real-TD3/ros/devel/.private/vicon_bridge/lib/vicon_bridge/calibrate

.PHONY : CMakeFiles/calibrate.dir/build

CMakeFiles/calibrate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calibrate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calibrate.dir/clean

CMakeFiles/calibrate.dir/depend:
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge /home/jack/crazyflie/sim-to-real-TD3/ros/src/vicon_bridge /home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge /home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge /home/jack/crazyflie/sim-to-real-TD3/ros/build/vicon_bridge/CMakeFiles/calibrate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calibrate.dir/depend
