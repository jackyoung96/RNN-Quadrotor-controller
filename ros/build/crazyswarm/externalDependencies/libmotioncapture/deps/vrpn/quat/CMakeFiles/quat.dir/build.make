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
CMAKE_SOURCE_DIR = /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm

# Include any dependencies generated for this target.
include externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/depend.make

# Include the progress variables for this target.
include externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/progress.make

# Include the compile flags for this target's objects.
include externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/flags.make

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.o: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/flags.make
externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.o: /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/matrix.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.o"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/quat.dir/matrix.c.o   -c /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/matrix.c

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/quat.dir/matrix.c.i"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/matrix.c > CMakeFiles/quat.dir/matrix.c.i

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/quat.dir/matrix.c.s"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/matrix.c -o CMakeFiles/quat.dir/matrix.c.s

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.o: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/flags.make
externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.o: /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/quat.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.o"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/quat.dir/quat.c.o   -c /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/quat.c

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/quat.dir/quat.c.i"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/quat.c > CMakeFiles/quat.dir/quat.c.i

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/quat.dir/quat.c.s"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/quat.c -o CMakeFiles/quat.dir/quat.c.s

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.o: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/flags.make
externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.o: /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/vector.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.o"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/quat.dir/vector.c.o   -c /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/vector.c

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/quat.dir/vector.c.i"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/vector.c > CMakeFiles/quat.dir/vector.c.i

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/quat.dir/vector.c.s"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/vector.c -o CMakeFiles/quat.dir/vector.c.s

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.o: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/flags.make
externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.o: /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/xyzquat.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.o"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/quat.dir/xyzquat.c.o   -c /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/xyzquat.c

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/quat.dir/xyzquat.c.i"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/xyzquat.c > CMakeFiles/quat.dir/xyzquat.c.i

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/quat.dir/xyzquat.c.s"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/xyzquat.c -o CMakeFiles/quat.dir/xyzquat.c.s

# Object files for target quat
quat_OBJECTS = \
"CMakeFiles/quat.dir/matrix.c.o" \
"CMakeFiles/quat.dir/quat.c.o" \
"CMakeFiles/quat.dir/vector.c.o" \
"CMakeFiles/quat.dir/xyzquat.c.o"

# External object files for target quat
quat_EXTERNAL_OBJECTS =

externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/matrix.c.o
externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/quat.c.o
externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/vector.c.o
externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/xyzquat.c.o
externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/build.make
externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a: externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C static library libquat.a"
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && $(CMAKE_COMMAND) -P CMakeFiles/quat.dir/cmake_clean_target.cmake
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/quat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/build: externalDependencies/libmotioncapture/deps/vrpn/quat/libquat.a

.PHONY : externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/build

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/clean:
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat && $(CMAKE_COMMAND) -P CMakeFiles/quat.dir/cmake_clean.cmake
.PHONY : externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/clean

externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/depend:
	cd /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm /home/jack/crazyflie/sim-to-real-TD3/ros/src/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat /home/jack/crazyflie/sim-to-real-TD3/ros/build/crazyswarm/externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : externalDependencies/libmotioncapture/deps/vrpn/quat/CMakeFiles/quat.dir/depend
