# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/akarvin/cuda_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/akarvin/cuda_project

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/akarvin/cuda_project/CMakeFiles /home/akarvin/cuda_project//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/akarvin/cuda_project/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Sample

# Build rule for target.
Sample: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 Sample
.PHONY : Sample

# fast build rule for target.
Sample/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/build
.PHONY : Sample/fast

sample.o: sample.cu.o
.PHONY : sample.o

# target to build an object file
sample.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample.cu.o
.PHONY : sample.cu.o

sample.i: sample.cu.i
.PHONY : sample.i

# target to preprocess a source file
sample.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample.cu.i
.PHONY : sample.cu.i

sample.s: sample.cu.s
.PHONY : sample.s

# target to generate assembly for a file
sample.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample.cu.s
.PHONY : sample.cu.s

sample2.o: sample2.cu.o
.PHONY : sample2.o

# target to build an object file
sample2.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample2.cu.o
.PHONY : sample2.cu.o

sample2.i: sample2.cu.i
.PHONY : sample2.i

# target to preprocess a source file
sample2.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample2.cu.i
.PHONY : sample2.cu.i

sample2.s: sample2.cu.s
.PHONY : sample2.s

# target to generate assembly for a file
sample2.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Sample.dir/build.make CMakeFiles/Sample.dir/sample2.cu.s
.PHONY : sample2.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... Sample"
	@echo "... sample.o"
	@echo "... sample.i"
	@echo "... sample.s"
	@echo "... sample2.o"
	@echo "... sample2.i"
	@echo "... sample2.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

